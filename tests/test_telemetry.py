import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from paimon import telemetry


class TelemetryTestCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name)
        env = patch.dict("os.environ", {"PAIMON_CONFIG_HOME": tmp.name,
                                        "PAIMON_NO_TELEMETRY": "",
                                        "DO_NOT_TRACK": ""})
        env.start()
        self.addCleanup(env.stop)

    def _state(self) -> dict:
        return json.loads((self.home / "telemetry.json").read_text(encoding="utf-8"))


class PrepareTest(TelemetryTestCase):
    def test_first_launch_mints_a_client_id_and_marks_first_visit(self) -> None:
        params = telemetry._prepare("tui")
        self.assertEqual(params["_fv"], "1")
        self.assertEqual(params["en"], "app_start")
        self.assertEqual(params["ep.mode"], "tui")
        self.assertEqual(params["tid"], telemetry._MEASUREMENT_ID)
        self.assertEqual(params["cid"], self._state()["client_id"])

    def test_later_launches_keep_the_id_and_count_sessions(self) -> None:
        first = telemetry._prepare("tui")
        second = telemetry._prepare("headless")
        self.assertEqual(first["cid"], second["cid"])
        self.assertNotIn("_fv", second)
        self.assertEqual(first["sct"], "1")
        self.assertEqual(second["sct"], "2")
        self.assertEqual(self._state()["session_count"], 2)

    def test_model_is_split_into_provider_and_model(self) -> None:
        params = telemetry._prepare("tui", model="zai:glm-4.7")
        self.assertEqual(params["ep.model"], "zai:glm-4.7")
        self.assertEqual(params["up.provider"], "zai")

    def test_unqualified_model_sends_no_provider(self) -> None:
        params = telemetry._prepare("tui", model="glm-4.7")
        self.assertEqual(params["ep.model"], "glm-4.7")
        self.assertNotIn("up.provider", params)

    def test_no_model_sends_neither(self) -> None:
        params = telemetry._prepare("tui")
        self.assertNotIn("ep.model", params)
        self.assertNotIn("up.provider", params)

    def test_damaged_state_is_regenerated(self) -> None:
        (self.home / "telemetry.json").write_text("not json", encoding="utf-8")
        params = telemetry._prepare("tui")
        self.assertEqual(params["_fv"], "1")
        self.assertEqual(params["cid"], self._state()["client_id"])


class OptOutTest(TelemetryTestCase):
    def test_env_vars_disable_everything(self) -> None:
        for var in ("PAIMON_NO_TELEMETRY", "DO_NOT_TRACK"):
            with patch.dict("os.environ", {var: "1"}):
                self.assertIsNone(telemetry._prepare("tui"))
                self.assertFalse((self.home / "telemetry.json").exists())

    def test_zero_does_not_opt_out(self) -> None:
        with patch.dict("os.environ", {"DO_NOT_TRACK": "0"}):
            self.assertTrue(telemetry.enabled())


class RecordLaunchTest(TelemetryTestCase):
    def test_sends_in_the_background(self) -> None:
        sent = []
        with patch.object(telemetry, "_send", sent.append), \
                patch.object(telemetry.threading, "Thread",
                             lambda *, target, args, **kw: type(
                                 "T", (), {"start": lambda self: target(*args)})()):
            telemetry.record_launch("status")
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0]["ep.mode"], "status")

    def test_never_raises(self) -> None:
        with patch.object(telemetry, "_prepare", side_effect=OSError("disk")):
            telemetry.record_launch("tui")
