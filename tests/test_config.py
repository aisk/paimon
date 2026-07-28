import json
import tempfile
import unittest
from unittest.mock import patch

from paimon.config import Config, config_path


class ConfigProfileTest(unittest.TestCase):
    """load/save are bound to the instance's profile, not any global state."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        env = patch.dict("os.environ", {"PAIMON_CONFIG_HOME": tmp.name})
        env.start()
        self.addCleanup(env.stop)

    def test_instances_on_different_profiles_do_not_interfere(self) -> None:
        work = Config.load("work")
        default = Config.load()
        work.save(model="test:work")
        default.save(model="test:default")

        self.assertEqual(Config.load("work").model, "test:work")
        self.assertEqual(Config.load().model, "test:default")
        self.assertEqual(json.loads(config_path("work").read_text())["model"], "test:work")

    def test_load_records_the_profile_on_the_instance(self) -> None:
        self.assertEqual(Config.load().profile, "default")
        self.assertEqual(Config.load("work").profile, "work")

    def test_invalid_profile_name_raises(self) -> None:
        with self.assertRaises(ValueError):
            Config.load("../evil")


if __name__ == "__main__":
    unittest.main()
