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

    def test_save_none_clears_while_unpassed_fields_keep_their_values(self) -> None:
        config = Config.load()
        config.save(model="test:m", api_base="https://old/v1", api_key="sk-old")
        config.save(model="test:m2", api_base=None)

        data = json.loads(config_path().read_text())
        self.assertEqual(data["model"], "test:m2")
        self.assertNotIn("api_base", data)
        self.assertEqual(data["api_key"], "sk-old")
        self.assertIsNone(config.api_base)
        self.assertEqual(Config.load().api_key, "sk-old")

    def test_unrelated_save_keeps_a_runtime_override(self) -> None:
        """paimon --model X assigns the field on the live instance; persisting a
        theme from the TUI must not swap the running agent back to the stored model."""
        config = Config.load()
        config.save(model="test:stored", api_key="sk-stored")
        config.model = "test:override"
        config.save(theme="nord")

        self.assertEqual(config.model, "test:override")
        self.assertEqual(config.theme, "nord")
        self.assertEqual(config.api_key, "sk-stored")
        stored = json.loads(config_path().read_text())
        self.assertEqual(stored["model"], "test:stored", "the override is not persisted")
        self.assertEqual(stored["theme"], "nord")

    def test_save_empty_string_clears_too(self) -> None:
        config = Config.load()
        config.save(api_base="https://old/v1")
        config.save(api_base="")
        self.assertNotIn("api_base", json.loads(config_path().read_text()))

    def test_save_persists_show_reasoning_false(self) -> None:
        config = Config.load()
        config.save(show_reasoning=True)
        config.save(show_reasoning=False)
        self.assertFalse(Config.load().show_reasoning)

    def test_save_persists_recap_enabled_false(self) -> None:
        config = Config.load()
        config.save(recap_enabled=False)
        self.assertFalse(config.recap_enabled)
        self.assertFalse(Config.load().recap_enabled)

    def test_load_records_the_profile_on_the_instance(self) -> None:
        self.assertEqual(Config.load().profile, "default")
        self.assertEqual(Config.load("work").profile, "work")

    def test_invalid_profile_name_raises(self) -> None:
        with self.assertRaises(ValueError):
            Config.load("../evil")

    def test_safe_commands_defaults_on(self) -> None:
        self.assertTrue(Config.load().safe_commands)

    def test_safe_commands_loaded_and_preserved(self) -> None:
        path = config_path("default")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"safe_commands": False}))

        config = Config.load()
        self.assertFalse(config.safe_commands)
        config.save(model="test:stub")
        self.assertFalse(json.loads(path.read_text())["safe_commands"])


if __name__ == "__main__":
    unittest.main()
