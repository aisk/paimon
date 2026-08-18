import json
import multiprocessing
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from paimon.config import Config, ConfigError, config_path


def _hammer_save(config_home: str, field: str, rounds: int) -> None:
    """Child-process worker: flip one config field over and over."""
    os.environ["PAIMON_CONFIG_HOME"] = config_home
    from paimon.config import Config
    for i in range(rounds):
        Config.load().save(**{field: f"{field}-{i}"})


def _hold_config_lock(lock_path: str, ready, stop) -> None:
    """Child-process worker: hold a config sidecar lock until told to stop."""
    from paimon import lockfile
    path = Path(lock_path)
    lockfile.acquire(path)
    try:
        ready.set()
        stop.wait(10)
    finally:
        lockfile.release(path)


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
        self.assertEqual(data["providers"], {"test": {"api_key": "sk-old"}})
        self.assertEqual(config.provider_auth(), (None, "sk-old"))
        self.assertEqual(Config.load().provider_auth(), (None, "sk-old"))

    def test_unrelated_save_keeps_a_runtime_override(self) -> None:
        """paimon --model X assigns the field on the live instance; persisting a
        theme from the TUI must not swap the running agent back to the stored model."""
        config = Config.load()
        config.save(model="test:stored", api_key="sk-stored")
        config.model = "test:override"
        config.save(theme="nord")

        self.assertEqual(config.model, "test:override")
        self.assertEqual(config.theme, "nord")
        self.assertEqual(config.provider_auth(), (None, "sk-stored"))
        stored = json.loads(config_path().read_text())
        self.assertEqual(stored["model"], "test:stored", "the override is not persisted")
        self.assertEqual(stored["theme"], "nord")

    def test_save_empty_string_clears_too(self) -> None:
        config = Config.load()
        config.save(model="test:m", api_base="https://old/v1")
        config.save(api_base="")
        self.assertNotIn("providers", json.loads(config_path().read_text()))

    def test_credentials_are_scoped_to_their_provider(self) -> None:
        """AUTH-1: each provider keeps its own key; switching models must not
        overwrite or reuse another provider's credentials."""
        config = Config.load()
        config.save(model="zai:glm-5.2", api_key="sk-zai", api_base="https://z/v4")
        config.save(model="deepseek:deepseek-chat", api_key="sk-deep")

        self.assertEqual(config.provider_auth(), (None, "sk-deep"))
        self.assertEqual(config.provider_auth("zai:glm-5.2"), ("https://z/v4", "sk-zai"))
        self.assertEqual(config.provider_auth("deepseek:other-model"), (None, "sk-deep"))
        self.assertEqual(config.provider_auth("openai:gpt-5"), (None, None))
        stored = json.loads(config_path().read_text())["providers"]
        self.assertEqual(stored, {"zai": {"api_key": "sk-zai", "api_base": "https://z/v4"},
                                  "deepseek": {"api_key": "sk-deep"}})

    def test_saving_credentials_without_a_model_is_refused(self) -> None:
        with self.assertRaises(ConfigError):
            Config.load().save(api_key="sk-nowhere")
        self.assertFalse(config_path().exists() and
                         "sk-nowhere" in config_path().read_text())

    def test_provider_auth_without_a_model_is_empty(self) -> None:
        self.assertEqual(Config.load().provider_auth(), (None, None))
        self.assertEqual(Config.load().provider_auth("unqualified"), (None, None))

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


class ConfigDurableWriteTest(unittest.TestCase):
    """CFG-2: concurrent writers and dying writers must not lose fields."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        env = patch.dict("os.environ", {"PAIMON_CONFIG_HOME": tmp.name})
        env.start()
        self.addCleanup(env.stop)

    def test_save_merges_fields_written_by_another_process(self) -> None:
        """Another Paimon's save must survive this instance's later save.

        The lost update this guards against is load, someone else writes,
        save, where the old read-modify-write shipped the stale snapshot
        from load time.
        """
        config = Config.load()
        config.save(model="test:m")
        # A second process saved between our load and our save.
        path = config_path()
        data = json.loads(path.read_text())
        data["api_key"] = "sk-other"
        path.write_text(json.dumps(data))

        config.save(theme="nord")

        stored = json.loads(path.read_text())
        self.assertEqual(stored, {"model": "test:m", "api_key": "sk-other", "theme": "nord"})

    def test_concurrent_processes_do_not_lose_fields(self) -> None:
        """Real processes hammering one profile keep every writer's field."""
        config = Config.load()
        config.save(model="test:base")
        home = os.environ["PAIMON_CONFIG_HOME"]

        ctx = multiprocessing.get_context("spawn")
        workers = [ctx.Process(target=_hammer_save, args=(home, field, 6))
                   for field in ("theme", "api_base", "show_reasoning")]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(30)
            self.assertEqual(worker.exitcode, 0)

        stored = json.loads(config_path().read_text())
        self.assertEqual(stored["model"], "test:base")
        for field in ("theme", "show_reasoning"):
            self.assertTrue(str(stored[field]).startswith(f"{field}-"), stored)
        self.assertTrue(stored["providers"]["test"]["api_base"].startswith("api_base-"), stored)

    def test_interrupted_write_leaves_previous_config_intact(self) -> None:
        config = Config.load()
        config.save(model="test:old", api_key="sk-old")

        real_replace = os.replace

        def failing_replace(src, dst, **kwargs):
            if str(dst).endswith("config.json"):
                raise OSError("disk went away")
            return real_replace(src, dst, **kwargs)

        with patch("os.replace", side_effect=failing_replace):
            with self.assertRaises(ConfigError):
                config.save(model="test:new")

        # The old config is complete, and the temp file was cleaned up.
        stored = json.loads(config_path().read_text())
        self.assertEqual(stored["model"], "test:old")
        self.assertEqual(stored["providers"], {"test": {"api_key": "sk-old"}})
        leftovers = [p.name for p in config_path().parent.iterdir() if ".tmp" in p.name]
        self.assertEqual(leftovers, [])

    def test_short_writes_are_looped_to_completion(self) -> None:
        """os.write returning early must not truncate the payload."""
        config = Config.load()
        real_write = os.write

        def trickling_write(fd, data):
            chunk = bytes(data)
            return real_write(fd, chunk[: max(1, len(chunk) // 3)])

        with patch("os.write", side_effect=trickling_write):
            config.save(model="test:m", api_key="sk-x")

        stored = json.loads(config_path().read_text())
        self.assertEqual(stored, {"model": "test:m", "providers": {"test": {"api_key": "sk-x"}}})

    def test_save_writes_through_a_symlinked_config(self) -> None:
        """A config.json symlinked from a dotfiles repo keeps being a link."""
        path = config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        target = path.with_name("dotfiles-config.json")
        target.write_text("{}")
        path.symlink_to(target)

        Config.load().save(model="test:m")

        self.assertTrue(path.is_symlink())
        self.assertEqual(json.loads(target.read_text())["model"], "test:m")

    def test_corrupt_config_is_reported_and_preserved(self) -> None:
        path = config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        torn = '{"model": "test:m", "api_key"'
        path.write_text(torn)

        with self.assertRaises(ConfigError):
            Config.load()
        # A save must refuse to write over the remains with {} plus one field.
        with self.assertRaises(ConfigError):
            Config(profile="default").save(theme="nord")
        self.assertEqual(path.read_text(), torn)

    def test_non_object_config_is_reported(self) -> None:
        path = config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("[]")
        with self.assertRaises(ConfigError):
            Config.load()

    def test_save_gives_up_when_the_lock_stays_held(self) -> None:
        path = config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = path.with_name(path.name + ".lock")
        ctx = multiprocessing.get_context("spawn")
        ready, stop = ctx.Event(), ctx.Event()
        holder = ctx.Process(target=_hold_config_lock, args=(str(lock), ready, stop))
        holder.start()
        self.addCleanup(stop.set)
        self.addCleanup(holder.join, 10)
        self.assertTrue(ready.wait(5), "lock holder never started")

        with patch("paimon.config._SAVE_LOCK_TIMEOUT", 0.3):
            with self.assertRaises(ConfigError):
                Config.load().save(model="test:m")

        stop.set()
        holder.join(10)
        # With the holder gone the save goes through.
        Config.load().save(model="test:m")
        self.assertEqual(json.loads(path.read_text())["model"], "test:m")


if __name__ == "__main__":
    unittest.main()
