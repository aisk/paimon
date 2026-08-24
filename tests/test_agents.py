"""Agent type loading, discovery and the dynamic spawn_agent schema."""

import tempfile
import unittest
from pathlib import Path

from paimon import tools
from paimon.agents import (
    builtin_types,
    discover_agent_types,
    find_type,
    load_agent_type,
    load_agent_types_from_dir,
    spawn_tool_with_types,
)


def write_agent_type(directory: Path, name: str, description: str = "Does things.",
                     body: str = "Be helpful.", filename: str | None = None,
                     **extra) -> Path:
    """One agent type file; ``extra`` keys land in the frontmatter as-is
    (underscores become hyphens, matching the skills helper convention)."""
    directory.mkdir(parents=True, exist_ok=True)
    lines = ["---", f"name: {name}", f"description: {description}"]
    for key, value in extra.items():
        lines.append(f"{key.replace('_', '-')}: {value}")
    lines += ["---", body]
    path = directory / (filename or f"{name}.md")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


class LoadAgentTypeTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def test_frontmatter_fields_and_body_are_loaded(self) -> None:
        path = write_agent_type(self.dir, "scout", description="Explores code.",
                                body="You are a scout.", tools="read_file, grep",
                                model="test:stub")
        agent_type, diagnostics = load_agent_type(path)
        self.assertEqual(diagnostics, [])
        self.assertEqual(agent_type.name, "scout")
        self.assertEqual(agent_type.description, "Explores code.")
        self.assertEqual(agent_type.body, "You are a scout.")
        self.assertEqual(agent_type.tools, ("read_file", "grep"))
        self.assertEqual(agent_type.model, "test:stub")
        self.assertEqual(agent_type.path, path)

    def test_tools_accepts_a_yaml_list_too(self) -> None:
        path = self.dir / "scout.md"
        path.write_text("---\ndescription: X.\ntools:\n  - read_file\n  - glob\n---\nBody",
                        encoding="utf-8")
        agent_type, diagnostics = load_agent_type(path)
        self.assertEqual(diagnostics, [])
        self.assertEqual(agent_type.tools, ("read_file", "glob"))

    def test_the_name_falls_back_to_the_file_stem(self) -> None:
        path = self.dir / "helper.md"
        path.write_text("---\ndescription: X.\n---\nBody", encoding="utf-8")
        agent_type, _ = load_agent_type(path)
        self.assertEqual(agent_type.name, "helper")

    def test_a_missing_description_is_a_diagnostic_and_no_type(self) -> None:
        path = self.dir / "bad.md"
        path.write_text("---\nname: bad\n---\nBody", encoding="utf-8")
        agent_type, diagnostics = load_agent_type(path)
        self.assertIsNone(agent_type)
        self.assertIn("description is required", diagnostics[0].message)

    def test_a_bad_name_is_reported_but_the_type_still_loads(self) -> None:
        path = write_agent_type(self.dir, "Bad_Name", filename="x.md")
        agent_type, diagnostics = load_agent_type(path)
        self.assertIsNotNone(agent_type)
        self.assertTrue(any("invalid characters" in d.message for d in diagnostics))

    def test_an_unknown_tool_is_reported_but_kept(self) -> None:
        path = write_agent_type(self.dir, "scout", tools="read_file, teleport")
        agent_type, diagnostics = load_agent_type(path)
        self.assertEqual(agent_type.tools, ("read_file", "teleport"))
        self.assertTrue(any('unknown tool "teleport"' in d.message for d in diagnostics))

    def test_malformed_yaml_is_a_diagnostic(self) -> None:
        path = self.dir / "bad.md"
        path.write_text("---\ndescription: [unclosed\n---\nBody", encoding="utf-8")
        agent_type, diagnostics = load_agent_type(path)
        self.assertIsNone(agent_type)
        self.assertIn("invalid frontmatter", diagnostics[0].message)

    def test_a_directory_scan_is_flat_and_sorted(self) -> None:
        write_agent_type(self.dir, "b-type")
        write_agent_type(self.dir, "a-type")
        write_agent_type(self.dir / "nested", "c-type")
        (self.dir / "notes.txt").write_text("not markdown")
        found, diagnostics = load_agent_types_from_dir(self.dir)
        self.assertEqual([t.name for t in found], ["a-type", "b-type"])
        self.assertEqual(diagnostics, [])


class DiscoverAgentTypesTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cwd = Path(tmp.name)

    def test_builtin_explore_is_always_there(self) -> None:
        types, diagnostics = discover_agent_types(self.cwd)
        self.assertIsNotNone(find_type("explore", types))
        self.assertEqual(diagnostics, [])

    def test_extra_paths_beat_defaults_and_collisions_are_reported(self) -> None:
        first = write_agent_type(self.cwd / "extra", "scout", description="First.")
        write_agent_type(self.cwd / "extra2", "scout", description="Second.")
        types, diagnostics = discover_agent_types(
            self.cwd, extra_paths=[self.cwd / "extra", self.cwd / "extra2"])
        self.assertEqual(find_type("scout", types).description, "First.")
        self.assertTrue(any("collision" in d.message and d.path != first
                            for d in diagnostics))

    def test_a_user_file_shadows_builtin_explore_without_a_diagnostic(self) -> None:
        write_agent_type(self.cwd / "extra", "explore", description="Mine.")
        types, diagnostics = discover_agent_types(self.cwd, extra_paths=[self.cwd / "extra"])
        self.assertEqual(find_type("explore", types).description, "Mine.")
        self.assertEqual(diagnostics, [])

    def test_a_single_file_extra_path_loads(self) -> None:
        path = write_agent_type(self.cwd, "scout")
        types, _ = discover_agent_types(self.cwd, extra_paths=[path])
        self.assertIsNotNone(find_type("scout", types))

    def test_a_missing_extra_path_is_a_diagnostic(self) -> None:
        _, diagnostics = discover_agent_types(self.cwd, extra_paths=[self.cwd / "gone"])
        self.assertIn("does not exist", diagnostics[0].message)

    def test_include_defaults_false_still_keeps_builtins(self) -> None:
        types, _ = discover_agent_types(self.cwd, include_defaults=False)
        self.assertIsNotNone(find_type("explore", types))


class BuiltinExploreTest(unittest.TestCase):
    def test_the_toolset_is_the_read_only_registry_slice(self) -> None:
        expected = {name for name, tool in tools.REGISTRY.items()
                    if tool.access in ("read", "none") and name not in tools.SUBAGENT_DENIED}
        explore = builtin_types()[0]
        self.assertEqual(set(explore.tools), expected)
        self.assertIn("grep", explore.tools)
        self.assertNotIn("shell", explore.tools)
        self.assertNotIn("write_file", explore.tools)
        self.assertTrue(explore.body)


class SpawnToolWithTypesTest(unittest.TestCase):
    def test_the_copy_gains_the_parameter_and_the_listing(self) -> None:
        tool = spawn_tool_with_types(tools.REGISTRY["spawn_agent"], builtin_types())
        function = tool.schema["function"]
        self.assertIn("agent", function["parameters"]["properties"])
        self.assertNotIn("agent", function["parameters"].get("required", []))
        self.assertIn("- explore:", function["description"])

    def test_the_registry_schema_is_never_touched(self) -> None:
        before = tools.REGISTRY["spawn_agent"].schema
        spawn_tool_with_types(tools.REGISTRY["spawn_agent"], builtin_types())
        function = before["function"]
        self.assertNotIn("agent", function["parameters"]["properties"])
        self.assertNotIn("- explore:", function["description"])


if __name__ == "__main__":
    unittest.main()
