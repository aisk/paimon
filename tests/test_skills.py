import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from paimon import skills
from paimon.prompt import build_system_prompt
from paimon.skills import (
    Skill,
    default_skill_dirs,
    discover_skills,
    expand_skill_command,
    format_skills_for_prompt,
    load_skills_from_dir,
    parse_frontmatter,
    parse_skill_block,
)


def write_skill(directory: Path, name: str | None, description: str | None = "Does things.",
                body: str = "# Body\n\nDo it.", filename: str = "SKILL.md", **extra) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    lines = ["---"]
    if name is not None:
        lines.append(f"name: {name}")
    if description is not None:
        lines.append(f"description: {description}")
    for key, value in extra.items():
        lines.append(f"{key.replace('_', '-')}: {value}")
    lines += ["---", body]
    path = directory / filename
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


class FrontmatterTest(unittest.TestCase):
    def test_header_and_body_are_split(self) -> None:
        header, body = parse_frontmatter("---\nname: a\ndescription: b\n---\n\n# Title\n")
        self.assertEqual(header, {"name": "a", "description": "b"})
        self.assertEqual(body, "# Title")

    def test_folded_multiline_description(self) -> None:
        text = "---\nname: a\ndescription: >\n  first line\n  second line\n---\nbody"
        header, _ = parse_frontmatter(text)
        self.assertEqual(header["description"].strip(), "first line second line")
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "SKILL.md").write_text(text)
            skill, _ = skills.load_skill_file(Path(directory) / "SKILL.md", declared=True)
        self.assertEqual(skill.description, "first line second line", "stripped when loaded")

    def test_no_header_means_empty_mapping(self) -> None:
        self.assertEqual(parse_frontmatter("just text"), ({}, "just text"))
        self.assertEqual(parse_frontmatter("---\nunterminated"), ({}, "---\nunterminated"))

    def test_fence_must_be_exactly_three_dashes(self) -> None:
        self.assertEqual(parse_frontmatter("---\n---\nbody"), ({}, "body"))
        header, body = parse_frontmatter("---\nname: a\ndescription: b ---- c\n---\nbody")
        self.assertEqual(header, {"name": "a", "description": "b ---- c"})
        self.assertEqual(body, "body")

    def test_crlf_is_normalized(self) -> None:
        header, body = parse_frontmatter("---\r\nname: a\r\n---\r\nbody\r\n")
        self.assertEqual(header, {"name": "a"})
        self.assertEqual(body, "body")


class LoadSkillsFromDirTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def test_skill_root_is_not_descended_into(self) -> None:
        write_skill(self.root / "outer", "outer")
        write_skill(self.root / "outer" / "nested", "nested")
        found, diagnostics = load_skills_from_dir(self.root)
        self.assertEqual([s.name for s in found], ["outer"])
        self.assertEqual(diagnostics, [])

    def test_nested_directories_and_root_markdown(self) -> None:
        write_skill(self.root / "group" / "deep", "deep")
        write_skill(self.root, "loose", filename="loose.md")
        (self.root / "README.md").write_text("# Not a skill")
        (self.root / "group" / "notes.md").write_text("---\ndescription: ignored below root\n---\nx")
        found, diagnostics = load_skills_from_dir(self.root)
        self.assertEqual(sorted(s.name for s in found), ["deep", "loose"])
        self.assertEqual(diagnostics, [])

    def test_declared_skill_without_description_warns(self) -> None:
        write_skill(self.root / "bad", "bad", description=None)
        found, diagnostics = load_skills_from_dir(self.root)
        self.assertEqual(found, [])
        self.assertEqual([d.message for d in diagnostics], ["description is required"])

    def test_malformed_frontmatter_warns_only_when_declared(self) -> None:
        (self.root / "bad").mkdir()
        (self.root / "bad" / "SKILL.md").write_text("---\nname: [unclosed\n---\nx")
        (self.root / "stray.md").write_text("---\nname: [unclosed\n---\nx")
        found, diagnostics = load_skills_from_dir(self.root)
        self.assertEqual(found, [])
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(diagnostics[0].path, self.root / "bad" / "SKILL.md")

    def test_name_falls_back_to_directory_and_is_validated(self) -> None:
        write_skill(self.root / "from-dir", None)
        write_skill(self.root / "x", "Bad--Name-")
        found, diagnostics = load_skills_from_dir(self.root)
        self.assertEqual(sorted(s.name for s in found), ["Bad--Name-", "from-dir"])
        messages = {d.message for d in diagnostics}
        self.assertIn("name contains invalid characters (must be lowercase a-z, 0-9, hyphens only)", messages)
        self.assertIn("name must not start or end with a hyphen", messages)
        self.assertIn("name must not contain consecutive hyphens", messages)

    def test_long_description_warns_but_loads(self) -> None:
        write_skill(self.root / "long", "long", description="x" * 1025)
        found, diagnostics = load_skills_from_dir(self.root)
        self.assertEqual(len(found), 1)
        self.assertIn("description exceeds 1024 characters", diagnostics[0].message)

    def test_disable_model_invocation_flag(self) -> None:
        write_skill(self.root / "hidden", "hidden", disable_model_invocation="true")
        found, _ = load_skills_from_dir(self.root)
        self.assertTrue(found[0].disable_model_invocation)

    def test_dotdirs_node_modules_and_ignore_files_are_skipped(self) -> None:
        write_skill(self.root / ".hidden" / "a", "a")
        write_skill(self.root / "node_modules" / "b", "b")
        write_skill(self.root / "build" / "c", "c")
        write_skill(self.root / "keep" / "d", "d")
        write_skill(self.root / "keep" / "drafts" / "e", "e")
        (self.root / ".gitignore").write_text("build/\n")
        (self.root / "keep" / ".ignore").write_text("# comment\ndrafts\n")
        found, _ = load_skills_from_dir(self.root)
        self.assertEqual([s.name for s in found], ["d"])

    def test_ignore_rules_apply_below_their_own_directory_only(self) -> None:
        write_skill(self.root / "one" / "tmp" / "a", "a")
        write_skill(self.root / "two" / "tmp" / "b", "b")
        write_skill(self.root / "two" / "sub" / "out" / "c", "c")
        write_skill(self.root / "two" / "out" / "d", "d")
        write_skill(self.root / "two" / "Tmp" / "e", "e")
        (self.root / "one" / ".gitignore").write_text("tmp\n")
        (self.root / "two" / ".gitignore").write_text("/out/\n")
        found, _ = load_skills_from_dir(self.root)
        self.assertEqual(sorted(s.name for s in found), ["b", "c", "e"])

    def test_missing_directory_is_empty(self) -> None:
        self.assertEqual(load_skills_from_dir(self.root / "nope"), ([], []))


class DiscoverSkillsTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.cwd = self.root / "project"
        self.cwd.mkdir()

    def test_default_dirs_walk_up_to_the_git_root(self) -> None:
        (self.root / ".git").mkdir()
        deep = self.cwd / "src" / "pkg"
        deep.mkdir(parents=True)
        home = self.root / "home"
        with patch("paimon.skills.Path.home", return_value=home), \
                patch("paimon.skills.config_root", return_value=home / ".config" / "paimon"):
            dirs = default_skill_dirs(deep)
        self.assertEqual(dirs, [
            deep.resolve() / ".agents" / "skills",
            (self.cwd / "src").resolve() / ".agents" / "skills",
            self.cwd.resolve() / ".agents" / "skills",
            self.root.resolve() / ".agents" / "skills",
            home / ".agents" / "skills",
            home / ".config" / "paimon" / "skills",
        ])

    def test_explicit_paths_beat_defaults_on_collision(self) -> None:
        loser = write_skill(self.root / "defaults" / "dup", "dup")
        winner = write_skill(self.root / "extra" / "dup", "dup")
        other = write_skill(self.root / "defaults" / "other", "other")
        with patch("paimon.skills.default_skill_dirs", return_value=[self.root / "defaults"]):
            found, diagnostics = discover_skills(self.cwd, extra_paths=[self.root / "extra"])
        self.assertEqual({s.name: s.path for s in found}, {"dup": winner, "other": other})
        self.assertEqual([d.path for d in diagnostics], [loser])
        self.assertIn("collision", diagnostics[0].message)

    def test_extra_path_may_be_a_file_relative_to_cwd_or_missing(self) -> None:
        write_skill(self.cwd / "one", "one")
        found, diagnostics = discover_skills(
            self.cwd, extra_paths=["one/SKILL.md", "missing", str(self.cwd / "one")],
            include_defaults=False)
        self.assertEqual([s.name for s in found], ["one"], "the same file twice loads once, silently")
        self.assertEqual([(d.message, d.path) for d in diagnostics],
                         [("skill path does not exist", self.cwd / "missing")])

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks")
    def test_symlinked_duplicates_are_dropped_silently(self) -> None:
        write_skill(self.root / "real" / "s", "s")
        try:
            os.symlink(self.root / "real", self.root / "link")
        except OSError as exc:
            self.skipTest(str(exc))
        found, diagnostics = discover_skills(
            self.cwd, extra_paths=[self.root / "real", self.root / "link"], include_defaults=False)
        self.assertEqual(len(found), 1)
        self.assertEqual(diagnostics, [])


class PromptFormattingTest(unittest.TestCase):
    def test_block_escapes_and_hides_disabled_skills(self) -> None:
        shown = Skill("a", 'Use for <x> & "y"', Path("/s/a/SKILL.md"), Path("/s/a"))
        hidden = Skill("b", "manual only", Path("/s/b/SKILL.md"), Path("/s/b"), True)
        block = format_skills_for_prompt([shown, hidden])
        self.assertIn("<available_skills>", block)
        self.assertIn("<description>Use for &lt;x&gt; &amp; &quot;y&quot;</description>", block)
        self.assertIn("<location>/s/a/SKILL.md</location>", block)
        self.assertNotIn("<name>b</name>", block)
        self.assertEqual(format_skills_for_prompt([hidden]), "")

    def test_system_prompt_lists_skills_before_the_environment(self) -> None:
        skill = Skill("a", "desc", Path("/s/a/SKILL.md"), Path("/s/a"))
        with tempfile.TemporaryDirectory() as directory:
            prompt = build_system_prompt(Path(directory), [skill])
            bare = build_system_prompt(Path(directory))
        self.assertLess(prompt.index("<available_skills>"), prompt.index("<environment>"))
        self.assertNotIn("<available_skills>", bare)


class InvocationTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.path = write_skill(self.root / "demo", "demo", body="# Demo\n\nRun ./go.sh")
        self.skill = Skill("demo", "Does things.", self.path, self.path.parent)

    def test_expands_with_and_without_arguments(self) -> None:
        plain = expand_skill_command("/skill:demo", [self.skill])
        self.assertEqual(plain, (
            f'<skill name="demo" location="{self.path}">\n'
            f"References are relative to {self.path.parent}.\n\n# Demo\n\nRun ./go.sh\n</skill>"))
        with_args = expand_skill_command("/skill:demo do it now", [self.skill])
        self.assertTrue(with_args.endswith("</skill>\n\ndo it now"))
        multi = expand_skill_command("/skill:demo\nsecond line", [self.skill])
        self.assertTrue(multi.endswith("</skill>\n\nsecond line"))

    def test_unknown_or_unreadable_skill_is_left_alone(self) -> None:
        self.assertEqual(expand_skill_command("/skill:nope x", [self.skill]), "/skill:nope x")
        self.assertEqual(expand_skill_command("plain text", [self.skill]), "plain text")
        self.path.unlink()
        self.assertEqual(expand_skill_command("/skill:demo", [self.skill]), "/skill:demo")

    def test_arguments_alone_go_through_expand_args(self) -> None:
        seen: list[str] = []

        def upper(args: str) -> str:
            seen.append(args)
            return args.upper()

        text = expand_skill_command("/skill:demo do it", [self.skill], expand_args=upper)
        self.assertEqual(seen, ["do it"])
        self.assertTrue(text.endswith("</skill>\n\nDO IT"))
        self.assertIn("Run ./go.sh", text, "the body is untouched")

    def test_quotes_in_paths_and_closing_tags_in_bodies_round_trip(self) -> None:
        odd_dir = self.root / 'we"ird'
        path = write_skill(odd_dir, "odd", body="first\n</skill>\n\nnot the end")
        skill = Skill("odd", "d", path, odd_dir)
        text = expand_skill_command("/skill:odd tail", [skill])
        self.assertIn('location="' + str(path).replace('"', "&quot;") + '"', text)
        block = parse_skill_block(text)
        assert block is not None
        self.assertEqual(block.location, str(path))
        self.assertTrue(block.body.endswith("first\n</skill>\n\nnot the end"))
        self.assertEqual(block.user_message, "tail")

    def test_parse_skill_block_is_the_inverse(self) -> None:
        text = expand_skill_command("/skill:demo do it", [self.skill])
        block = parse_skill_block(text)
        assert block is not None
        self.assertEqual((block.name, block.location), ("demo", str(self.path)))
        self.assertTrue(block.body.startswith("References are relative to"))
        self.assertTrue(block.body.endswith("Run ./go.sh"))
        self.assertEqual(block.user_message, "do it")
        self.assertIsNone(parse_skill_block(expand_skill_command("/skill:demo", [self.skill])).user_message)
        self.assertIsNone(parse_skill_block("hello <skill"))
