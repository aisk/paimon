import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from paimon.tools.base import ToolContext
from paimon.tools.files import _glob, _inside
from paimon.tools.permissions import _GIT_LOCATION_ENV, MODES, gate, safe_command
from paimon.tools.shell import _shell, shell_output_dir
from tests.support.shell import MODELLED_SHELL, filler, modelled_shell, overflow_path


def _scrub_git_location(test: unittest.TestCase) -> None:
    """Unset, for the test, every variable that stops git being recognized."""
    env = patch.dict(os.environ)
    env.start()
    test.addCleanup(env.stop)
    for name in _GIT_LOCATION_ENV:
        os.environ.pop(name, None)


class GateTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cwd = Path(tmp.name).resolve()

    def test_yolo_allows_everything(self) -> None:
        for name in ("read_file", "glob", "write_file", "edit_file", "shell", "write_todos"):
            self.assertEqual(gate(name, {"path": "/etc/hosts"}, "yolo", self.cwd), "allow")

    def test_reads_inside_cwd_are_free_outside_held(self) -> None:
        for mode, held in (("read", "deny"), ("auto", "review")):
            self.assertEqual(gate("read_file", {"path": "a.py"}, mode, self.cwd), "allow")
            self.assertEqual(gate("read_file", {"path": "/etc/hosts"}, mode, self.cwd), held)
            self.assertEqual(gate("read_file", {"path": "../x"}, mode, self.cwd), held)
            self.assertEqual(gate("glob", {"pattern": "*.py"}, mode, self.cwd), "allow")
            self.assertEqual(gate("glob", {"pattern": "*", "path": "/tmp"}, mode, self.cwd), held)

    def test_only_the_agents_own_overflow_files_are_read_freely(self) -> None:
        """A command's own overflow file is readable back; nothing else moves.

        The exemption is per agent: the directory is shared by every session
        and project on the machine, so a blanket one would be a free sideways
        read into another agent's command output.
        """
        with tempfile.TemporaryDirectory() as data_home:
            with patch.dict(os.environ, {"PAIMON_DATA_HOME": data_home}):
                directory = shell_output_dir()
                directory.mkdir(parents=True, exist_ok=True)
                mine = directory / "20260811-120000-1-abc.log"
                theirs = directory / "20260811-120000-2-def.log"
                mine.write_text("x")
                theirs.write_text("x")
                ctx = ToolContext(shell_outputs={mine.resolve()})

                self.assertEqual(gate("read_file", {"path": str(mine)}, "read", self.cwd, ctx=ctx), "allow")
                self.assertEqual(gate("read_file", {"path": str(theirs)}, "read", self.cwd, ctx=ctx), "deny")
                self.assertEqual(gate("read_file", {"path": str(mine)}, "read", self.cwd), "deny")
                self.assertEqual(gate("read_file", {"path": "/etc/hosts"}, "read", self.cwd, ctx=ctx), "deny")
                self.assertEqual(gate("write_file", {"path": str(mine)}, "auto", self.cwd, ctx=ctx), "review")

    async def test_a_command_records_the_overflow_file_it_wrote(self) -> None:
        """The gate exemption above is only reachable through this."""
        with tempfile.TemporaryDirectory() as data_home:
            with patch.dict(os.environ, {"PAIMON_DATA_HOME": data_home}):
                ctx = ToolContext()
                result = (await _shell({"command": filler(3000, "line")}, self.cwd, ctx)).text
                self.assertEqual(ctx.shell_outputs, {overflow_path(result).resolve()})
                self.assertEqual(gate("read_file", {"path": str(overflow_path(result))},
                                      "read", self.cwd, ctx=ctx), "allow")

    def test_read_mode_denies_all_dangerous_tools(self) -> None:
        self.assertEqual(gate("write_file", {"path": "a.py", "content": "x"}, "read", self.cwd), "deny")
        self.assertEqual(gate("edit_file", {"path": "a.py"}, "read", self.cwd), "deny")
        self.assertEqual(gate("shell", {"command": "rm -rf x"}, "read", self.cwd), "deny")

    def test_auto_mode_allows_writes_inside_cwd(self) -> None:
        self.assertEqual(gate("write_file", {"path": "a.py", "content": "x"}, "auto", self.cwd), "allow")
        self.assertEqual(gate("edit_file", {"path": "sub/a.py"}, "auto", self.cwd), "allow")
        self.assertEqual(gate("write_file", {"path": "/tmp/a.py", "content": "x"}, "auto", self.cwd), "review")
        self.assertEqual(gate("edit_file", {"path": "../a.py"}, "auto", self.cwd), "review")
        self.assertEqual(gate("shell", {"command": "rm -rf x"}, "auto", self.cwd), "review")

    def test_auto_mode_holds_writes_into_a_git_directory(self) -> None:
        """A hook or core.fsmonitor written there runs on the next git status,
        which is let through on its own."""
        (self.cwd / ".git" / "hooks").mkdir(parents=True)
        for path in (".git/config", ".git/hooks/pre-commit", "sub/.git/config", ".GIT/config",
                     ".git", str(self.cwd / ".git" / "config")):
            self.assertEqual(gate("write_file", {"path": path, "content": "x"}, "auto", self.cwd),
                             "review", path)
            self.assertEqual(gate("edit_file", {"path": path}, "auto", self.cwd), "review", path)
        self.assertEqual(gate("write_file", {"path": ".github/x.yml", "content": "x"}, "auto",
                              self.cwd), "allow")
        self.assertEqual(gate("write_file", {"path": ".gitignore", "content": "x"}, "auto",
                              self.cwd), "allow")
        self.assertEqual(gate("read_file", {"path": ".git/config"}, "auto", self.cwd), "allow")
        self.assertEqual(gate("read_file", {"path": ".git/config"}, "read", self.cwd), "allow")

    def test_write_todos_and_missing_path_are_allowed(self) -> None:
        for mode in MODES:
            self.assertEqual(gate("write_todos", {"todos": []}, mode, self.cwd), "allow")
        self.assertEqual(gate("read_file", {}, "read", self.cwd), "allow")

    @modelled_shell
    def test_safe_shell_commands_auto_allowed(self) -> None:
        (self.cwd / ".git").mkdir()  # git is only recognized in a repository's root
        _scrub_git_location(self)
        with patch("paimon.tools.permissions.shell_executable", return_value=MODELLED_SHELL):
            for mode in ("read", "auto"):
                self.assertEqual(gate("shell", {"command": "ls"}, mode, self.cwd), "allow")
                self.assertEqual(gate("shell", {"command": "git status"}, mode, self.cwd), "allow")
            with patch.dict(os.environ, {"CDPATH": ""}):
                self.assertEqual(gate("shell", {"command": "cd sub && ls"}, "read", self.cwd), "allow")
            self.assertEqual(gate("shell", {}, "read", self.cwd), "deny")

    def test_strict_disables_safe_commands(self) -> None:
        for mode, held in (("read", "deny"), ("auto", "review")):
            self.assertEqual(gate("shell", {"command": "ls"}, mode, self.cwd, safe_commands=False), held)
        self.assertEqual(gate("shell", {"command": "ls"}, "yolo", self.cwd, safe_commands=False), "allow")


class SafeCommandTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cwd = Path(tmp.name).resolve()
        # The list models sh, so it is tested as if sh ran the command on
        # POSIX, and against the Git Bash that runs it on Windows; under
        # cmd.exe nothing is recognized at all.
        if MODELLED_SHELL is None:
            self.skipTest("commands run in cmd.exe")
        shell = patch("paimon.tools.permissions.shell_executable", return_value=MODELLED_SHELL)
        shell.start()
        self.addCleanup(shell.stop)
        # The root of a repository, the one place git is recognized at all,
        # with nothing pointing git elsewhere (a git hook running the tests
        # sets GIT_INDEX_FILE, for one).
        (self.cwd / ".git").mkdir()
        _scrub_git_location(self)

    def test_recognized_read_only_commands(self) -> None:
        for cmd in (
            "ls", "ls -la", "pwd", "cat sub/a.txt", "head -n 20 a.py",
            "tail -n 5 a.log", "wc -l src/*.py", "wc -l ./*.py", "wc -l -- *.py",
            "stat a.py", "file a.py",
            "which python3", "df", "du -sh .", "grep TODO src", "grep -rn foo .",
            "rg TODO", "rg -n foo src", "find . -name '*.py' -type f",
            "rg -g '*.py' foo", "grep -rn --include='*.py' foo .", "grep -rn foo src/*",
            "diff a.txt b.txt", "readlink -f a.py",
            "realpath a.py", "echo done", "uname -a", "whoami", "date", "nproc",
            "git status", "git status --porcelain", "git log --oneline -20",
            "git log HEAD~3..", "git diff --stat", "git show HEAD~1",
            "git branch -a", "git branch --show-current",
            "git blame cli.py", "git shortlog -sn", "git describe --tags",
            "git rev-parse HEAD", "git ls-files", "git ls-tree -r HEAD",
            "git remote -v", "git tag -l",
            # Git for Windows ships no tree, and System32's tree.com is not it.
            *(("tree -L 2",) if os.name == "posix" else ()),
        ):
            self.assertTrue(safe_command(cmd, self.cwd), cmd)

    def test_globs_that_could_expand_to_parent_reject(self) -> None:
        # ".*" expands to "." and "..", so the literal path check is not enough.
        for cmd in ("ls .*", "grep -r SECRET .*", "cat .*/*", "grep -r x .*/.*",
                    "ls ..*", "ls .?", "wc -l sub/.*"):
            self.assertFalse(safe_command(cmd, self.cwd), cmd)

    def test_globs_that_could_expand_to_options_reject(self) -> None:
        # A file named "--pre=sh" or "-delete" matches "*" and the command
        # takes it for an option; "*.py" is matched by "--files0-from=x.py".
        for cmd in ("rg foo *", "find * -name x", "find . -name *.py", "ls *", "cat ?",
                    "wc -l *.py", "grep -r foo *.txt", "echo *", "rg --pr* x foo",
                    "find -- *", "git diff *", "git log --oneline *.py", "git log --out*"):
            self.assertFalse(safe_command(cmd, self.cwd), cmd)
        # Quoted, a pattern is the command's to interpret; after "--" a name
        # is an operand to a command that ends its options there; and a
        # literal first directory fixes how every expansion begins.
        for cmd in ("find . -name '*.py'", "find . -name \"*.py\"", "rg -g '*.py' foo",
                    "ls -- *", "cat -- *.txt", "grep -rn foo -- *", "ls src/*.py", "ls ./*",
                    "git diff -- *", "git log --oneline -- '*.py'"):
            self.assertTrue(safe_command(cmd, self.cwd), cmd)

    def test_a_glob_matching_a_symlink_out_of_cwd_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as outside:
            (self.cwd / "src").mkdir()
            (self.cwd / "src" / "a.txt").write_text("x")
            self.assertTrue(safe_command("cat src/*", self.cwd))
            (self.cwd / "src" / "link").symlink_to(Path(outside), target_is_directory=True)
            self.assertFalse(safe_command("cat src/*", self.cwd))
            self.assertFalse(safe_command("ls -- src/*", self.cwd))
            self.assertTrue(safe_command("ls -- *", self.cwd), "lists src, not into the link")

    def test_a_parent_segment_rejects(self) -> None:
        """The shell follows a link before applying "..", while Python strikes
        "link/.." out as text (always on Windows, and on POSIX when the path
        does not exist as written), so link/../x is read outside the cwd."""
        with tempfile.TemporaryDirectory() as outside:
            deep = Path(outside) / "deep"
            deep.mkdir()
            (Path(outside) / "secret.txt").write_text("x")
            (self.cwd / "secret.txt").write_text("x")
            (self.cwd / "link").symlink_to(deep, target_is_directory=True)
            for cmd in ("cat link/../secret.txt", "cat lin*/../secret.txt",
                        "cat sub/../secret.txt", "ls sub/..", "grep -r x link/..",
                        "diff --from-file=.. secret.txt", "grep -f.. secret.txt"):
                self.assertFalse(safe_command(cmd, self.cwd), cmd)
        with patch.dict(os.environ, {"CDPATH": ""}):
            self.assertFalse(safe_command("cd sub && ls ../", self.cwd))

    def test_abbreviated_long_options_are_denied_like_the_full_ones(self) -> None:
        # getopt_long and git's parser both accept any unambiguous prefix.
        for cmd in ("tail --foll a.log", "tail -n1 --follo a.log", "date --se 2020-01-01",
                    "date --s=2020", "file --compil x", "rg --pr x foo", "rg --hostname-bi x foo",
                    "git blame --conten x a.txt", "git log --outp=x", "git show --textc HEAD",
                    "git diff --ext", "git log --show-sig"):
            self.assertFalse(safe_command(cmd, self.cwd), cmd)
        for cmd in ("tail -n 5 a.log", "rg --pre-glob '*.gz' foo", "date --utc",
                    "git log --oneline", "git log --stat", "git log --no-ext-diff",
                    "git show --name-only HEAD"):
            self.assertTrue(safe_command(cmd, self.cwd), cmd)

    def test_lists_of_file_names_reject(self) -> None:
        # The names inside the list are opened and never checked.
        for cmd in ("wc --files0-from list", "wc --files0-from=list", "du --files0-from list",
                    "file -f list", "file --files-from list", "find . -files0-from list",
                    "tree -R -L 1"):
            self.assertFalse(safe_command(cmd, self.cwd), cmd)

    def test_attached_flag_values_reject(self) -> None:
        # A path attached to a flag never reaches the containment check.
        for cmd in ("diff --from-file=/etc/passwd x", "diff --to-file=/etc/passwd x",
                    "grep -f/etc/passwd -r .", "grep --file=/etc/passwd x",
                    "file -m/etc/passwd .", "du --exclude-from=/etc/passwd .",
                    "date -f/etc/passwd", "date -f /etc/passwd"):
            self.assertFalse(safe_command(cmd, self.cwd), cmd)

    def test_hard_metacharacters_reject(self) -> None:
        # Substitution, redirection, escaping and expansion are rejected even
        # inside quotes; only ";", "&" and "|" get position-aware handling.
        for cmd in (
            "cat $(pwd)/x", "echo `id`", "ls > out.txt", "cat < /etc/passwd",
            "(ls)", "ls \\\n x", "cat {..,x}/y", "ls [.][.]",
            'echo "$HOME"', "echo \"`id`\"",
        ):
            self.assertFalse(safe_command(cmd, self.cwd), cmd)

    def test_compound_commands_allowed(self) -> None:
        for cmd in ("ls | wc -l", "git log --oneline | head -5",
                    "grep -rn foo . | wc -l", "ls && pwd",
                    "git log; git status", "ls || pwd", "echo a&&pwd"):
            self.assertTrue(safe_command(cmd, self.cwd), cmd)

    def test_compound_unsafe_segment_rejects(self) -> None:
        for cmd in ("ls; rm x", "ls | rm x", "ls && rm -rf .",
                    "git status; git push", "cat f | sed -i s/a/b/ f",
                    'echo "&&" && rm x'):
            self.assertFalse(safe_command(cmd, self.cwd), cmd)

    def test_quoted_operators_allowed(self) -> None:
        for cmd in ('grep "a|b" f', "grep -E 'foo|bar' .", "echo 'a && b'",
                    "grep ';' f"):
            self.assertTrue(safe_command(cmd, self.cwd), cmd)

    def test_nothing_is_recognized_under_cmd(self) -> None:
        # cmd.exe parses differently from the sh this list models: "'" is not
        # a quote there, %VAR% expands, "^" escapes, and a command is looked
        # up in the current directory before PATH.
        with patch("paimon.tools.permissions.shell_executable", return_value=None):
            for cmd in ("ls", "git status", "echo %USERNAME%", "cat .^./x",
                        "echo ' && del important.txt && echo '"):
                self.assertFalse(safe_command(cmd, self.cwd), cmd)

    def test_operator_edge_cases_reject(self) -> None:
        for cmd in ("ls &&", "&& ls", "| wc", "ls |", "ls ;; pwd",
                    "ls && && pwd", "ls & pwd", "ls & rm x", "ls &",
                    "ls |& pwd", "ls 'x && pwd"):
            self.assertFalse(safe_command(cmd, self.cwd), cmd)

    def test_cd_chains_allowed(self) -> None:
        with patch.dict(os.environ, {"CDPATH": ""}):
            for cmd in ("cd sub && ls", "cd sub && cat a.txt",
                        "cd sub && cd deeper && ls",
                        "cd 'sub' && ls", f"cd {self.cwd.as_posix()}/sub && ls", "cd sub"):
                self.assertTrue(safe_command(cmd, self.cwd), cmd)

    def test_cd_forms_reject(self) -> None:
        for cmd in ("cd", "cd -", "cd -P sub", "cd a b", "cd ..", "cd ../x",
                    "cd a/../b && ls", "cd .* && ls", "cd sub* && ls",
                    "cd s?b && ls"):
            self.assertFalse(safe_command(cmd, self.cwd), cmd)

    def test_cd_glob_target_cannot_escape_via_symlink(self) -> None:
        # A glob's literal form hides the symlink it matches, so the target's
        # containment check would otherwise pass while the shell cd's out.
        (self.cwd / "sublink").symlink_to("/etc")
        self.assertFalse(safe_command("cd sublink* && cat passwd", self.cwd))

    def test_cd_requires_pure_and_chain(self) -> None:
        # With ";" or "||" a failed or skipped cd leaves later segments in a
        # different directory than modeled; in a pipeline cd runs in its own
        # subshell and applies to nothing.
        for cmd in ("cd sub; ls", "cd sub | ls", "ls | cd sub",
                    "cd sub || ls", "ls; cd sub && cat f",
                    "ls || cd sub && cat ../f"):
            self.assertFalse(safe_command(cmd, self.cwd), cmd)

    def test_cd_boundary_is_original_cwd(self) -> None:
        self.assertFalse(safe_command("cd sub && cat ../../f", self.cwd))
        (self.cwd / "link").symlink_to("/etc")
        self.assertFalse(safe_command("cd link && ls", self.cwd))

    def test_cdpath_rejects_cd_only(self) -> None:
        with patch.dict(os.environ, {"CDPATH": "/tmp"}):
            self.assertFalse(safe_command("cd sub && ls", self.cwd))
            self.assertTrue(safe_command("ls | wc -l", self.cwd))

    def test_tilde_and_parse_failures_reject(self) -> None:
        for cmd in ("cat ~/secrets", "grep --include=~/x foo", "ls 'unclosed", ""):
            self.assertFalse(safe_command(cmd, self.cwd), cmd)

    def test_unlisted_commands_reject(self) -> None:
        for cmd in ("rm x", "sed -i s/a/b/ f", "sort -o out in", "env",
                    "./ls", "/bin/ls", "FOO=1 ls"):
            self.assertFalse(safe_command(cmd, self.cwd), cmd)

    def test_denied_flags_reject(self) -> None:
        for cmd in ("tail -f log", "tail -20f log", "tail --follow log",
                    "file -C -m magic src", "rg --pre sh foo",
                    "rg --hostname-bin=x foo", "find . -delete",
                    "find . -name x -exec rm {} +", "find . -fprintf out %p",
                    "tree -o out", "date -s 12:00"):
            self.assertFalse(safe_command(cmd, self.cwd), cmd)

    def test_paths_outside_cwd_reject(self) -> None:
        for cmd in ("cat /etc/passwd", "cat ../x", "grep foo /etc/passwd", "du -sh /"):
            self.assertFalse(safe_command(cmd, self.cwd), cmd)

    def test_symlink_escape_rejects(self) -> None:
        (self.cwd / "link").symlink_to("/etc")
        self.assertFalse(safe_command("cat link/passwd", self.cwd))

    def test_unsafe_git_rejects(self) -> None:
        for cmd in (
            "git", "git push", "git commit -m x", "git -c core.pager=sh log",
            "git -C /tmp status", "git log --output=/tmp/x",
            "git log --output-directory=/tmp", "git diff --no-index a b",
            "git diff --ext-diff", "git show --textconv HEAD:f",
            "git branch foo", "git branch -D foo", "git branch --set-upstream-to=x",
            "git remote add origin url", "git remote prune origin",
            "git tag v1", "git tag -d v1", "git stash", "git reflog expire --all",
            # git grep -O runs a command, and bundling (-nOsh) hides it.
            "git grep -n foo", "git grep -nOsh foo", "git grep -Ovim foo",
            # Options that read a file the cwd check never sees.
            "git blame --contents /etc/passwd -- x", "git blame --contents=/etc/passwd x",
            "git ls-files -X /etc/passwd", "git ls-files -cX /etc/passwd",
            "git ls-files --exclude-from=/etc/passwd",
            "git log --show-signature", "git show --textconv HEAD:f",
            "git blame -S revs a.txt", "git blame -wS revs a.txt",
            "git blame --ignore-revs-file revs a.txt", "git blame --ignore-revs-file=revs a.txt",
            "git log -O order", "git diff -Oorder", "git show -O order HEAD",
            "git rev-parse --resolve-git-dir x", "git ls-files --exclude-per-directory=x",
            # A %G placeholder verifies signatures, running gpg.program.
            "git log --format=%G?", "git log --format %GG", "git log '--pretty=format:%h %GS'",
        ):
            self.assertFalse(safe_command(cmd, self.cwd), cmd)
        for cmd in ("git log '--format=%h %s'", "git blame --ignore-rev HEAD a.txt",
                    "git log -L 1,2:a.txt", "git diff --relative"):
            self.assertTrue(safe_command(cmd, self.cwd), cmd)

    def test_git_is_recognized_only_in_a_real_repository_root(self) -> None:
        """git takes any directory holding HEAD, objects/ and refs/ for a
        repository and runs what its config says (core.fsmonitor on status).
        The agent may write such a directory unreviewed, just not a .git."""
        evil = self.cwd / "evil"
        for name in ("objects", "refs"):
            (evil / name).mkdir(parents=True)
        (evil / "HEAD").write_text("ref: refs/heads/main\n")
        (evil / "config").write_text("[core]\n\tfsmonitor = touch pwned\n")
        with patch.dict(os.environ, {"CDPATH": ""}):
            self.assertTrue(safe_command("git status", self.cwd))
            self.assertFalse(safe_command("cd evil && git status", self.cwd))
            self.assertTrue(safe_command("cd evil && ls", self.cwd))
            # Without its own .git the cwd itself could be made that shape.
            self.assertFalse(safe_command("git status", evil))
            (self.cwd / ".git").rmdir()
            self.assertFalse(safe_command("git status", self.cwd))
            self.assertTrue(safe_command("ls", self.cwd))
            (self.cwd / ".git").mkdir()
            for variable in ("GIT_DIR", "GIT_WORK_TREE", "GIT_CONFIG_GLOBAL"):
                with patch.dict(os.environ, {variable: "evil"}):
                    self.assertFalse(safe_command("git status", self.cwd), variable)

    def test_an_environment_that_redefines_commands_rejects(self) -> None:
        # bash imports exported functions and runs BASH_ENV before the
        # command; either can make "ls" something else.
        for env in ({"BASH_FUNC_ls%%": "() { echo pwned; }"}, {"BASH_ENV": "setup.sh"}):
            with patch.dict(os.environ, env):
                self.assertFalse(safe_command("ls", self.cwd), env)
        # A function named like some other command is no reason to refuse:
        # Fedora exports one for "which" in every login shell.
        with patch.dict(os.environ, {"BASH_FUNC_which%%": "() { :; }"}):
            self.assertTrue(safe_command("ls", self.cwd))
            self.assertFalse(safe_command("which ls", self.cwd))

    @unittest.skipUnless(os.name == "posix", "the POSIX PATH lookup")
    def test_a_relative_path_entry_ahead_of_the_command_rejects(self) -> None:
        # An empty entry is the current directory to sh, where the agent may
        # have just written an executable named ls.
        real = os.environ["PATH"]
        for path in (f":{real}", f"bin:{real}"):
            with patch.dict(os.environ, {"PATH": path}):
                self.assertFalse(safe_command("ls", self.cwd), path)
                self.assertTrue(safe_command("echo hi", self.cwd), "a builtin is not looked up")
        with patch.dict(os.environ, {"PATH": f"{real}:"}):
            self.assertTrue(safe_command("ls", self.cwd), "found before the relative entry")


@unittest.skipUnless(os.name == "nt", "Windows path semantics")
class WindowsSafeCommandTest(unittest.TestCase):
    """Under Git Bash a path token is resolved twice: the Win32 way by the
    check, the MSYS way by the command. Only tokens both read alike pass."""

    def setUp(self) -> None:
        if MODELLED_SHELL is None:
            self.skipTest("commands run in cmd.exe")
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cwd = Path(tmp.name).resolve()
        shell = patch("paimon.tools.permissions.shell_executable", return_value=MODELLED_SHELL)
        shell.start()
        self.addCleanup(shell.stop)

    def test_tokens_msys_and_win32_read_differently_reject(self) -> None:
        for cmd in (
            # MSYS paths: Git's own /etc and /tmp (the user's temp directory),
            # drive and UNC spellings, emulated devices.
            "cat /etc/passwd", "cat /tmp/x", "cat /c/Users/x", "ls /",
            "cat //server/share/x", "cat /dev/stdin", "cd /c && ls",
            # Drive-relative paths and alternate data streams.
            "cat C:x", "cd C:x && ls", "cat a.txt:secret", "grep -fC:x .", "grep --file=C:x .",
            # Device names, in any directory and with any extension.
            "cat CON", "cat con.txt", "rg x sub/nul", "cat COM1", "cat 'AUX .log'",
            # Win32 strips trailing dots and spaces; MSYS does not.
            "cat x.", "cat 'x '", "cat sub./a",
            # The same, attached to an option.
            "grep -fCON .", "grep --file=con.txt .", "diff --from-file=NUL a.txt",
        ):
            self.assertFalse(safe_command(cmd, self.cwd), cmd)

    def test_drive_paths_both_read_alike_are_allowed(self) -> None:
        root = self.cwd.as_posix()  # C:/Users/...
        for cmd in (f"cat {root}/a.txt", f"cd {root}/sub && ls", "cat sub/a.txt", "ls ."):
            self.assertTrue(safe_command(cmd, self.cwd), cmd)
        self.assertFalse(safe_command(f"cat {root}/../x", self.cwd))

    def test_only_the_programs_git_ships_are_recognized(self) -> None:
        self.assertFalse(safe_command("tree", self.cwd), "System32's tree.com is not GNU tree")
        # Git's own directories come first in Git Bash's PATH, and the
        # current directory is not on it, so this never runs for "cat".
        (self.cwd / "cat.exe").write_bytes(b"MZ")
        self.assertTrue(safe_command("cat a.txt", self.cwd))

    def test_a_relative_path_entry_rejects_what_comes_from_path(self) -> None:
        system = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
        with patch.dict(os.environ, {"PATH": f"bin;{system}"}):
            self.assertFalse(safe_command("rg foo", self.cwd), "rg is looked up on PATH")
            self.assertTrue(safe_command("ls", self.cwd), "ls never gets that far")
        with patch.dict(os.environ, {"PATH": f"{system};;"}):
            self.assertTrue(safe_command("rg foo", self.cwd), "MSYS drops empty entries")

    def test_nothing_is_recognized_under_an_unknown_bash(self) -> None:
        # Command lookup is modelled for Git's launcher only.
        with patch("paimon.tools.permissions.shell_executable", return_value="/bin/sh"):
            self.assertFalse(safe_command("ls", self.cwd))


class InsideTest(unittest.TestCase):
    def test_symlink_escape_is_outside(self) -> None:
        with tempfile.TemporaryDirectory() as outer:
            outer = Path(outer).resolve()
            cwd = outer / "project"
            cwd.mkdir()
            secret = outer / "secret.txt"
            secret.write_text("secret")
            link = cwd / "link.txt"
            link.symlink_to(secret)

            self.assertTrue(_inside(cwd / "a.py", cwd))
            self.assertFalse(_inside(secret, cwd))
            self.assertFalse(_inside(link, cwd))

    def test_symlink_loop_is_held_instead_of_raising(self) -> None:
        """A loop must come back as a permission decision, not end the turn.

        Before Python 3.13 resolve() raises RuntimeError on a loop, so the path
        counts as outside and the gate confirms. From 3.13 resolve() returns
        the path with the loop left in place; that sits inside cwd, the gate
        allows, and the ELOOP surfaces at open time inside the tool-call error
        boundary. Either way nothing raises here.
        """
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory).resolve()
            loop = cwd / "loop"
            loop.symlink_to(loop)

            try:
                loop.resolve()
            except (OSError, RuntimeError):
                self.assertFalse(_inside(loop, cwd))
                expected = "deny"
            else:
                self.assertTrue(_inside(loop, cwd))
                expected = "allow"
            self.assertEqual(gate("read_file", {"path": "loop"}, "read", cwd), expected)


class GlobSandboxTest(unittest.TestCase):
    def test_sandboxed_glob_filters_symlink_escapes(self) -> None:
        with tempfile.TemporaryDirectory() as outer:
            outer = Path(outer).resolve()
            cwd = outer / "project"
            cwd.mkdir()
            (cwd / "a.py").write_text("a")
            secret = outer / "secret.py"
            secret.write_text("secret")
            (cwd / "b.py").symlink_to(secret)

            sandboxed = _glob({"pattern": "*.py"}, cwd, sandboxed=True)
            self.assertIn("a.py", sandboxed)
            self.assertNotIn("b.py", sandboxed)

            free = _glob({"pattern": "*.py"}, cwd, sandboxed=False)
            self.assertIn("a.py", free)
            self.assertIn("b.py", free)


class BackgroundGateTest(unittest.TestCase):
    """run_background is held even when the command itself looks harmless."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cwd = Path(tmp.name).resolve()

    def test_a_safe_looking_command_is_still_held(self) -> None:
        args = {"command": "ls -la", "description": "listing"}
        if MODELLED_SHELL is not None:
            with patch("paimon.tools.permissions.shell_executable", return_value=MODELLED_SHELL):
                self.assertEqual(gate("shell", args, "read", self.cwd), "allow")
        self.assertEqual(gate("run_background", args, "read", self.cwd), "deny",
                         "nothing that keeps running is waved through")
        self.assertEqual(gate("run_background", args, "auto", self.cwd), "review")

    def test_yolo_still_means_yolo(self) -> None:
        self.assertEqual(
            gate("run_background", {"command": "npm run dev"}, "yolo", self.cwd), "allow")

    def test_looking_at_and_stopping_a_job_are_not_gated(self) -> None:
        for name in ("read_job", "stop_job"):
            self.assertEqual(gate(name, {"job_id": "a1f2"}, "read", self.cwd), "allow")
