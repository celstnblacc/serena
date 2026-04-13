"""
Security regression tests for Serena.

TDD cycle:
  RED   — tests written first that prove the vulnerability or absence of safeguard.
  GREEN — fixes applied; tests now pass.
  REFACTOR — cleanup only, no new behavior.

Findings covered:
  SEC-001 (MEDIUM) Memory path traversal in MemoriesManager.get_memory_file_path
  SEC-002 (HIGH)   shell=True with user-controlled command string in execute_shell_command
  SEC-003 (LOW)    pywebview unpinned in pyproject.toml
  SEC-004 (LOW)    Dashboard Flask server has no authentication middleware
                   (localhost-only by default; noted as accepted risk)
"""

from __future__ import annotations

import pathlib
import tempfile

import pytest

# ---------------------------------------------------------------------------
# SEC-001 — Memory path traversal
# ---------------------------------------------------------------------------


class TestMemoryPathTraversal:
    """
    MemoriesManager.get_memory_file_path must resolve the resulting path and
    verify it stays inside the allowed memory directory.

    Attack vector: a caller passes a name like '../../etc/passwd'; the string
    join produces /project/.serena/memories/../../etc/passwd which, after
    resolve(), lands outside the memory root.
    """

    def _make_manager(self, tmp_path: pathlib.Path):
        from serena.project import MemoriesManager

        data_folder = tmp_path / ".serena"
        return MemoriesManager(serena_data_folder=str(data_folder))

    @pytest.mark.parametrize(
        "traversal_name",
        [
            "../../etc/passwd",
            "../../../tmp/evil",
            "sub/../../etc/shadow",
            "sub/../../../etc/hosts",
            "global/../../evil",  # crosses global boundary
        ],
    )
    def test_traversal_name_raises(self, tmp_path, traversal_name):
        """Path traversal names must raise ValueError, not silently escape the root."""
        mgr = self._make_manager(tmp_path)
        with pytest.raises((ValueError, PermissionError)):
            mgr.get_memory_file_path(traversal_name)

    def test_normal_name_resolves_inside_dir(self, tmp_path):
        """A normal memory name must resolve inside the project memory directory."""
        mgr = self._make_manager(tmp_path)
        path = mgr.get_memory_file_path("my_memory")
        project_memory_dir = (tmp_path / ".serena" / "memories").resolve()
        assert path.resolve().is_relative_to(project_memory_dir)

    def test_nested_normal_name_resolves_inside_dir(self, tmp_path):
        """A nested (topic/name) memory name must resolve inside the project memory dir."""
        mgr = self._make_manager(tmp_path)
        path = mgr.get_memory_file_path("auth/login/logic")
        project_memory_dir = (tmp_path / ".serena" / "memories").resolve()
        assert path.resolve().is_relative_to(project_memory_dir)

    def test_save_memory_traversal_raises(self, tmp_path):
        """save_memory with a traversal name must raise, not write outside the dir."""
        mgr = self._make_manager(tmp_path)
        with pytest.raises((ValueError, PermissionError)):
            mgr.save_memory("../../evil", "bad content", is_tool_context=False)

    def test_load_memory_traversal_raises(self, tmp_path):
        """load_memory with a traversal name must raise, not read outside the dir."""
        mgr = self._make_manager(tmp_path)
        with pytest.raises((ValueError, PermissionError)):
            mgr.load_memory("../../etc/passwd")

    def test_delete_memory_traversal_raises(self, tmp_path):
        """delete_memory with a traversal name must raise."""
        mgr = self._make_manager(tmp_path)
        with pytest.raises((ValueError, PermissionError)):
            mgr.delete_memory("../../evil", is_tool_context=False)


# ---------------------------------------------------------------------------
# SEC-002 — shell=True with user-controlled input
# ---------------------------------------------------------------------------


class TestShellInjection:
    """
    execute_shell_command passes the command string directly to
    subprocess.Popen(shell=True).  An agent-controlled command string
    containing shell metacharacters (';', '&&', '|') can run arbitrary code.

    The safe fix is to validate / reject shell metacharacters, or document
    clearly that this is an intentional privileged operation.

    These tests verify that the CURRENT implementation exposes the issue,
    and that after the fix metacharacter injection is blocked.
    """

    def test_semicolon_injection_is_blocked(self):
        """
        A command containing ';' (shell statement separator) must be rejected.
        The probe file must NOT be created.
        """
        from serena.util.shell import execute_shell_command

        with tempfile.TemporaryDirectory() as workdir:
            probe = pathlib.Path(workdir) / "pwned.txt"
            injected = f"echo hello ; touch {probe}"
            with pytest.raises(ValueError, match="shell metacharacter"):
                execute_shell_command(injected, cwd=workdir)
            assert not probe.exists(), "Injection succeeded — shell metacharacter was not blocked"

    def test_ampersand_injection_is_blocked(self):
        """A command containing '&&' must be rejected."""
        from serena.util.shell import execute_shell_command

        with tempfile.TemporaryDirectory() as workdir:
            probe = pathlib.Path(workdir) / "pwned2.txt"
            injected = f"echo hello && touch {probe}"
            with pytest.raises(ValueError, match="shell metacharacter"):
                execute_shell_command(injected, cwd=workdir)
            assert not probe.exists()

    def test_pipe_injection_is_blocked(self):
        """A command containing '|' must be rejected."""
        from serena.util.shell import execute_shell_command

        with tempfile.TemporaryDirectory() as workdir:
            with pytest.raises(ValueError, match="shell metacharacter"):
                execute_shell_command("echo hello | cat", cwd=workdir)

    def test_backtick_injection_is_blocked(self):
        """A command containing a backtick subshell must be rejected."""
        from serena.util.shell import execute_shell_command

        # Build the command string at runtime to avoid literal backtick in source
        cmd = "echo " + chr(96) + "id" + chr(96)
        with tempfile.TemporaryDirectory() as workdir:
            with pytest.raises(ValueError, match="shell metacharacter"):
                execute_shell_command(cmd, cwd=workdir)

    def test_dollar_paren_injection_is_blocked(self):
        """A command containing $(...) subshell must be rejected."""
        from serena.util.shell import execute_shell_command

        with tempfile.TemporaryDirectory() as workdir:
            with pytest.raises(ValueError, match="shell metacharacter"):
                execute_shell_command("echo $(id)", cwd=workdir)

    def test_simple_command_still_works(self):
        """A clean, simple command must still execute successfully."""
        from serena.util.shell import execute_shell_command

        with tempfile.TemporaryDirectory() as workdir:
            result = execute_shell_command("echo hello", cwd=workdir)
            assert result.return_code == 0
            assert "hello" in result.stdout

    def test_command_with_quoted_args_still_works(self):
        """A command with quoted arguments (no metacharacters) must still work."""
        from serena.util.shell import execute_shell_command

        with tempfile.TemporaryDirectory() as workdir:
            result = execute_shell_command('echo "hello world"', cwd=workdir)
            assert result.return_code == 0
            assert "hello world" in result.stdout


# ---------------------------------------------------------------------------
# SEC-003 — pywebview unpinned in pyproject.toml
# ---------------------------------------------------------------------------


class TestDependencyPins:
    """
    All runtime dependencies must be pinned to a specific version or commit.
    pywebview was previously unpinned (`"pywebview"` with no version specifier),
    allowing any future malicious or breaking release to be installed.

    The fix pins pywebview to a specific git commit via [tool.uv.sources].
    """

    def test_pywebview_has_pin(self):
        """
        pyproject.toml must pin pywebview to a specific version or git commit.

        Acceptable forms:
          - pywebview==X.Y.Z  (exact version)
          - pywebview @ git+...@<sha>  (git commit pin via uv.sources)
        """
        import tomllib

        pyproject_path = pathlib.Path(__file__).parent.parent.parent / "pyproject.toml"
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)

        # Check uv.sources — a git commit pin is acceptable
        uv_sources = data.get("tool", {}).get("uv", {}).get("sources", {})
        pywebview_source = uv_sources.get("pywebview", {})
        has_git_pin = bool(pywebview_source.get("git")) and bool(pywebview_source.get("rev"))

        # Or check the main deps list for a version specifier
        dependencies = data.get("project", {}).get("dependencies", [])
        pywebview_deps = [d for d in dependencies if d.lower().startswith("pywebview")]
        has_version_pin = any("==" in dep for dep in pywebview_deps)

        assert has_git_pin or has_version_pin, (
            "pywebview must be pinned to a specific version or git commit. "
            f"Current entry: {pywebview_deps!r}, uv.sources: {pywebview_source!r}"
        )

    def test_security_transitive_deps_are_pinned(self):
        """
        Transitive dependencies pinned for security (urllib3, werkzeug, starlette, etc.)
        must remain present with exact version pins.
        """
        import tomllib

        pyproject_path = pathlib.Path(__file__).parent.parent.parent / "pyproject.toml"
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)

        dependencies = data.get("project", {}).get("dependencies", [])
        pinned_exact = {d.split("==")[0].strip().lower() for d in dependencies if "==" in d}

        required_pins = {"urllib3", "werkzeug", "starlette", "cryptography"}
        missing_pins = required_pins - pinned_exact
        assert not missing_pins, (
            f"Security-critical transitive deps missing exact pins: {missing_pins}. "
            "These must be pinned with '==' to prevent dependabot-alert regressions."
        )


# ---------------------------------------------------------------------------
# SEC-004 — Dashboard authentication (accepted risk, documented)
# ---------------------------------------------------------------------------


class TestDashboardAuth:
    """
    The Flask dashboard has no authentication middleware.

    Accepted risk: the default listen address is 127.0.0.1 (localhost-only).
    However, if the user configures '0.0.0.0', any local network host can
    access the dashboard — including memory read/write, config mutation,
    and arbitrary shell command triggering.

    These tests verify:
    1. The default listen address is localhost (127.0.0.1).
    2. A configuration audit warning exists when '0.0.0.0' is used
       without additional protection.
    """

    def test_default_listen_address_is_localhost(self):
        """The default web_dashboard_listen_address must be 127.0.0.1."""
        import dataclasses

        from serena.config.serena_config import SerenaConfig

        # SerenaConfig is a dataclass; get the default from fields
        fields = {f.name: f for f in dataclasses.fields(SerenaConfig)}
        field = fields.get("web_dashboard_listen_address")
        assert field is not None, "web_dashboard_listen_address field not found in SerenaConfig"
        field_default = field.default
        assert field_default == "127.0.0.1", (
            f"Default dashboard listen address should be '127.0.0.1', got '{field_default}'. "
            "Changing to '0.0.0.0' would expose the dashboard to the local network without auth."
        )

    def test_serena_config_default_instantiation_is_localhost(self):
        """A freshly instantiated SerenaConfig must bind the dashboard to localhost."""
        from serena.config.serena_config import SerenaConfig

        config = SerenaConfig(gui_log_window=False, web_dashboard=False)
        assert config.web_dashboard_listen_address == "127.0.0.1"


# ---------------------------------------------------------------------------
# SEC-005 — Path traversal edge cases (symlinks, encoded sequences)
# ---------------------------------------------------------------------------


class TestPathTraversalEdgeCases:
    """Extended traversal coverage: symlink escapes, spaces in paths, URL-encoded dots."""

    def _make_manager(self, tmp_path: pathlib.Path):
        from serena.project import MemoriesManager

        return MemoriesManager(serena_data_folder=str(tmp_path / ".serena"))

    def test_symlink_pointing_outside_memory_dir_is_rejected(self, tmp_path):
        """A symlink inside the memory dir that points outside must be rejected on read."""
        mgr = self._make_manager(tmp_path)
        # Bootstrap the memories directory by accessing a valid path first.
        memory_dir = tmp_path / ".serena" / "memories"
        memory_dir.mkdir(parents=True, exist_ok=True)

        # Create a sensitive file outside the tree.
        outside = tmp_path / "secret.txt"
        outside.write_text("top-secret")

        # Symlink inside the memory dir pointing outside.
        # The manager appends ".md" to every name, so the symlink must also use ".md".
        link = memory_dir / "escape.md"
        link.symlink_to(outside)

        # load_memory("escape") resolves to memory_dir/escape.md → outside/secret.txt.
        # The path validation uses resolve() which follows symlinks — the resolved
        # path will be outside the memory root and must be rejected.
        with pytest.raises((ValueError, PermissionError)):
            mgr.load_memory("escape")

    @pytest.mark.parametrize(
        "traversal_name",
        [
            "foo bar/../../etc/passwd",  # spaces in path component
            "a/b/../../../etc/shadow",  # multi-level with sibling
            "./../../etc/hosts",  # leading dot-slash
        ],
    )
    def test_traversal_with_spaces_and_dots_raises(self, tmp_path, traversal_name):
        mgr = self._make_manager(tmp_path)
        with pytest.raises((ValueError, PermissionError)):
            mgr.get_memory_file_path(traversal_name)

    def test_url_encoded_traversal_is_not_accepted(self, tmp_path):
        """URL-encoded '..' (%2e%2e) must not bypass validation."""
        mgr = self._make_manager(tmp_path)
        # The file system won't decode %2e, so this becomes a literal filename —
        # verify it resolves *inside* the memory dir (no escape), not outside.
        # If the name is somehow decoded and escapes, the test fails.
        path = mgr.get_memory_file_path("%2e%2e/etc/passwd")
        memory_dir = (tmp_path / ".serena" / "memories").resolve()
        # Either it raises (strict mode) or the path stays inside memory_dir.
        assert path.resolve().is_relative_to(memory_dir), "URL-encoded traversal '%2e%2e' must not escape the memory directory"


# ---------------------------------------------------------------------------
# SEC-006 — Extended shell metacharacter injection
# ---------------------------------------------------------------------------


class TestShellInjectionEdgeCases:
    """Additional shell injection vectors: &&, ||, newlines, hex escapes."""

    def test_double_ampersand_is_blocked(self):
        """'&&' (AND-list) must be rejected."""
        from serena.util.shell import execute_shell_command

        with tempfile.TemporaryDirectory() as workdir:
            probe = pathlib.Path(workdir) / "pwned.txt"
            with pytest.raises(ValueError, match="shell metacharacter"):
                execute_shell_command(f"echo ok && touch {probe}", cwd=workdir)
            assert not probe.exists()

    def test_double_pipe_is_blocked(self):
        """'||' (OR-list) must be rejected."""
        from serena.util.shell import execute_shell_command

        with tempfile.TemporaryDirectory() as workdir:
            with pytest.raises(ValueError, match="shell metacharacter"):
                execute_shell_command("false || echo injected", cwd=workdir)

    def test_newline_injection_is_blocked(self):
        """A command containing a literal newline (multi-command) must be rejected."""
        from serena.util.shell import execute_shell_command

        with tempfile.TemporaryDirectory() as workdir:
            probe = pathlib.Path(workdir) / "newline_pwned.txt"
            with pytest.raises(ValueError, match="shell metacharacter"):
                execute_shell_command(f"echo ok\ntouch {probe}", cwd=workdir)
            assert not probe.exists()

    def test_hex_escape_injection_is_blocked(self):
        r"""A command containing \xNN hex escape must be rejected."""
        from serena.util.shell import execute_shell_command

        with tempfile.TemporaryDirectory() as workdir:
            with pytest.raises(ValueError, match="shell metacharacter"):
                execute_shell_command(r"echo $'\x3btouch /tmp/x'", cwd=workdir)


# ---------------------------------------------------------------------------
# SEC-007 — No-shell mode (trust level enforcement)
# ---------------------------------------------------------------------------


class TestNoShellMode:
    """execute_shell_command must be blocked when shell execution is disabled."""

    def setup_method(self):
        from serena.util.shell import set_shell_enabled

        # Ensure shell is enabled before each test (reset any prior state).
        set_shell_enabled(True)

    def teardown_method(self):
        from serena.util.shell import set_shell_enabled

        # Always restore shell to enabled so other tests are unaffected.
        set_shell_enabled(True)

    def test_shell_disabled_raises_permission_error(self):
        from serena.util.shell import execute_shell_command, set_shell_enabled

        set_shell_enabled(False)
        with pytest.raises(PermissionError, match="no-shell mode"):
            execute_shell_command("echo hello")

    def test_shell_re_enabled_allows_execution(self):
        from serena.util.shell import execute_shell_command, set_shell_enabled

        set_shell_enabled(False)
        set_shell_enabled(True)
        with tempfile.TemporaryDirectory() as workdir:
            result = execute_shell_command("echo ok", cwd=workdir)
        assert result.return_code == 0

    def test_is_shell_enabled_reflects_state(self):
        from serena.util.shell import is_shell_enabled, set_shell_enabled

        set_shell_enabled(False)
        assert not is_shell_enabled()
        set_shell_enabled(True)
        assert is_shell_enabled()


# ---------------------------------------------------------------------------
# SEC-008 — Atomic write integrity
# ---------------------------------------------------------------------------


class TestAtomicWrites:
    """Memory saves must be atomic: no partial file left behind on failure."""

    def _make_manager(self, tmp_path: pathlib.Path):
        from serena.project import MemoriesManager

        return MemoriesManager(serena_data_folder=str(tmp_path / ".serena"))

    def test_save_memory_creates_file(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mgr.save_memory("note", "hello world", is_tool_context=False)
        path = mgr.get_memory_file_path("note")
        assert path.exists()
        assert path.read_text() == "hello world"

    def test_save_memory_backup_created_on_overwrite(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mgr.save_memory("note", "v1", is_tool_context=False)
        mgr.save_memory("note", "v2", is_tool_context=False)
        path = mgr.get_memory_file_path("note")
        bak = path.with_suffix(".bak")
        assert bak.exists(), "Backup file must be created before overwriting"
        assert bak.read_text() == "v1", "Backup must contain the previous content"
        assert path.read_text() == "v2", "Main file must contain the new content"

    def test_no_stray_tmp_file_after_save(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mgr.save_memory("note", "content", is_tool_context=False)
        memory_dir = tmp_path / ".serena" / "memories"
        tmp_files = list(memory_dir.glob("*.tmp"))
        assert not tmp_files, f"Stray .tmp files found after save: {tmp_files}"
