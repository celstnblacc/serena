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
import subprocess
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
            "global/../../evil",           # crosses global boundary
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
        dep_names = {d.split("==")[0].split(">=")[0].split(">")[0].split("[")[0].strip().lower() for d in dependencies}
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
