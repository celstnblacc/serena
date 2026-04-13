import os
import re
import subprocess

from pydantic import BaseModel

from solidlsp.util.subprocess_util import subprocess_kwargs

# SEC-002: Pattern that matches shell metacharacters enabling injection attacks.
# Blocked characters/sequences: ; | & ` $( ) && || \n \r hex/octal shell escapes
# These are the primary vectors for command injection when shell=True is used.
_SHELL_METACHAR_RE = re.compile(r"[;|&`]|\$\(|\band\b|\bor\b|&&|\|\||\r|\n|\\x[0-9a-fA-F]{2}|\\[0-7]{3}")

# Trust level controlling whether shell execution is permitted at runtime.
# A mutable list is used as a container to avoid the PLW0603 global-statement lint.
# Index 0: True = shell allowed, False = shell blocked (--no-shell mode).
_shell_state: list[bool] = [True]


def set_shell_enabled(enabled: bool) -> None:
    """Enable or disable shell execution globally.  Call before starting the MCP server.

    When disabled, :func:`execute_shell_command` raises :class:`PermissionError` unconditionally.
    Language-server processes (which use a separate Popen path) are unaffected — only
    the :class:`~serena.tools.cmd_tools.ExecuteShellCommandTool` MCP tool is blocked.
    """
    _shell_state[0] = enabled


def is_shell_enabled() -> bool:
    """Return True if shell execution is currently permitted."""
    return _shell_state[0]


def _validate_no_shell_metacharacters(command: str) -> None:
    """Raise ValueError if the command string contains shell injection metacharacters.

    SEC-002: execute_shell_command uses shell=True, which allows the shell to
    interpret metacharacters and run arbitrary secondary commands.  Agents
    constructing commands from user/LLM input could be manipulated into
    injecting additional commands via ';', '&&', '|', backticks, or '$(...)'
    subshells.

    Note: this is a defence-in-depth measure.  The primary safeguard remains
    the principle that only trusted callers should invoke this function.
    """
    match = _SHELL_METACHAR_RE.search(command)
    if match:
        raise ValueError(
            f"Rejected command containing shell metacharacter '{match.group()}' at position {match.start()}. "
            "Use individual arguments with subprocess.run([...]) for untrusted input, "
            "or confirm the command is safe before calling execute_shell_command."
        )


class ShellCommandResult(BaseModel):
    stdout: str
    return_code: int
    cwd: str
    stderr: str | None = None


def execute_shell_command(command: str, cwd: str | None = None, capture_stderr: bool = False) -> ShellCommandResult:
    """
    Execute a shell command and return the output.

    :param command: The command to execute.
    :param cwd: The working directory to execute the command in. If None, the current working directory will be used.
    :param capture_stderr: Whether to capture the stderr output.
    :return: The output of the command.
    :raises PermissionError: if shell execution has been disabled via :func:`set_shell_enabled`.
    """
    if not _shell_state[0]:
        raise PermissionError(
            "Shell execution is disabled (no-shell mode). Remove --no-shell from the server startup command to re-enable."
        )
    _validate_no_shell_metacharacters(command)

    if cwd is None:
        cwd = os.getcwd()

    process = subprocess.Popen(
        command,
        shell=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE if capture_stderr else None,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=cwd,
        **subprocess_kwargs(),
    )

    stdout, stderr = process.communicate()
    return ShellCommandResult(stdout=stdout, stderr=stderr, return_code=process.returncode, cwd=cwd)


def subprocess_check_output(args: list[str], encoding: str = "utf-8", strip: bool = True, timeout: float | None = None) -> str:
    output = subprocess.check_output(
        args, stdin=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=timeout, env=os.environ.copy(), **subprocess_kwargs()
    ).decode(encoding)  # type: ignore
    if strip:
        output = output.strip()
    return output
