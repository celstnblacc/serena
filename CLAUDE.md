# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

**Essential Commands (use these exact commands):**
- `uv run poe format` - Format code (RUFF) - ONLY allowed formatting command
- `uv run poe type-check` - Run mypy type checking - ONLY allowed type checking command  
- `uv run poe test` - Run tests with default markers (excludes java/rust by default)
- `uv run poe test -m "python or go"` - Run specific language tests
- `uv run poe test -m vue` - Run Vue tests
- `uv run poe lint` - Check code style without fixing

**Test Markers:**
Available pytest markers for selective testing (43 language markers as of v0.1.6):
- Core: `python`, `go`, `java`, `kotlin`, `groovy`, `rust`, `typescript`, `vue`, `php`, `perl`, `powershell`, `csharp`, `elixir`, `elm`, `terraform`, `clojure`, `swift`, `bash`, `ruby`, `r`, `zig`, `lua`, `luau`, `nix`, `dart`, `erlang`, `ocaml`, `scala`, `al`, `fsharp`, `rego`, `markdown`, `julia`, `fortran`, `haskell`, `yaml`, `pascal`, `cpp`, `toml`, `matlab`, `systemverilog`, `hlsl`, `lean4`, `solidity`, `ansible`
- `snapshot` - for symbolic editing operation tests
- `slow` - tests requiring extra Expert instances (~60-90s startup)

**Project Management:**
- `uv run serena` - Main CLI entry point (top-level command)
- `uv run serena-mcp-server` - Start MCP server from project root
- `uv run index-project` - Index project for faster tool performance (deprecated alias)

**Always run format, type-check, and test before completing any task.**

## Architecture Overview

Serena is a dual-layer coding agent toolkit:

### Core Components

**1. SerenaAgent (`src/serena/agent.py`)**
- Central orchestrator managing projects, tools, and user interactions
- Coordinates language servers, memory persistence, and MCP server interface
- Manages tool registry and context/mode configurations

**2. SolidLanguageServer (`src/solidlsp/ls.py`)**  
- Unified wrapper around Language Server Protocol (LSP) implementations
- Provides language-agnostic interface for symbol operations
- Handles caching, error recovery, and multiple language server lifecycle

**3. Tool System (`src/serena/tools/`)**
- **file_tools.py** - File system operations, search, regex replacements
- **symbol_tools.py** - Language-aware symbol finding, navigation, editing
- **memory_tools.py** - Project knowledge persistence and retrieval
- **config_tools.py** - Project activation, mode switching
- **workflow_tools.py** - Onboarding and meta-operations

**4. Configuration System (`src/serena/config/`)**
- **Contexts** - Define tool sets for different environments (desktop-app, agent, ide-assistant)
- **Modes** - Operational patterns (planning, editing, interactive, one-shot)
- **Projects** - Per-project settings and language server configs

### Language Support Architecture

Each supported language has:
1. **Language Server Implementation** in `src/solidlsp/language_servers/`
2. **Runtime Dependencies** - Automatic language server downloads when needed
3. **Test Repository** in `test/resources/repos/<language>/`
4. **Test Suite** in `test/solidlsp/<language>/`

### Memory & Knowledge System

- **Markdown-based storage** in `.serena/memories/` directories
- **Project-specific knowledge** persistence across sessions
- **Contextual retrieval** based on relevance
- **Onboarding support** for new projects

## Development Patterns

### Adding New Languages
1. Create language server class in `src/solidlsp/language_servers/`
2. Add to Language enum in `src/solidlsp/ls_config.py` 
3. Update factory method in `src/solidlsp/ls.py`
4. Create test repository in `test/resources/repos/<language>/`
5. Write test suite in `test/solidlsp/<language>/`
6. Add pytest marker to `pyproject.toml`

### Adding New Tools
1. Inherit from `Tool` base class in `src/serena/tools/tools_base.py`
2. Implement required methods and parameter validation
3. Register in appropriate tool registry
4. Add to context/mode configurations

### Testing Strategy
- Language-specific tests use pytest markers
- Symbolic editing operations have snapshot tests
- Integration tests in `test_serena_agent.py`
- Test repositories provide realistic symbol structures

## Configuration Hierarchy

Configuration is loaded from (in order of precedence):
1. Command-line arguments to `serena-mcp-server`
2. Project-specific `.serena/project.yml`
3. User config `~/.serena/serena_config.yml`
4. Active modes and contexts

## Key Implementation Notes

- **Symbol-based editing** - Uses LSP for precise code manipulation
- **Caching strategy** - Reduces language server overhead
- **Error recovery** - Automatic language server restart on crashes
- **Multi-language support** - 43+ languages with LSP integration (including Vue, Solidity, Ansible, Lean 4)
- **MCP protocol** - Exposes tools to AI agents via Model Context Protocol
- **Async operation** - Non-blocking language server interactions

## Working with the Codebase

- Project uses Python 3.11 with `uv` for dependency management
- Strict typing with mypy, formatted with ruff
- Language servers run as separate processes with LSP communication
- Memory system enables persistent project knowledge
- Context/mode system allows workflow customization

## Guardrails

- This is a **fork** of [oraios/serena](https://github.com/oraios/serena). Upstream URL: `https://github.com/oraios/serena`. Fork: `celstnblacc/serena`.
- Security fixes (S-1 shell injection, S-2 path traversal) in `util/shell.py` and `project.py` must not be reverted. Verify with `pytest test/serena/test_security.py` after any change to those files.
- Never push directly to `main`. Feature branches only.
- Never edit `.env`, credentials, or secret files.
- Run `uv run poe format && uv run poe type-check && uv run poe test` before any commit.
- `index-project` is a deprecated alias for the `serena index` subcommand -- prefer the `serena` CLI.
- Do not add dependencies without pinning exact versions (security constraint; see `pyproject.toml` pattern).

## Strict Installation Decoupling

Once installed (e.g., to ~/.local/bin), the project binary must NEVER depend on the local repository path for execution, configuration, or data. All paths must be relative to the installation root or use standard system config paths (~/.config).
