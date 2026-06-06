# My Project

Knowledge Organizer for eXtended documents

## Features

- ✅ **UV Package Manager**: Fast, modern Python dependency and virtual environment management
- ✅ **ty Type Checking**: Strict type checking for all Python code
- ✅ **Pytest**: Testing framework with 100% coverage requirement
- ✅ **Ruff**: Fast Python linter and formatter
- ✅ **codespell**: Spell checker for code and documentation
- ✅ **mise Task System**: Advanced task runner with DAG dependency management
- ✅ **Pre-commit Hooks**: Automated code quality checks
- ✅ **Zensical**: Modern documentation with Material Design theme
- ✅ **Dynamic Versioning**: Git-based versioning with uv-dynamic-versioning

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/knox.git
cd knox

# Install dependencies
mise dev

# Or manually
uv sync --dev
```

## Usage

### Running Tests

```bash
mise test
# or
uv run pytest
```

### Code Quality

```bash
# Run linter
mise lint

# Format code
mise format

# Check spelling
mise spell

# Run all checks
mise all
```

### CLI Commands

```bash
# Default command
uv run knox

# Greet someone
uv run knox greet Alice

# Add numbers
uv run knox add 5 3

# Show version
uv run knox --version
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `KNOX_CACHE` | Enable/disable caching (true/false) |
| `KNOX_OUTPUT_FILE` | Default output file path |

## Development

### Setting Up Development Environment

```bash
# Install all dependencies including dev tools
mise dev
```

### Pre-commit Hooks

Install pre-commit hooks to run quality checks before commits:

```bash
pre-commit install
pre-commit run --all-files
```

## Project Structure

```
knox/
├── .config/mise/config.toml                # mise tasks/configuration
├── src/knox/
│   ├── __init__.py
│   ├── __about__.py                        # Version info
│   └── cli/                                # Typer CLI module
├── tests/                                  # Pytest tests
├── .pre-commit-config.yaml                 # Pre-commit hooks configuration
├── ruff.toml                               # Ruff linter configuration
├── ty.toml                                 # ty type checker configuration
└── pyproject.toml                          # Project configuration
```

## Versioning

This project uses **uv-dynamic-versioning** for automatic version management based on Git tags.

- Version is derived from Git tags (e.g., `v0.1.0`)
- Fallback version: `0.0.0` for development mode
- No need to manually update version strings

## License

This project is licensed under the **MIT** license.

See the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run quality checks: `mise all`
5. Submit a pull request

## AGENTS.md

This project includes a generic [AGENTS.md](AGENTS.md) template with global instructions for AI assistants, adapted from [Anbeeld/AGENTS.md](https://github.com/Anbeeld/AGENTS.md).

## Support

For issues and feature requests, please use the GitHub issue tracker.
