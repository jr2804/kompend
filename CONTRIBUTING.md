# Contributing

Contributions are welcome! Here's how you can help.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/Hermes/knox.git
cd knox

# Install dependencies
uv sync --dev

# Install mise tasks
mise install
```

> **Full contributor guide at [knox.anomalyco.dev](https://knox.anomalyco.dev)** — architecture, CLI reference, and design decisions.

## Running Tests

```bash
mise test
```

## Code Quality

```bash
# Lint and format
mise lint
mise format

# Type check
mise type-check
```

## Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
