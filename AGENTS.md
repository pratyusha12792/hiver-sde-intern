# Repository Guidelines

## Project Structure & Module Organization

This repository is currently empty apart from this guide. Keep the root focused on project-wide configuration and documentation. When implementation begins, use a predictable layout:

- `src/` for application or library code.
- `tests/` for automated tests that mirror `src/` paths.
- `assets/` for static files such as images, fixtures, or sample data.
- `docs/` for design notes and user-facing documentation.

Group code by feature or domain. Avoid generic dumping grounds such as `utils/` unless helpers are genuinely shared.

## Build, Test, and Development Commands

No build system or package manager is configured yet. When one is added, expose common tasks through one standard entry point, such as `Makefile` targets or package scripts. Document exact commands here and in `README.md`. Prefer familiar names:

- `make run` starts the project locally.
- `make test` runs the complete test suite.
- `make lint` checks formatting and static-analysis rules.
- `make build` creates a production artifact.

Do not add placeholder tooling before code requires it.

## Coding Style & Naming Conventions

Follow the formatter and linter standard for the chosen language; commit their configuration with the first source files. Use consistent indentation and UTF-8 text with final newlines. Choose descriptive names: `snake_case` for Python files and functions, `camelCase` for JavaScript variables, and `PascalCase` for exported types or components. Keep modules small and focused.

## Testing Guidelines

Add tests with every non-trivial behavior change. Mirror source names using the ecosystem convention, such as `tests/test_parser.py` or `src/parser.test.ts`. Cover success, boundary, and failure cases. Bug fixes should include a regression test. Keep tests deterministic and independent of external services unless explicitly marked as integration tests.

## Commit & Pull Request Guidelines

No Git history is available to establish an existing convention. Use concise, imperative commit subjects, optionally following Conventional Commits: `feat: add parser` or `fix: reject empty input`.

Pull requests should explain purpose, key changes, and verification performed. Link related issues. Include screenshots for visible UI changes and note any configuration, migration, or compatibility impact.
