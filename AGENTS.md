# Project Agent Instructions

Use `CLAUDE.md` as the shared contributor and coding-agent guide for this
repository. Before structural changes, also read `docs/architecture.md`. Before
changing reference lines, fitting, undo/reset behavior, or linked depth plots,
read `docs/reference_line_alignment_contract.md`. Open work is in `TODO.md`.

Use `uv` for Python environment management and commands. The required local
gates are:

```bash
uv run ruff check src tests
uv run pytest -q
```

Do not install project packages into the global Python environment.
