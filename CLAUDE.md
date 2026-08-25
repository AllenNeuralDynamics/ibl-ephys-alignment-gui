# Agent Guide

This repository is the AIND fork of the IBL ephys alignment GUI: a PyQt5 and
pyqtgraph workstation for mapping electrophysiology feature depth onto a
histology probe track.

Read `docs/architecture.md` before structural changes and
`docs/reference_line_alignment_contract.md` before changing reference lines,
fit/undo/reset behavior, or linked depth displays. Check `TODO.md` for open
correctness work. Those files describe the current implementation; historical
refactor diaries are not architecture authorities.

## Development Commands

Use `uv`:

```bash
uv sync --group dev
uv run pytest -q
uv run pytest tests/core/test_document.py::test_name -q
uv run ruff check src tests
uv run launch
uv run python scripts/generate_alignment_output_schema.py
```

CI runs `uv run ruff check src tests` and `uv run pytest -q` on Python 3.10 for
Linux, macOS, and Windows. The suite currently collects about 780 tests and does
not require a display server. Worker tests use a `QCoreApplication` and explicit
event pumping; other desktop tests use fakes. Do not introduce tests that need a
real widget display unless the test infrastructure is deliberately changed.

`ruff format --check` and `mypy` are not CI gates and currently report existing
debt. The legacy `run_tests` script and coverage `fail_under = 100` setting are
not the supported test path.

The workflow file currently listens to `master` and `develop`, not `main`.
Until that CI configuration is corrected, pushes directly to `main` do not
trigger it; run the two gates locally.

## Architecture Rules

`AlignmentWorkspace` is the Qt-free composition root. `AlignmentApp` is the
frontend boundary and exposes grouped `commands`, `queries`, and typed `events`.
Desktop code receives the app port; it must not reach through the workspace to
the document, controller, runtime, or services.

Only `desktop/` and `launch_gui.py` may depend on Qt or pyqtgraph. Everything in
`application/`, `core/`, `runtime/`, `io/`, `services/`, `geometry/`, and
`plotting/` must remain toolkit-free.

Keep these owners separate:

- `AlignmentDocument`: durable selection and editable per-`AlignmentKey` work;
- `AlignmentDataContext`: normalized input datapackage facts and channel tables;
- runtime objects: heavy, cacheable, evictable arrays and derived caches;
- desktop views/displays: widgets, pyqtgraph items, signals, and teardown.

`AlignmentKey = (recording_id, ephys_collection, shank_idx)` is the stable unit
of alignment work. `shank_idx` is zero-based internally. Never substitute probe
display names or row order for stable identity.

Vocabulary is intentional:

- command: user intent and side-effect sequencing;
- controller: validated document/domain mutation only;
- query: plain read-model construction, optionally materializing named caches;
- event: toolkit-free semantic outcome;
- job/loader: Qt-free long-running work or IO boundary;
- coordinator: desktop action and dialog choreography;
- presenter: event/query DTO to view rendering;
- view/display: concrete Qt and pyqtgraph ownership.

## Common Change Paths

- New ephys plot: add a `PlotSpec` in `plotting/registry.py` and a plain payload
  builder. Menu wiring comes from the registry.
- New user behavior with IO or mutation: add a focused handler under
  `application/commands/` and use the controller for document transitions.
- New render state: add a focused query returning a plain dataclass or payload.
- New cross-panel semantic outcome: add a typed event only when it replaces a
  concrete imperative refresh path.
- New persistent edit setting: add it to document state and the autosave
  snapshot contract.
- New heavy derived data: add an explicitly invalidated runtime cache.
- New producer schema: validate and normalize it in `io/`; keep downstream code
  on one internal representation.

Do not recreate the removed `MainWindow`, `LoadDataLocal`,
`DesktopViewSession`, or `ShankAlignment` god-object patterns under new names.
Broad facades should contain composition/delegation, not mixed business logic.

## Critical Behavior

Loading is selection-driven. Mouse-root selection validates the datapackage and
starts histology warmup; recording selection leaves probe selection blank;
probe/shank selection activates cached data or starts a background load.
Matching preloads and histology warmups are joined. Worker results are guarded
by lifecycle identity, and QThreads must remain owned until stopped.

Depth plots share a physically coaxial ViewBox. Do not use `PlotItem.setTitle()`
on image, line, probe, warped/original annotation, scale/ruler, or perpendicular
slice panels: a title adds a plot-local layout row and breaks visual alignment.
Use `desktop/displays/depth_panel_layout.py` for strip labels and axis geometry.

Full Save writes every saveable document alignment, not just dirty or active
work. It uses the input snapshot and lightweight `SaveGeometryCatalog`; it must
not load full stream runtimes. CCF transforms are batched. CCF failures must
warn and preserve alignment history plus anatomical output rather than discard
user work.

Autosave is an atomic document-only checkpoint at
`<output-package>/autosave/alignment_document.json`. Recovery is explicit from
`File -> Recover Autosave...` after loading the matching mouse root. Successful
full save clears the checkpoint.

## Runtime Configuration

The GUI reads a mouse-root `datapackage.json`; it does not discover sessions by
directory globbing and does not support ONE/Alyx online mode. Relevant
environment variables are:

- `EPHYS_ALIGNMENT_INPUT_ROOT`
- `EPHYS_ALIGNMENT_OUTPUT_ROOT`
- `EPHYS_ALIGNMENT_MAX_CACHED_STREAMS`
- `EPHYS_ALIGNMENT_GUI_TIMING=1`
- `EPHYS_ALIGNMENT_ANTS_POINTS_SUBPROCESS`
- `IBL_ASSET_ROOTS`
- `IBL_ASSET_CONFIG`
- `IBL_ASSET_OVERRIDES`

See `README.md` for user-facing meanings and examples.
