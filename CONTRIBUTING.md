# Contributing

This is the main development guide for human contributors and coding agents.
Agent-specific instructions are in [AGENTS.md](AGENTS.md).

Thank you for improving the IBL Ephys Alignment GUI. This is a stateful
scientific application, so changes should preserve user alignment work,
coordinate correctness, and GUI responsiveness before optimizing for code
movement or abstraction.

## Development Setup

Python 3.10 or newer is required. Use `uv` for the project environment:

```bash
uv sync --group dev
uv run launch
uv run python scripts/generate_alignment_output_schema.py
```

Do not install project dependencies into the global Python environment.

## Required Checks

Run the same lint and test commands used by CI:

```bash
uv run ruff check src tests
uv run pytest -q
```

Run a focused test with `uv run pytest tests/core/test_document.py::test_name -q`.

The legacy `run_tests` script and coverage `fail_under = 100` setting are
not the supported test path.

The test suite does not require a display server. Desktop tests normally use
plain fakes; QThread lifecycle tests use a `QCoreApplication` and explicit
event pumping. Avoid introducing tests that require a real `QApplication` or
visible widgets unless the test infrastructure is deliberately extended.

`ruff format --check` and `mypy` are not currently required gates because the
repository has existing formatting and typing debt. Keep edited code consistent
with its surroundings, but do not mix broad formatting or typing cleanup into
an unrelated change.

The current GitHub Actions workflow listens to `master` and `develop`, while
active integration occurs on `main`. Until that configuration is corrected,
run the required checks locally for changes targeting `main`.

## Read Before Changing Architecture

Before changing input paths, asset resolution, ephys loading, channel/shank
selection, or QC plots, read
[docs/input_datapackage_contract.md](docs/input_datapackage_contract.md).
It distinguishes schema fields from convention-loaded collection contents;
both are compatibility contracts shared with the producers.

[docs/architecture.md](docs/architecture.md) describes the implemented package
boundaries, state ownership, load and worker lifecycles, plotting caches,
autosave, and full-save behavior. Treat it as the architecture authority.

The central dependency rule is that only `desktop/` and the process entry point
may depend on Qt or pyqtgraph. The reusable engine under `application/`,
`core/`, `runtime/`, `io/`, `services/`, `geometry/`, and `plotting/` must remain
toolkit-free.

Before changing reference lines, fit/undo/reset behavior, alignment coordinate
conversion, or linked depth plots, also read
[docs/reference_line_alignment_contract.md](docs/reference_line_alignment_contract.md).
Its coaxial-line, coordinate-space, and repeated-fit idempotence rules are hard
correctness requirements.

Open work and acceptance criteria are tracked in [TODO.md](TODO.md). Do not add
completed implementation history to the TODO or architecture document; Git
history already records it.

## Place Changes At The Owning Boundary

- User intent, mutation, or side-effect sequencing belongs in a focused
  `application/commands/` handler.
- Document transitions go through `AlignmentController`; persistent edit state
  belongs in `AlignmentDocument` and its autosave snapshot.
- Render/read state belongs in focused application queries returning plain
  dataclasses, arrays, mappings, or scalars.
- Semantic outcomes shared by desktop consumers may use typed app events.
- Heavy and disposable derived data belongs in runtime caches with explicit
  invalidation.
- Producer formats and path compatibility belong at the `io/` normalization
  boundary.
- Qt widgets, dialogs, pyqtgraph items, signals, and teardown belong under
  `desktop/`.
- New ephys plots use `PlotSpec` entries and plain payload builders; menu wiring
  is derived from the registry.

Do not give desktop views access to `AlignmentWorkspace` internals, put Qt
objects in app DTOs, store large arrays in the document, or grow
`MainWindow`/`DesktopWorkbench` into mixed-concern application objects.

## Correctness Expectations

- Preserve per-shank edits and pending reference lines across navigation.
- Treat `AlignmentKey(recording_id, ephys_collection, shank_idx)` as the stable
  alignment identity. Do not key behavior by display name or channel row order.
- Use physical channel depth for physiology plots, including irregular and
  split-bank channel maps.
- Keep linked depth-panel ViewBoxes physically coaxial. Do not add plot-local
  titles to those panels; use the fixed axis-label helpers in
  `desktop/displays/depth_panel_layout.py`.
- Guard asynchronous results against stale mouse-root, stream, and load
  identities. Keep QThread owners alive until their threads stop.
- Autosave must remain a lightweight document-only checkpoint.
- Full Save must include every saveable document alignment without loading full
  stream runtimes.
- Derived CCF failures must warn and preserve alignment history and anatomical
  output whenever those products remain valid.

## Testing Changes

Add focused tests at the boundary that owns the behavior:

- `tests/core/` for document, controller, policy, history, and snapshot rules;
- `tests/application/` for command/query transactions and cross-service use
  cases;
- `tests/runtime/` for cache ownership and eviction;
- `tests/io/` for producer contracts and normalization;
- `tests/services/` and `tests/geometry/` for output and numerical behavior;
- `tests/plotting/` for registry, payload, and channel-layout behavior;
- `tests/desktop/` for Qt adaptation, rendering calls, workers, and dialogs.

Regression tests should reproduce the user-visible failure before the fix and
assert the durable behavior rather than an incidental implementation detail.
Broader integration coverage is appropriate for high-risk workflows such as
selection-driven load, cancellation and stale results, autosave recovery,
batched no-runtime save, CCF fallback, and reference-line fitting.

## Scope And Commits

Keep changes narrowly scoped and preserve unrelated work in the tree. Avoid
opportunistic renames, formatting churn, or broad facade splits unless they are
necessary for the behavior being changed.

Use conventional commit subjects, for example:

```text
fix(plotting): preserve split-bank depth scale
feat(save): recover document autosaves
refactor(desktop): isolate worker lifecycle ownership
docs: clarify alignment coordinate contract
```

Describe behavioral changes and verification in the pull request. For
correctness-sensitive changes, include the affected data contract or workflow
in the description and note any real-data validation that could not be run
locally.

## Application Boundaries And Vocabulary

`AlignmentWorkspace` is the Qt-free composition root. `AlignmentApp` is the
frontend boundary, exposing grouped `commands`, `queries`, and typed `events`.
Desktop code receives the app port and must not reach through the workspace to
the document, controller, runtime, or services.

`AlignmentDataContext` owns normalized input datapackage facts and channel
tables. Keep it separate from durable document edits, evictable runtime arrays,
and desktop widgets. `shank_idx` is zero-based internally.

Use consistent vocabulary: commands sequence user intent and side effects;
controllers validate domain mutations; queries build plain read models;
events express toolkit-free outcomes; jobs/loaders perform long-running work
or IO; coordinators handle desktop actions and dialogs; presenters translate
query/event DTOs for views; views/displays own Qt and pyqtgraph objects.

Do not recreate the removed `MainWindow`, `LoadDataLocal`,
`DesktopViewSession`, or `ShankAlignment` patterns under new names. Broad
facades should compose and delegate rather than mix business logic.

## Load, Save, And Runtime Contracts

Loading is selection-driven. Mouse-root selection validates the datapackage and
starts histology warmup; recording selection leaves probe selection blank;
probe/shank selection activates cached data or starts a background load.
Join matching preloads and histology warmups, guard results by lifecycle
identity, and retain QThread owners until their threads stop.

Full Save uses the input snapshot and lightweight `SaveGeometryCatalog`, with
batched CCF transforms. It includes every saveable alignment, including clean
and inactive work, without loading full stream runtimes.

Autosave is an atomic document-only checkpoint at
`<output-package>/autosave/alignment_document.json`. Recovery is explicit from
`File -> Recover Autosave...` after loading the matching mouse root. A successful
full save clears the checkpoint.

The GUI reads a mouse-root `datapackage.json`; it does not discover sessions by
directory globbing or support ONE/Alyx online mode. See [README.md](README.md)
for runtime configuration, including `EPHYS_ALIGNMENT_INPUT_ROOT`,
`EPHYS_ALIGNMENT_OUTPUT_ROOT`, `EPHYS_ALIGNMENT_MAX_CACHED_STREAMS`,
`EPHYS_ALIGNMENT_GUI_TIMING`, `EPHYS_ALIGNMENT_ANTS_POINTS_SUBPROCESS`,
`IBL_ASSET_ROOTS`, `IBL_ASSET_CONFIG`, and `IBL_ASSET_OVERRIDES`.
