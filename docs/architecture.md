# Ephys Alignment GUI Architecture

Status: current implementation and design contract.

This document describes the architecture that exists on `main`. It is the
durable reference for ownership and dependency decisions. Historical refactor
plans and commit-by-commit migration notes are intentionally not kept here.

## Goals

The GUI is a stateful scientific workstation. It handles large arrays,
interactive pyqtgraph items, long-running IO and ANTs transforms, recoverable
user edits, and multiple cached probe streams. The architecture is intended to:

- keep alignment semantics independent of Qt and pyqtgraph;
- make user work durable across navigation, cache eviction, and crashes;
- isolate large, disposable runtime data from the editable document;
- keep long-running work off the GUI thread while preserving Qt object
  lifetimes;
- make new plots and workflows additive instead of expanding `MainWindow`;
- leave the alignment engine reusable by another frontend.

The goal is clear ownership, not maximal abstraction. Broad composition objects
are acceptable when they contain wiring rather than business behavior.

## System Overview

```text
launch_gui.py
    |
    v
desktop shell, views, displays, coordinators, presenters, workers
    |                       Qt / pyqtgraph boundary
    v
AlignmentApp {commands, queries, events}
    |
    +--> application command handlers --> controller --> document/core
    |               |                    --> runtime / IO / services
    |               +--> typed events
    |
    +--> application queries -----------> document / runtime / services
    |
    +--> EventBus ----------------------> desktop presenters/coordinators

AlignmentWorkspace composes the Qt-free objects behind AlignmentApp.
```

Only `desktop/` and the process entry point may depend on Qt or pyqtgraph.
`application/`, `core/`, `runtime/`, `io/`, `services/`, `geometry/`, and
`plotting/` must remain toolkit-free. Plotting payloads are plain arrays and
values; concrete plot items belong to desktop displays.

## Package Responsibilities

| Package | Responsibility |
|---|---|
| `application/` | Frontend port, use-case commands, rendering queries, save input preparation, and workspace composition |
| `core/` | Editable document, alignment state/history, controller transitions, workflow results/policy, display settings, events, and snapshots |
| `runtime/` | Disposable loaded-stream, shank, histology-load, and slice-cache state |
| `io/` | Input datapackage validation and normalization, stream loaders, load jobs, input snapshots, and DocDB adapters |
| `services/` | Qt-free ephys/histology/slice operations, alignment math orchestration, persistence, output packaging, and CCF transforms |
| `geometry/` | Atlas, channel, probe-track, perpendicular-slice, and warp geometry |
| `plotting/` | Declarative plot registry, menu state, payload builders/caches, and channel-layout logic; no widgets |
| `desktop/` | Qt shell, concrete views/displays, action coordinators, render presenters, worker runners, dialogs, and teardown |

`launch_gui.py` is a process composition boundary. Its Qt import does not make
it part of the reusable engine.

## State Ownership

Four state owners must remain distinct.

### AlignmentDocument

`core/document.py` owns durable user work and workflow selection:

- mouse, recording, probe, and shank selection;
- output-root and output-package identity;
- per-`AlignmentKey` editable `AlignmentState` values;
- active feature/track control points, pending reference lines, edit history,
  previous loaded alignments, and save revisions;
- loaded/dirty flags needed by commands and recovery.

It contains no Qt objects, image volumes, spike arrays, plot payloads, or file
services. It is the source of truth for autosave and full-save scope.

`AlignmentKey = (recording_id, ephys_collection, shank_idx)` identifies one
alignment unit. `shank_idx` is zero-based internally. Display names and probe
IDs are not stable substitutes for this key.

### AlignmentDataContext

`io/alignment_data_context.py` owns normalized facts from the selected input
datapackage:

- the `MouseRoot` and immutable `InputDatasetSnapshot`;
- selected `ProbeInfo` and channel-table metadata;
- recording/probe/shank lookup and external asset resolution.

This context does not own edits or dirty state. Save-time geometry is derived
from its input snapshot through `SaveGeometryCatalog`, rather than by scanning
the filesystem or loading interactive stream runtimes.

### Runtime State

`SessionRuntime` owns active and cached `EphysStreamRuntime` objects. A stream
runtime owns its ephys arrays, shank views, and derived plot payload caches.
Runtime state is an accelerator: it may be evicted without losing user edits.

The stream cache is LRU-bounded by `EPHYS_ALIGNMENT_MAX_CACHED_STREAMS`
(default `3`). Evicting a stream clears its derived caches. Histology is
mouse-scoped and reused across streams for the same mouse root. Slice and
histology-channel caches are similarly disposable.

### Desktop View State

Desktop views and displays own widgets, menus, pyqtgraph items, axes, popup
lifetimes, reference-line handles, and signal connections. `MainWindow` owns
composition and top-level shutdown only. It must not become a source of domain
state or runtime data.

Frontend-independent display choices, such as the unit filter and annotation
source, live in core display state and are changed through app commands.

## Application Boundary

`AlignmentWorkspace` is the Qt-free composition root. It builds the document,
controller, metadata context, runtime, services, command handlers, queries, and
event bus. Frontends receive its narrow `AlignmentApp` port:

```python
app.commands  # mutate or sequence a use case
app.queries   # return plain rendering/read-model data
app.events    # typed semantic event bus
```

Desktop code should not reach through the workspace to its document, runtime,
controller, or services. `MainWindow` retains `_workspace` only to own the
composition root's lifetime and passes `workspace.app` into the workbench.

### Commands

A command represents user intent and may mutate document state, coordinate IO,
update runtime caches, and publish events. Non-trivial commands are split into
focused handlers under `application/commands/` for metadata selection, paths,
shanks, loading, loaded-shank preparation, alignment editing, display state,
autosave, and persistence.

Commands establish policy and sequence side effects. They use
`AlignmentController` for validated document/domain transitions rather than
mutating the document ad hoc.

### Controller

`AlignmentController` is document-mutation authority. It applies validated
selection and alignment state transitions and delegates pure alignment edits to
domain services. It does not own Qt presentation, worker threads, runtime cache
lifecycle, repositories, or event publication.

### Queries

Queries answer what the frontend should render. They return plain dataclasses,
arrays, mappings, and scalar values. They may materialize explicitly named
runtime caches, such as a plot payload or slice, but they must not alter
document edit history or persist files.

### Events

Events report semantic outcomes such as alignment edits, shank changes, stream
activation, load progress, save progress, path changes, autosave recovery, and
display-setting changes. Events carry no Qt objects and should not become
generic "refresh everything" messages. Presenters respond by querying current
read models and updating focused views.

## Desktop Boundary

The desktop package is organized by role:

- `shell/` constructs the window, menus, handles, and top-level lifecycle;
- `workbench/` composes the desktop feature objects and exposes a narrow shell
  facade;
- `coordinators/` translate widget actions and dialogs into app commands;
- `presenters/` translate app events and query DTOs into render operations;
- `views/` own widgets and user-facing dialog/view behavior;
- `displays/` own pyqtgraph items and panel rendering;
- `workers/` adapt Qt-free jobs to QThread lifecycles;
- `actions/` contain focused desktop interaction helpers.

A coordinator may orchestrate a dialog and command call, but it must not
compute alignment math. A presenter may choose how to render a read model, but
it must not perform persistence or mutate the document. A view/display may own
Qt handles, but it must not reach into application internals.

## Selection And Load Lifecycle

Loading is selection-driven; there is no normal Load Data button path.

1. Selecting a mouse root validates `datapackage.json`, builds the normalized
   input snapshot, clears state belonging to a different root, and starts a
   mouse-scoped histology warmup.
2. The user selects a recording session. Probe selection remains blank so a
   cold load is not spent on an unwanted default stream.
3. Selecting a probe loads its lightweight channel table and automatically
   activates or loads shank zero. Selecting another shank activates that shank.
4. `SessionRuntime.plan_load_data()` resolves the request as already active,
   cached, or requiring fresh IO.
5. A fresh load runs as a Qt-free `LoadDataJob` through a desktop QThread
   runner. Result publication and activation return to the GUI thread.

After activation, the desktop may preload the next unloaded probe in the same
recording and warm selected plot payload caches. If the user selects the stream
currently being preloaded, the foreground operation promotes/joins that work
instead of duplicating IO. A foreground load likewise joins matching in-flight
or completed histology warmup.

Load IDs and mouse-root identity guard against stale worker results. Changing
mouse root is the hard cache/cancellation boundary; same-root recording and
probe navigation preserves useful cached runtimes and preloads.

## Threading And Shutdown

Long-running units of work are Qt-free jobs with typed requests, results,
progress callbacks, and cooperative cancellation tokens. Desktop runners own
QThreads and deliver results to the GUI thread. Current threaded work includes
fresh loads, speculative stream preloads, plot-payload warmup, and full save.

Rules:

- workers must not touch widgets or pyqtgraph objects;
- cancellation is cooperative and may wait for non-interruptible native IO;
- late or stale results must be ignored by lifecycle identity checks;
- a QThread object must remain referenced until the native thread has stopped;
- cleanup callbacks must not destroy their own still-running QThread;
- application shutdown requests cancellation and keeps the event loop alive
  behind a shutdown dialog until all runners can be finalized.

The save worker launches ANTs point transforms in a subprocess by default. The
parent polls the cancellation token and terminates, then kills if necessary,
the transform process on cancellation. Temporary request, array, stdout, and
stderr files live in a scoped temporary directory and are removed when the
operation exits. `EPHYS_ALIGNMENT_ANTS_POINTS_SUBPROCESS=0` is a diagnostic
escape hatch that restores in-process execution and therefore loses
mid-transform cancellation.

## Plotting And Depth Displays

Ephys plots are declared as `PlotSpec` values in `plotting/registry.py`.
Builders produce plain payloads and the registry derives menu entries. Dynamic
availability checks should inspect required inputs cheaply; opening a menu must
not compute every candidate plot.

Plot payload caches belong to each stream/shank runtime and are additionally
keyed where display choices, such as unit filtering, change the payload. Cache
eviction follows stream-runtime eviction. Desktop displays convert payloads
into pyqtgraph items and own their teardown.

Depth plots use physical channel depth. Channel row/index order is never a
valid proxy for depth, especially for split-bank or irregular channel maps.

The image, line, probe, warped annotation, original annotation, scale, ruler,
and perpendicular-slice plots share one depth viewport. Their actual ViewBox
areas must remain vertically coaxial. Plot-local `PlotItem` titles add a layout
row and must not be used on these panels. Use the fixed bottom-axis strip labels
from `desktop/displays/depth_panel_layout.py`; tests there protect this
constraint.

## Alignment Semantics

Reference-line capture and fitting use three distinct coordinate concepts:
feature depth, raw track depth, and warped display depth. The exact behavior and
idempotence invariants are specified in
`docs/reference_line_alignment_contract.md`. Read that contract before changing
reference lines, fitting, undo/redo, or depth rendering.

Core warp math lives in `geometry/ephys_alignment.py`. Desktop line handles are
only interaction/rendering objects; pending line coordinates and committed
alignments belong to document state.

## Save, Autosave, And Recovery

### Output Package

Choosing an output root creates one mouse-level annotation package:

```text
<output-root>/ibl_annotations_<mouse-id>_<timestamp>/
    datapackage.json
    autosave/alignment_document.json
    <recording-id>/<probe-name>/
        channel_locations*.json
        ccf_channel_locations*.json
        prev_alignments*.json
        alignment_output_metadata*.json
```

Multi-shank suffixes are applied where needed. The output manifest is authored
from Pydantic models by the development script and validated at runtime against
the packaged JSON Schema.

### Autosave

Autosave is a cheap, atomic serialization of `AlignmentDocument`; it performs
no ephys loading, histology loading, plotting, or ANTs work. Commands checkpoint
after alignment/import changes and before navigation transitions when an output
package exists. A successful full save clears the checkpoint.

Recovery is explicit through `File -> Recover Autosave...`. The user first
loads the matching mouse root. Recovery validates mouse and alignment keys,
writes `alignment_document.pre_restore.json` when replacing meaningful live
state, restores valid states into the existing document, marks runtime data
unloaded, and activates the recovered selection through the normal desktop
selection path.

### Full Save

Full save writes every saveable document state, not only the active or dirty
state. A state is saveable once it has an active alignment, including a visited
initialized alignment or a previously loaded alignment. Unvisited states with
no active alignment are reported before save and are not fabricated.

Save preparation uses the input dataset snapshot, `SaveGeometryCatalog`, xyz
picks, atlas context, channel identity, and document control points. It does not
rehydrate full ephys stream runtimes. CCF/ANTs point transforms are batched
across the transaction. Files are committed per key with progress and
cancellation checks.

Missing save-critical metadata fails visibly before document history is marked
saved. A CCF transform or validation problem is handled differently because
user data are more important than the derived CCF product: anatomical channel
locations and alignment history are still written, while CCF status and issues
are recorded in `alignment_output_metadata*.json` and shown as warnings.

DocDB is optional. A server-side DocDB failure disables further DocDB writes in
the same save batch without preventing local output files from being written.

## Data Contracts

The GUI reads a mouse-root `datapackage.json`; it does not discover sessions by
directory globbing. Input schema majors 3 and 4 are supported. Exact bundled
schemas are under `io/schemas/aind-ibl-ephys-alignment-datapackage/`; newer minor
versions in a supported major are validated against the newest compatible
bundled schema with the version const relaxed.

Paths and transform frames must be normalized at the IO/service boundary.
Notably, CCF export detects whether image-to-template transforms expect
SPIM-native or pipeline geometry from the transform sidecar, defaulting to the
pipeline frame when the sidecar is unavailable.

Saved channel rows carry stable output identity (`raw_ind`, `contact_id`, and
`shank_idx`) rather than relying on row order alone.

## Extension Rules

When adding a feature:

- new user intent or side-effect sequencing: add a focused app command handler;
- new render state: add a focused query returning a plain DTO;
- new semantic outcome needed by multiple desktop consumers: add a typed event;
- new ephys plot: add a registry `PlotSpec` and plain payload builder;
- new Qt behavior or dialog: add it under the appropriate desktop role;
- new persistent edit state: put it in the document and snapshot contract;
- new heavy derived data: put it in a runtime cache with explicit invalidation;
- new producer format: normalize it in `io/` and keep downstream code on one
  internal representation.

Do not add behavior by giving a view access to `AlignmentWorkspace`, by storing
arrays in the document, by putting Qt values in query DTOs, or by flattening
unrelated operations onto `MainWindow`, `DesktopWorkbench`, or `AlignmentApp`.

## Verification

The supported development commands are:

```bash
uv sync --group dev
uv run ruff check src tests
uv run pytest -q
```

CI runs the same lint and test commands on Python 3.10 for Linux, macOS, and
Windows. Desktop tests use fakes for widgets and a `QCoreApplication` event loop
for worker lifecycle tests; they do not require a display server.

Focused tests should be added at the owning boundary. Cross-layer tests are
appropriate for high-value workflows such as selection-driven load, stale
worker rejection, autosave recovery, no-runtime full save, CCF fallback, and
reference-line idempotence.

Open correctness and product work is tracked in `TODO.md`. The TODO is not an
architecture authority; completed implementation belongs in this document and
tests, not in a growing migration diary.
