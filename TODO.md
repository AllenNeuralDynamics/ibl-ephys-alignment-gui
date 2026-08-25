# Open Work

Last reconciled against `main`: 2026-08-25.

This file contains unresolved correctness and product work only. Implemented
architecture belongs in `docs/architecture.md`; completed work is available in
Git history and should not accumulate here.

## Selection Guidance

Selection-driven loading intentionally leaves the probe selector blank after a
recording is chosen. Make the required sequence clear without presenting an
error state:

- highlight the session selector while no recording is selected;
- after recording selection, highlight the probe/stream selector;
- after stream activation, highlight the shank selector only when an explicit
  shank choice is still required;
- use placeholder text such as `Select session` and `Select stream` plus a
  blue/amber focus treatment; reserve red for invalid or failed states;
- derive this desktop presentation state from app selection/load read models,
  not from load-command logic.

Acceptance: the next required selection is obvious after mouse-root load, and
guidance clears when work starts or the target is already loaded.

## Alignment Extrapolation Scale

Allow users to set a default feature-to-track scale for unconstrained portions
of a shank warp.

Semantics:

- treat the value as an extrapolation prior, not a synthetic correspondence;
- store it per `AlignmentKey` so multi-shank probes may diverge;
- keep a global last-used value that seeds newly created alignment states only;
- with no correspondences, anchor the warp at the probe tip:
  `track = tip_track + scale * (feature - tip_feature)`;
- with correspondences, use the prior only outside the constrained span,
  anchored to the nearest real point;
- when Linear fit is enabled, linear-fit extrapolation takes precedence and
  the control is disabled;
- repeated fitting with unchanged points and scale must be idempotent;
- autosave and full save must reproduce the same materialized warp.

UI proposal: place an `Extrapolation scale` numeric control beside Linear fit,
defaulting to `1.0`. Changing it updates the active shank and the seed for
future states, but not already visited states.

Tests must cover zero, one, and multiple correspondences; linear-fit priority;
idempotence; autosave restore; and propagation to new versus existing keys.

## CCF Export Hardening

Current behavior detects the image-to-template input frame from a transform
sidecar when available, defaults to pipeline geometry when absent, batches
transforms, and preserves anatomical/alignment output when CCF export fails.

Remaining work:

- validate final CCF coordinates against the actual fixed-grid header of
  `template_to_ccf_warp`, and include its observed bounding box in warnings;
- improve absent-sidecar frame detection using affine/header domain overlap
  against SPIM-native and pipeline candidates;
- record the selected transform input frame and decision reason in output
  metadata for auditability;
- run real-asset acceptance checks for a standard SmartSPIM pipeline mouse and
  a `registration_override` mouse.

Acceptance: frame/origin mistakes are diagnosed from real transform geometry,
the chosen frame is auditable, and user alignment/anatomical output remains
available even when CCF output is unsafe.

## Firing-Rate Spike-Sorting Support

The firing-rate image cannot yet distinguish no spikes from channels/chunks
that were never spike sorted. Recorded channel geometry and observed unit depth
are both invalid proxies.

Remaining work:

- determine whether preprocessing already emits sorted-channel support for each
  Open Ephys chunk;
- if absent, extend the producer/datapackage contract with a chunk-aware support
  sidecar containing stable channel identity (`row`, `raw_ind`, `contact_id`,
  and shank where available);
- consume that metadata in the FR payload's no-data mask;
- keep the fallback conservative when support cannot be proven.

Acceptance: unsupported chunk/channel depths are visibly distinct; sorted
channels with zero accepted units remain supported; and an unsorted recording
is distinguishable from a sorted recording with low firing rate.

## Performance Follow-Up

- Profile full save on real 20+ probe/session packages. Improve progress detail
  only where users cannot identify the slow phase.
- Revisit speculative preload ordering only if timing traces show that
  same-session-next-probe is a poor predictor.
- Consider visible histology-warmup status only if users mistake joined cold
  loads for hangs.

## Developer Infrastructure

- Update `.github/workflows/ci.yml` to trigger on `main` (or confirm and
  document an intentional `master`/`develop` branch policy). The current
  workflow does not run for pushes or pull requests targeting `main`.
- Decide whether `ruff format --check` and static typing should become gates in
  dedicated cleanup slices. They currently have known debt and are not part of
  CI.
