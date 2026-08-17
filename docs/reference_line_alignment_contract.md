# Reference Line Alignment Contract

Status: developer contract for alignment reference lines and fit behavior.

## Spaces

The alignment editor has three depth coordinate concepts:

- Feature space: electrophysiology feature depth, shown by the image, line, and
  probe plots.
- Raw track space: depth along the histology track before the current
  feature-to-track warp is applied.
- Warped display space: raw track positions projected through the current
  track-to-feature warp so they can be shown beside feature-space plots.

Warped display space uses feature-space depth units on screen. A user dragging
a line in a warped panel is manipulating a displayed feature-depth coordinate,
but the selected semantic target is the raw track depth obtained by inverting
the current warp at that displayed coordinate.

## Reference Line Pairs

Each correspondence point is represented by two linked display groups:

- Feature line group: one coaxial line across the feature-space plots
  (image, line, and probe).
- Warped line group: one coaxial line across the warped-space plots
  (warped annotation and perpendicular slice).

The optional line on the original raw-track annotation panel is a passive echo
only. It is derived from the warped line group by applying the inverse current
warp, and it is not the coordinate captured for fitting.

The bottom-right fit plot shows the paired display coordinates:

- x: feature-space line depth.
- y: warped-space display line depth.

## Creation

Creating a new reference line from either a feature-space plot or a warped-space
plot must create a pair whose feature and warped display positions are initially
equal. The new pair should therefore appear coaxial across both display groups.

After creation, dragging the feature group changes only the feature coordinate
of the pair. Dragging the warped group changes only the warped display
coordinate of the pair, and updates the passive raw-track echo by inverting the
current warp.

## Fitting

Fit consumes paired display coordinates:

- `feature_positions_um`: feature-space line positions.
- `warped_positions_um`: warped display line positions.

To build the new piecewise-linear warp, the fit command converts each
`warped_positions_um` value through the previous feature-to-track warp to get a
raw track coordinate, then pairs that raw track coordinate with the matching
feature coordinate. The pairs are sorted by feature coordinate without breaking
their feature/raw-track association.

After fit, rendering the active alignment projects the raw track coordinates of
the active alignment back into warped display space. Therefore each existing
reference pair should again appear coaxial: the feature-side line and
warped-side line for each correspondence should have the same displayed depth.

## Invariants

- New reference line pairs are coaxial in display space.
- Feature-space plots share one line position per pair.
- Warped-space plots share one line position per pair.
- Raw-track annotation lines are passive derived echoes.
- Repeated fits with the same displayed correspondence pairs are idempotent:
  the active feature/track control points and the resulting warp do not change.
- Rendering from document state must distinguish warped display coordinates
  from raw track coordinates. They are not interchangeable.
