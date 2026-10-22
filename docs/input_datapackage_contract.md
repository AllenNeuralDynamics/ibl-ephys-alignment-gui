# Input Datapackage, Assets, And Ephys Collections

The GUI opens a mouse directory containing a preprocessing `datapackage.json`.
This input contract is separate from the saved alignment output schema. The
same input package should open in Code Ocean, on a desktop, or on another
computer by changing runtime asset locations without rewriting references.

## The Four Layers

| Layer | How it reaches the GUI |
|---|---|
| Identity and structure | Mouse metadata and `probes[recording_id][ephys_collection]` hierarchy |
| Asset descriptions | `external_assets` descriptors, independent of local mount locations |
| Location resolution | `{asset, path}` references plus consumer runtime roots/overrides |
| Collection contents | ALF/QC filenames, subdirectories, array ordering, and shank semantics |

The datapackage explicitly references transforms, histology, picks, ephys
directories, and channel tables. It does not enumerate all files inside the
ephys collections. One logical probe can span several collections; one
collection can contain several shanks. The collection's stable identity is
`(recording_id, ephys_collection)`.

The [producer datapackage guide](https://github.com/AllenNeuralDynamics/aind-ibl-ephys-alignment-preprocessing/blob/main/docs/external_asset_references.md)
documents fields and producer validation. The
[conversion collection contract](https://github.com/AllenNeuralDynamics/aind-ephys-ibl-gui-conversion/blob/main/docs/shank_channel_metadata_spec.md)
defines channel construction and metric artifacts.

## Asset Description Versus Location

`{"asset": null, "path": "..."}` resolves relative to the directory holding
the datapackage, regardless of the working directory. A non-null asset names
a logical key in `external_assets`; its path is relative to that asset root.
Descriptors require role/name and may record ID, URI, checksum, and provenance.
The producer's `platform` and provenance mount paths do not select locations.

`RootSearchResolver` locates already-present files; it does not download URIs
or verify checksums. It tries overrides by logical key, then recorded name.
An override points directly to an asset directory; a missing internal path
fails immediately. Otherwise each configured root is tried in order, by asset
name then ID within that root. Roots contain asset directories. Search is not
recursive, and there is no automatic `/data` fallback.

`IBL_ASSET_CONFIG` names a JSON file containing `asset_roots` and
`asset_overrides`. Explicit `load_mouse_root(asset_roots=...)` replaces roots;
otherwise nonempty `IBL_ASSET_ROOTS` replaces config-file roots. Overrides
merge config file, `IBL_ASSET_OVERRIDES`, then explicit function arguments;
later values win for identical keys. Lookup still prefers a logical key over
an asset-name key. The environment override value is a mapping directly,
without an outer `asset_overrides` key.

Use absolute config paths; `~` inside JSON strings is not expanded.
`IBL_ASSET_ROOTS` uses `:` on Linux/macOS and `;` on Windows; JSON arrays avoid
that platform difference.

## Code Ocean And Desktop Setup

On Code Ocean, attach the preprocessed asset and all external dependencies
named in its registry. Set `IBL_ASSET_ROOTS=/data` in the workstation session.
Select `/data/<preprocessed-asset>/<mouse_id>`, the directory containing the
JSON. Renamed mounts need overrides.

On a desktop or shared filesystem, copy the complete mouse tree and put
external assets in directories named after their registry names:

```text
/home/alice/alignments/mouse001/datapackage.json
/mnt/ibl-assets/SmartSPIM_mouse001_stitched/...
/mnt/ibl-assets/spim_template_to_ccf/...
```

Set `IBL_ASSET_CONFIG` to an absolute path to a config like:

```json
{
  "asset_roots": ["/mnt/ibl-assets"],
  "asset_overrides": {
    "SmartSPIM_mouse001_stitched": "/mnt/histology/renamed-mouse001"
  }
}
```

That override replaces root search for the named asset. Remove it if the asset
is stored under its recorded name. Prefer name-keyed overrides for multiple
subjects; a logical-key `smartspim` override applies across subjects. Windows
uses the same JSON structure with paths such as `D:/ibl-assets`. The portable
datapackage stays unchanged; the location config belongs to the deployment.

`EPHYS_ALIGNMENT_INPUT_ROOT` sets the initial chooser directory;
`EPHYS_ALIGNMENT_OUTPUT_ROOT` sets the default save root. Neither locates
external assets. See [README.md](../README.md#runtime-and-asset-configuration)
for runtime settings. The GUI capsule must supply defaults to the desktop
session; container `ENV` alone may not reach it.

## Ephys QC Lives Inside The Collection

The `ephys` reference resolves one collection directory. `EphysDataService`
loads spikes, clusters, channels, RMS, and PSD ALF objects there. It also
reads optional shank vectors, `session_notes.txt`, and per-row raw AP
event rates (`_iblqc_channels.rawActivity.npy`) there. LFP correlation builders read full/legacy matrices
and `row_channels.json` under `band_corr/`.

Channel-table paths are explicit, but moving only those files does not
redirect the ALF objects or `band_corr`. Preserve the entire collection when
copying or packing data. Changing a metric filename or subdirectory is a
compatibility change even if the JSON schema still passes. Histology
`emit_qc` controls separate CCF/Slicer outputs, not this ephys layout.

## Channel Rows, Shanks, And Histology Joins

Channel-table row position joins coordinates, metadata, RMS/PSD columns, and
full-matrix axes. `rawInd` is a positional placeholder in current conversion
output, not a hardware-channel lookup. Physical depth comes from
`localCoordinates[:, 1]` in micrometres, not row number.

The producer prefers physical prefixes in contact IDs (`s2e17`) for shank
identity, falling back to normalized SpikeInterface groups, then one shank.
The GUI validates the channel shank vector and renumbers sorted unique IDs
into local indices. For producer IDs
`[2, 2, 5, 5]`, local indices are `[0, 0, 1, 1]`. User labels are one-based.
Without valid explicit metadata, legacy geometry splitting uses x gaps and
column assumptions.

Pick selection normalizes `ephys_shank` values, falling back to legacy
`shank`; a one-shank entry uses its first pick. A physical
`histology_shank=3` can map to `ephys_shank=0` in a split stream. Filenames
can use physical one-based shank suffixes; do not infer ephys identity from
those names. Equal shank counts do not establish a correct join.

Full matrices are in table order. `row_channels.json` supplies per-shank table
positions sorted by depth then lateral coordinate. Select both matrix axes
with those positions and obtain plot coordinates from the table. Producer
JSON keys are producer shank IDs; legacy suffixes are those IDs plus one.
The current matrix reader tries local-index keys, one-based alternatives,
then filename fallbacks; it does not universally translate sparse producer
IDs. Successful channel selection does not establish full matrix
compatibility.

## Validation And Compatibility

JSON validation checks structure, not every collection file or array shape.
External resolution checks locations; loading and normalization check channel
arrays and individual products. Missing ALF objects warn and can disable plots.
Schema validity alone does not establish completeness or correct shank joins.

Supported input majors are 3 and 4. Bundled schemas are 3.0.0, 3.1.0, 3.2.0,
4.0.0, and 4.1.0. Other versions use the newest bundled schema in the same
major not newer than the input, with the version constant relaxed. Breaking
consumed fields requires a major bump. Array and filename changes also need
producer–consumer compatibility review even without a JSON change.

Regenerate old string-path datapackages with the producer rather than healing
`../../data` paths through Code Ocean-specific GUI logic. Review
`io/datapackage_loader.py`, `io/datapackage_schema.py`,
`services/ephys_data.py`, `geometry/channel_geometry.py`, and
`plotting/builders/lfp_correlation.py` when changing these contracts.
