from __future__ import annotations

import json
import logging
import re

# temporarily add this in for neuropixel course
# until figured out fix to problem on win32
import ssl
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import ants
import numpy as np
import one.alf.io as alfio
import pandas
import SimpleITK as sitk
from iblatlas import atlas
from iblatlas.regions import BrainRegions
from iblutil.util import Bunch
from numpy.typing import NDArray

from ephys_alignment_gui.alignment_data_context import AlignmentDataContext
from ephys_alignment_gui.anatomical_atlas import (
    _BLESSED_DIRECTION,
    BrainAtlasAnatomical,
)
from ephys_alignment_gui.datapackage_loader import (
    DataPackageError,
    MouseRoot,
    ProbeInfo,
    load_mouse_root,
)
from ephys_alignment_gui.ephys_data_service import (
    ChannelCollectionView,
    ChannelTable,
    EphysDataService,
    EphysStreamData,
)
from ephys_alignment_gui.rigid_rotation import (
    image_center_physical,
    load_affine_matrix,
    polar_rotation,
    rotate_image,
)
from ephys_alignment_gui.slice_service import SliceService

ssl._create_default_https_context = ssl._create_unverified_context
logger = logging.getLogger(__name__)

ANTS_DIMENSION = 3


@dataclass
class LoadDataLocal:
    """Loader driven by a preprocessed mouse-root directory.

    The entry-point is :meth:`set_mouse_root`, which reads ``datapackage.json``
    and surfaces the sessions and probes available. :meth:`select_probe` picks
    one for subsequent loading. All file paths come from the datapackage; the
    loader makes no assumptions about directory layout beyond that contract.
    """

    data_context: AlignmentDataContext | None = None
    mouse_root: MouseRoot | None = None
    probe_info: ProbeInfo | None = None
    brain_atlas: BrainAtlasAnatomical | None = None
    chn_coords: NDArray | None = None
    chn_coords_all: NDArray | None = None
    chn_contact_id_all: NDArray | None = None
    chn_shank_ind_all: NDArray | None = None
    n_shanks: int = 0
    channel_table: ChannelTable | None = None
    ephys_stream: EphysStreamData | None = None
    channel_collection: ChannelCollectionView | None = None
    ephys_data_service: EphysDataService = field(default_factory=EphysDataService)
    slice_service: SliceService = field(default_factory=SliceService)

    histology_images: dict[str, sitk.Image] = field(default_factory=dict)
    channel_dict: dict[str, dict[str, Any]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Mouse-root / probe selection
    # ------------------------------------------------------------------

    def _mouse_root(self) -> MouseRoot | None:
        if self.data_context is not None:
            return self.data_context.mouse_root
        return self.mouse_root

    def _probe_info(self) -> ProbeInfo | None:
        if self.data_context is not None:
            return self.data_context.probe_info
        return self.probe_info

    def _channel_table(self) -> ChannelTable | None:
        if self.data_context is not None:
            return self.data_context.channel_table
        return self.channel_table

    def _n_shanks(self) -> int:
        if self.data_context is not None:
            return self.data_context.n_shanks
        return self.n_shanks

    def _clear_channel_cache(self) -> None:
        """Clear legacy channel adapter fields derived from a selected stream."""
        self.chn_coords = None
        self.chn_coords_all = None
        self.chn_contact_id_all = None
        self.chn_shank_ind_all = None
        self.channel_table = None
        self.n_shanks = 0
        self.ephys_stream = None
        self.channel_collection = None

    def _cache_channel_table_arrays(self, channel_table: ChannelTable) -> None:
        """Refresh legacy array views from canonical channel metadata."""
        self.chn_coords_all = channel_table.local_coordinates
        self.chn_contact_id_all = channel_table.contact_ids
        self.chn_shank_ind_all = channel_table.shank_indices

    def reset_for_mouse_root_selection(self, *, root_changed: bool) -> None:
        """Clear loader-side caches after the selected mouse root changes."""
        if root_changed:
            self.brain_atlas = None
            self.histology_images = {}
            if hasattr(self, "_lazy_channel_paths"):
                delattr(self, "_lazy_channel_paths")
        self._clear_channel_cache()

    def reset_for_probe_selection(self) -> None:
        """Clear loader-side stream caches after the selected probe changes."""
        self._clear_channel_cache()

    def set_mouse_root(self, mouse_root: Path) -> MouseRoot:
        """Load a mouse-root directory. Resets probe-specific state.

        Parameters
        ----------
        mouse_root : Path
            Directory containing ``datapackage.json``.

        Returns
        -------
        MouseRoot
            Resolved view of the mouse-root.
        """
        logger.info(f"set_mouse_root: {mouse_root}")
        if self.data_context is not None:
            old_root = self.data_context.mouse_root
            mr = self.data_context.set_mouse_root(Path(mouse_root))
        else:
            old_root = self.mouse_root
            mr = load_mouse_root(Path(mouse_root))
            self.mouse_root = mr
            self.probe_info = None
        self.reset_for_mouse_root_selection(
            root_changed=old_root is not None and old_root.root != mr.root
        )
        return mr

    def list_sessions(self) -> list[str]:
        """Recording IDs available in the current mouse root."""
        mouse_root = self._mouse_root()
        if mouse_root is None:
            raise RuntimeError("No mouse root loaded — call set_mouse_root() first")
        return mouse_root.sessions

    def list_probes(self, recording_id: str) -> list[str]:
        """Probe names for a given recording in the current mouse root."""
        mouse_root = self._mouse_root()
        if mouse_root is None:
            raise RuntimeError("No mouse root loaded — call set_mouse_root() first")
        return mouse_root.probes_for_session(recording_id)

    def select_probe(self, recording_id: str, probe_name: str) -> ProbeInfo:
        """Select a probe for loading. Resets per-probe data caches."""
        mouse_root = self._mouse_root()
        if mouse_root is None:
            raise RuntimeError("No mouse root loaded — call set_mouse_root() first")
        if self.data_context is not None:
            probe = self.data_context.select_probe(recording_id, probe_name)
        else:
            probe = mouse_root.get_probe(recording_id, probe_name)
            self.probe_info = probe
        logger.info(
            f"select_probe: recording={recording_id!r}, probe={probe_name!r}, "
            f"num_shanks={probe.num_shanks}, ephys_dir={probe.ephys_dir}"
        )
        self.reset_for_probe_selection()
        if self.data_context is None:
            self.n_shanks = probe.num_shanks
        return probe

    @property
    def probe_id(self) -> str | None:
        """Shortcut for the current probe ID (if selected)."""
        probe = self._probe_info()
        return probe.probe_id if probe is not None else None

    # ------------------------------------------------------------------
    # Channel info / ephys / atlas loading
    # ------------------------------------------------------------------

    def load_channel_info(self) -> None:
        """Load channel local coordinates from the selected probe's ephys ALF."""
        probe = self._probe_info()
        if probe is None:
            raise RuntimeError("No probe selected — call select_probe() first")
        channel_table = self.ephys_data_service.load_channel_table(probe)
        self._set_channel_table(channel_table)
        self.ephys_stream = None
        self.channel_collection = None

        geom_n_shanks = channel_table.n_shanks
        if geom_n_shanks != probe.num_shanks:
            logger.warning(
                "Channel table implies %d shanks but datapackage says %d; "
                "trusting channel table.",
                geom_n_shanks,
                probe.num_shanks,
            )
        if self.data_context is None:
            self.n_shanks = geom_n_shanks

    def set_ephys_stream(self, stream: EphysStreamData) -> None:
        """Attach an already-loaded runtime stream to this loader adapter."""
        probe = self._probe_info()
        if probe is None:
            raise RuntimeError("No probe selected — call select_probe() first")
        if stream.recording_id != probe.recording_id:
            raise ValueError(
                "Cached stream recording does not match selected recording: "
                f"{stream.recording_id!r} != {probe.recording_id!r}"
            )
        if stream.ephys_collection != probe.ephys_collection:
            raise ValueError(
                "Cached stream collection does not match selected collection: "
                f"{stream.ephys_collection!r} != {probe.ephys_collection!r}"
            )
        self.ephys_stream = stream
        self._set_channel_table(stream.channel_table)
        self.channel_collection = None

    def _set_channel_table(self, channel_table: ChannelTable) -> None:
        """Update legacy channel-table adapter fields from a runtime model."""
        if self.data_context is not None:
            self.data_context.attach_channel_table(channel_table)
        else:
            self.channel_table = channel_table
        self._cache_channel_table_arrays(channel_table)
        if self.data_context is None:
            self.n_shanks = channel_table.n_shanks

    def get_shank_list(self) -> list[str] | None:
        """Build the shank-picker list for the current probe."""
        n_shanks = self._n_shanks()
        if n_shanks == 1:
            return ["1/1"]
        if n_shanks > 1:
            return [f"{i + 1}/{n_shanks}" for i in range(n_shanks)]
        return None

    def load_atlas_and_histology(self) -> None:
        """Load atlas + default histology channel from the mouse-root datapackage.

        Applies the SPIM->template polar rotation so all image-space assets
        share an atlas-aligned canonical frame. SPIM-native versions of the
        intensity and pipeline images are kept on the atlas for the ANTs CCF
        chain (which was computed in SPIM-native coords).
        """
        mouse_root = self._mouse_root()
        if mouse_root is None:
            raise RuntimeError("No mouse root loaded — call set_mouse_root() first")
        hist = mouse_root.histology
        logger.debug(f"Loading atlas and histology from {hist.registration.parent}")

        intensity_image = sitk.ReadImage(str(hist.ccf_template))
        label_image = sitk.ReadImage(str(hist.labels))
        pipeline_image = sitk.ReadImage(str(hist.registration_pipeline))
        histology_image = sitk.ReadImage(str(hist.registration))

        # Extract the rotational part of the SPIM->template affine and apply
        # it to every image-space asset, so the canonical in-memory frame has
        # atlas-aligned axes. SPIM-native recovery (for saving xyz_picks and
        # composing with the ANTs CCF chain) is done via R^T through the
        # BrainAtlasAnatomical.unrotate_to_spim_native helper.
        linear, _ = load_affine_matrix(
            mouse_root.transforms.image_to_template_affine
        )
        # An ANTs 0GenericAffine.mat maps points fixed->moving. The
        # ls_to_template registration has fixed=template, moving=SPIM, so this
        # linear part is the template->SPIM map. We want to rotate SPIM data
        # *into* the template-aligned canonical frame, i.e. the SPIM->template
        # rotation, which is the transpose (inverse) of the extracted rotation.
        # Applying it un-transposed rotated the histology (and the probe
        # points, which share this R via display_rotation) further from the
        # atlas instead of toward it.
        R = polar_rotation(linear).T
        rotation_center = image_center_physical(intensity_image)
        logger.debug(
            f"Computed SPIM->template display rotation (det={np.linalg.det(R):.6f})"
        )

        intensity_image_rot = rotate_image(
            intensity_image, R, rotation_center, interpolator="linear"
        )
        label_image_rot = rotate_image(
            label_image, R, rotation_center, interpolator="nearest"
        )
        pipeline_image_rot = rotate_image(
            pipeline_image, R, rotation_center, interpolator="linear"
        )
        histology_image_rot = rotate_image(
            histology_image, R, rotation_center, interpolator="linear"
        )

        self.brain_atlas = BrainAtlasAnatomical(
            intensity_img=intensity_image_rot,
            label_img=label_image_rot,
            pipeline_img=pipeline_image_rot,
            display_rotation=R,
            display_rotation_center=rotation_center,
            intensity_img_spim_native=intensity_image,
            pipeline_img_spim_native=pipeline_image,
        )

        # Ensure the rotated histology is in the blessed DICOM orientation
        # consumed by the rest of the pipeline (rotate_image emits identity
        # direction, so DICOMOrient only does a cheap axis permutation).
        dicom_orient_str = (
            sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(
                histology_image_rot.GetDirection()
            )
        )
        if dicom_orient_str != _BLESSED_DIRECTION:
            histology_image_rot = sitk.DICOMOrient(
                histology_image_rot, _BLESSED_DIRECTION
            )
        self.histology_images["histology_registration"] = histology_image_rot

        # Store metadata for lazy loading other channels. They'll be rotated
        # with the same (R, center) when first requested; per-channel DICOM
        # reorient is decided at load time.
        self._lazy_channel_paths = dict(hist.additional_channels)
        logger.debug(f"Setup lazy loading for {len(self._lazy_channel_paths)} channels")

    def set_channels_for_shank(self, shank_idx: int) -> NDArray:
        """Filter cached channel coordinates for selected shank. No disk I/O."""
        channel_table = self._channel_table()
        probe = self._probe_info()
        if channel_table is None:
            raise RuntimeError("Must call load_channel_info() first")
        if probe is None:
            raise RuntimeError("No probe selected — call select_probe() first")
        self._cache_channel_table_arrays(channel_table)

        if self.ephys_stream is not None:
            collection = self.ephys_stream.channel_collection(shank_idx)
        else:
            rows = channel_table.rows_for_shank(shank_idx)
            collection = ChannelCollectionView(
                stream=EphysStreamData(
                    recording_id=probe.recording_id,
                    ephys_collection=probe.ephys_collection,
                    ephys_dir=probe.ephys_dir or Path(),
                    channel_table=channel_table,
                    alf_data={},
                    session_notes="",
                    probe_id=probe.probe_id,
                    probe_name=probe.probe_name,
                    logical_probe=probe.logical_probe,
                ),
                shank_idx=shank_idx,
                rows=rows,
            )

        self.channel_collection = collection
        chn_coords = collection.local_coordinates
        self.chn_coords = chn_coords

        return collection.depths

    def get_ephys_data(
        self, shank_idx: int
    ) -> tuple[Path, NDArray, str, dict[str, Any]]:
        """Load ephys ALF for the current probe + shank.

        Returns
        -------
        tuple
            ``(ephys_dir, chn_depths, sess_notes, data)``. The ``ephys_dir`` is
            what downstream plot code stores as ``probe_path`` (it contains
            ``band_corr/`` etc.).
        """
        probe = self._probe_info()
        channel_table = self._channel_table()
        if probe is None:
            raise RuntimeError("No probe selected — call select_probe() first")
        if probe.ephys_dir is None:
            raise DataPackageError(
                f"Probe {probe.probe_name!r} has no ephys dir"
            )
        if channel_table is None:
            raise RuntimeError("Must call load_channel_info() first")
        self._cache_channel_table_arrays(channel_table)

        if self.ephys_stream is None:
            self.ephys_stream = self.ephys_data_service.load_stream_data(
                probe,
                channel_table=channel_table,
            )

        collection = self.ephys_stream.channel_collection(shank_idx)
        self.channel_collection = collection
        self.chn_coords = collection.local_coordinates

        return (
            self.ephys_stream.ephys_dir,
            collection.depths,
            self.ephys_stream.session_notes,
            self.ephys_stream.alf_data,
        )

    def load_allen_csv(self):
        allen_path = Path(Path(atlas.__file__).parent, "allen_structure_tree.csv")
        self.allen = alfio.load_file_content(allen_path)
        return self.allen

    def get_track_annotations(self, shank_idx: int) -> NDArray[np.floating]:
        """Read xyz-picks (image space) for the current probe + shank."""
        probe = self._probe_info()
        if probe is None:
            raise RuntimeError("No probe selected — call select_probe() first")
        picks = probe.picks_for_shank(shank_idx)
        path = picks.image_space
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing probe trajectory file: {path}. "
                "This file must contain probe insertion coordinates in image space."
            )
        with open(path) as f:
            user_picks = json.load(f)

        # xyz_picks on disk are SPIM-native image-space RAS, in microns.
        # The rest of the GUI operates in the rotated canonical frame, so
        # rotate into canonical here. If no rotation is configured the helper
        # returns the input unchanged.
        track_annotations_ras_spim = np.array(user_picks["xyz_picks"]) / 1e6
        if self.brain_atlas is None:
            raise RuntimeError(
                "brain_atlas not yet loaded; call load_atlas_and_histology() first"
            )
        return self.brain_atlas.rotate_to_canonical(track_annotations_ras_spim)

    # ------------------------------------------------------------------
    # Slice images
    # ------------------------------------------------------------------

    def get_slice_images(self, track_interpolation_ras):
        """Get atlas and histology slices for the current shank track."""
        if self.brain_atlas is None:
            raise RuntimeError(
                "brain_atlas not yet loaded; call load_atlas_and_histology() first"
            )
        return (
            self.slice_service.build_slice_set(
                brain_atlas=self.brain_atlas,
                histology_images=self.histology_images,
                lazy_channel_paths=getattr(self, "_lazy_channel_paths", {}),
                track_interpolation_ras=track_interpolation_ras,
            ),
            None,
        )

    def get_perpendicular_slice_image(
        self,
        ephysalign,
        feature_ref: NDArray,
        track_ref: NDArray,
        feature_grid_m: NDArray,
        channel_name: str = "histology_registration",
        extent_m: float = 500e-6,
        n_perp_samples: int = 41,
        sigma_samples: float = 2.0,
    ) -> NDArray[np.float64]:
        """Build the perpendicular slice image for the current alignment.

        See :func:`ephys_alignment_gui.perpendicular_slice.build_perpendicular_slice`
        for the full contract. This wrapper handles channel resolution (lazy
        loading via :meth:`_load_and_slice_channel`) and sources the atlas
        volume that matches the blessed orientation used by the rest of the
        GUI.

        Returns
        -------
        NDArray (n_perp_samples, len(feature_grid_m))
            NaN for samples that fall outside the rotated histology volume.
        """
        if self.brain_atlas is None:
            raise RuntimeError("brain_atlas not yet loaded")

        return self.slice_service.build_perpendicular_slice_image(
            brain_atlas=self.brain_atlas,
            histology_images=self.histology_images,
            lazy_channel_paths=getattr(self, "_lazy_channel_paths", {}),
            ephysalign=ephysalign,
            feature_ref=feature_ref,
            track_ref=track_ref,
            feature_grid_m=feature_grid_m,
            channel_name=channel_name,
            extent_m=extent_m,
            n_perp_samples=n_perp_samples,
            sigma_samples=sigma_samples,
        )

    def get_region_description(self, region_idx):
        struct_idx = np.where(self.allen["id"] == region_idx)[0][0]
        description = ""
        region_lookup = (
            self.allen["acronym"][struct_idx] + ": " + self.allen["name"][struct_idx]
        )

        if region_lookup == "void: void":
            region_lookup = "root: root"

        if not description:
            description = region_lookup + "\nNo information available for this region"
        else:
            description = region_lookup + "\n" + description

        return description, region_lookup

    # ------------------------------------------------------------------
    # CCF transform + alignment result export
    # ------------------------------------------------------------------

    def _transform_to_ccf(
        self,
        channel_locations_ras: NDArray,
        channel_dict: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        mouse_root = self._mouse_root()
        if mouse_root is None or self.brain_atlas is None:
            raise RuntimeError(
                "Mouse root or brain atlas not loaded; cannot transform to CCF"
            )
        # Unrotate from the canonical (rotated) frame back to SPIM-native, then
        # use the pre-rotation (SPIM-native) sitk images for the index<->physical
        # math. The ANTs CCF chain was computed in SPIM-native coords and is
        # invalid for rotated inputs.
        channel_locations_ras_spim = self.brain_atlas.unrotate_to_spim_native(
            channel_locations_ras
        )
        histology_img = self.brain_atlas.intensity_sitk_image_spim_native
        pipeline_img = self.brain_atlas.pipeline_sitk_image_spim_native
        ras_to_lps = np.array([-1, -1, 1])
        # Convert IBL app world coordinates, RAS m, to ITK world coordinates, LPS mm
        channel_locations_lps_mm = 1e3 * ras_to_lps * channel_locations_ras_spim
        reg_pipeline_physical_points: list[list[float]] = []
        for point in channel_locations_lps_mm:
            index = histology_img.TransformPhysicalPointToContinuousIndex(point)
            pipeline_point = pipeline_img.TransformContinuousIndexToPhysicalPoint(index)
            reg_pipeline_physical_points.append(list(pipeline_point))

        reg_pipeline_physical_points_array = np.array(reg_pipeline_physical_points)

        logger.info("Warping to ccf")
        this_probe_df = pandas.DataFrame(
            reg_pipeline_physical_points_array, columns=list("xyz")
        )

        tx = mouse_root.transforms
        tx_list = [
            str(tx.image_to_template_affine),
            str(tx.image_to_template_warp),
            str(tx.template_to_ccf_affine),
            str(tx.template_to_ccf_warp),
        ]
        invert_list = [True, False, True, False]

        logger.info("applying transforms ...")
        ccf_coordinates_dataframe: pandas.DataFrame = ants.apply_transforms_to_points(
            ANTS_DIMENSION,
            this_probe_df,
            tx_list,
            whichtoinvert=invert_list,
        )
        logger.info("Done warping to ccf")

        ccf_channel_dict: dict[str, dict[str, Any]] = {}
        pattern = re.compile(r"channel_(\d+)")

        channel_indices = []
        channel_names = []
        for ch in channel_dict.keys():
            m = pattern.match(ch)
            if m:
                channel_indices.append(int(m.group(1)))
                channel_names.append(ch)

        xyz_array = ccf_coordinates_dataframe.loc[
            channel_indices, ["x", "y", "z"]
        ].to_numpy(dtype=np.float64)

        for ch, (x, y, z) in zip(channel_names, xyz_array):
            info = channel_dict[ch]
            ccf_channel_dict[ch] = {
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "axial": info["axial"],
                "lateral": info["lateral"],
                "brain_region_id": info["brain_region_id"],
                "brain_region": info["brain_region"],
            }
        return ccf_channel_dict

    def get_alignment_results(
        self,
        channel_locations_ras: NDArray,
        chn_coords: NDArray,
    ) -> tuple[
        dict[str, dict[str, Any]],
        dict[str, dict[str, Any]],
        bool,
    ]:
        """Compute the histology-space + CCF channel dicts for a save.

        IO-only: the alignment history itself is owned per-shank by
        :class:`~ephys_alignment_gui.shank_alignment.ShankAlignment` (the caller
        records the new alignment there and persists it), so this no longer
        keeps a resident ``alignments`` dict. ``chn_coords`` is passed in rather
        than read from loader scratch so a save is decoupled from loader state.
        """
        logger.info("Saving channel locations locally")
        logger.debug(f"Channels: {channel_locations_ras}")
        if self.brain_atlas is None:
            raise ValueError("Brain atlas not loaded, cannot save channel locations")
        regions: BrainRegions = self.brain_atlas.regions
        brain_regions = regions.get(self.brain_atlas.get_labels(channel_locations_ras))
        # Persist xyz in SPIM-native coords so external tools reading the
        # output don't need to know about the GUI's display rotation.
        brain_regions["xyz"] = self.brain_atlas.unrotate_to_spim_native(
            channel_locations_ras
        )
        brain_regions["lateral"] = chn_coords[:, 0]
        brain_regions["axial"] = chn_coords[:, 1]

        assert np.unique([len(brain_regions[k]) for k in brain_regions]).size == 1
        channel_dict = self.create_channel_dict(brain_regions)
        self.channel_dict = channel_dict

        ccf_channel_dict = self._transform_to_ccf(channel_locations_ras, channel_dict)

        multi_shank = self._n_shanks() > 1

        return channel_dict, ccf_channel_dict, multi_shank

    @staticmethod
    def create_channel_dict(brain_regions: Bunch) -> dict[str, dict[str, Any]]:
        """
        Create channel dictionary in form to write to json file
        :param brain_regions: information about location of electrode channels in brain atlas
        :type brain_regions: Bunch
        :return channel_dict:
        :type channel_dict: dictionary of dictionaries
        """
        channel_dict: dict[str, dict[str, Any]] = {}

        for i in range(brain_regions.id.size):
            channel = {
                "x": np.float64(brain_regions.xyz[i, 0] * 1e6),
                "y": np.float64(brain_regions.xyz[i, 1] * 1e6),
                "z": np.float64(brain_regions.xyz[i, 2] * 1e6),
                "axial": np.float64(brain_regions.axial[i]),
                "lateral": np.float64(brain_regions.lateral[i]),
                "brain_region_id": int(brain_regions.id[i]),
                "brain_region": brain_regions.acronym[i],
            }
            data = {"channel_" + str(i): channel}
            channel_dict.update(data)

        return channel_dict
