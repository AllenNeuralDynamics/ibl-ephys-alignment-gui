import logging

import numpy as np
import SimpleITK as sitk
from iblatlas.atlas import BrainAtlas, BrainCoordinates
from iblatlas.regions import BrainRegions
from iblutil.numerical import ismember
from numpy.typing import NDArray

_logger = logging.getLogger(__name__)

_BLESSED_DIRECTION: str = "IRP"


class BrainAtlasAnatomical(BrainAtlas):
    """
    BrainAtlas subclass for anatomical atlases built from anatomical images.
    In addition to the BrainAtlas intensity and label arrays, this class also
    stores the SimpleITK images for potential further processing.

    pipeline_sitk_image: sitk.Image
        The pipeline image associated with this atlas for CCF conversion.
    intensity_sitk_image: sitk.Image
        The anatomical intensity image.
    display_rotation, display_rotation_center: np.ndarray | None
        The rigid rotation (``R``) and its physical-space center applied to the
        input SPIM volumes to produce the rotated canonical frame this atlas
        represents. Downstream consumers apply ``R^T`` about this center to
        unrotate back to SPIM-native coordinates (saving xyz_picks, composing
        with the SPIM-native ANTs CCF transform chain). ``None`` means the
        atlas was built directly from SPIM-native images without rotation.
    """

    # Pipeline image associated with this atlas for CCF conversion.
    # After the canonical rotation has been applied upstream, this image is in
    # the rotated canonical frame (same as ``intensity_sitk_image``). The
    # SPIM-native version is kept as ``pipeline_sitk_image_spim_native`` for
    # downstream ANTs-chain coord math (which requires pre-rotation inputs).
    pipeline_sitk_image: sitk.Image
    # Anatomical intensity image in the canonical (rotated) frame.
    intensity_sitk_image: sitk.Image
    # Pre-rotation SPIM-native versions, retained for composition with the
    # SPIM-native ANTs CCF transform chain. Equal to the rotated images when
    # no rotation is configured.
    intensity_sitk_image_spim_native: sitk.Image
    pipeline_sitk_image_spim_native: sitk.Image
    # Rotation applied to go from SPIM-native to canonical frame (None == identity).
    # Stored in SimpleITK-native (LPS, mm) units; the rotate/unrotate helpers
    # handle the conversion to the IBL GUI's RAS-metres working frame.
    display_rotation: NDArray[np.float64] | None
    display_rotation_center: NDArray[np.float64] | None

    # LPS <-> RAS axis flip (X and Y flip sign).
    _LPS_RAS_FLIP: NDArray[np.float64] = np.diag([-1.0, -1.0, 1.0])

    def __init__(
        self,
        intensity_img: sitk.Image,
        label_img: sitk.Image,
        pipeline_img: sitk.Image,
        display_rotation: NDArray[np.float64] | None = None,
        display_rotation_center: NDArray[np.float64] | None = None,
        intensity_img_spim_native: sitk.Image | None = None,
        pipeline_img_spim_native: sitk.Image | None = None,
    ) -> None:
        """
        Initialize the BrainAtlasAnatomical class.

        Parameters
        ----------
        intensity_img : sitk.Image
            The anatomical intensity image.
        label_img : sitk.Image
            The label image corresponding to the anatomical image. This must be the
            in the same space, and have the same array shape as the intensity image.
            The labels should be the **lateralized** IBL brain region labels,
            with the left hemisphere being the negative indices and the right
            hemisphere being the positive indices.
        pipeline_img : sitk.Image
            The pipeline image associated with this atlas for CCF conversion.
            This should have the same array shape as the intensity image, but
            may be in a different physical space. Its physical space should be
            the one used by the CCF-registration pipeline, and allow conversion
            of the voxel indices to the physical domain of the pipeline-computed
            CCF transforms.
        """

        # Validate that intensity and label images have the same shape and physical space
        methods_to_check = [
            "GetOrigin",
            "GetSpacing",
            "GetSize",
            "GetDirection",
        ]
        for m in methods_to_check:
            intensity_val = np.array(getattr(intensity_img, m)())
            label_val = np.array(getattr(label_img, m)())
            if not np.allclose(intensity_val, label_val):
                raise ValueError(
                    f"Intensity and label {m}() mismatch: {intensity_val} != {label_val}"
                )

        # Reorient to _BLESSED_DIRECTION if needed
        orientation_code = (
            sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(
                intensity_img.GetDirection()
            )
        )
        if orientation_code == _BLESSED_DIRECTION:
            intensity_img_blessed = intensity_img
            label_img_blessed = label_img
            pipeline_img_blessed = pipeline_img
        else:
            _logger.info(
                f"Reorienting volume from {orientation_code} to {_BLESSED_DIRECTION} "
                "for consistency with IBL convention."
            )
            intensity_img_blessed = sitk.DICOMOrient(intensity_img, _BLESSED_DIRECTION)
            label_img_blessed = sitk.DICOMOrient(label_img, _BLESSED_DIRECTION)
            pipeline_img_blessed = sitk.DICOMOrient(pipeline_img, _BLESSED_DIRECTION)

        cosine_dir_mat = np.array(intensity_img_blessed.GetDirection()).reshape(3, 3)

        # Check that the image is arranged IS, LR, AP in memory (fortran order)
        if not np.allclose(
            np.abs(cosine_dir_mat),
            np.array(
                [
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 0, 0],
                ]
            ),
        ):
            raise ValueError(
                f"After reorientation to blessed direction {_BLESSED_DIRECTION}, "
                f"image direction cosines {cosine_dir_mat} are not as expected for "
                "IS, LR, AP DICOM arrangement. Is the blessed orientation wrong?"
            )

        # SimpleITK uses fortran order, but when converting to numpy the
        # underlying data are not changed and are interpreted by numpy as
        # C-order. This code accounts for that.

        # Going from C-order array data to XYZ (RAS) requires swapping the AP
        # and LR Axes. This code assumes that the _BLESSED_DIRECTION is being used
        # and it is something like SRA/IRP, which is how the IBL app wants to
        # arrange its data to make it easy to get coronal slices with numpy
        # indexing, and have them be compact in memory.
        dims2xyz = np.array([1, 0, 2])
        xyz2dims = np.array([1, 0, 2])

        # --- Compute dxyz (IBL spacing) from SimpleITK spacing ---
        # SimpleITK spacing should be in mm for these images

        # First get the WORLD spacing in mm LPS. img.GetSpacing() returns
        # spacing of ijk indices: have to use direction cosine matrix to
        # convert to world axes
        spacimg_mm_lps = cosine_dir_mat @ np.array(intensity_img_blessed.GetSpacing())
        # Now convert from LPS to RAS, and mm to m
        spacing_m_ras = 1e-3 * np.array([-1, -1, 1]) * spacimg_mm_lps

        # This is what IBL calls dxyz
        dxyz = spacing_m_ras

        # IBL defines origin by saying which index is at world coordinate 0,0,0
        # This is kind of broken for many images, and we'll manually override the
        # origin (called xyz0 in IBL) later. Here we just set it to 0,0,0
        iorigin = [0, 0, 0]

        # We can use BrainRegions from iblatlas because the labels are
        # lateralized IBL
        regions = BrainRegions()

        # Get arrays from SimpleITK images. I think there's no mutation risk
        intensity_img_sra_arr = sitk.GetArrayFromImage(intensity_img_blessed)
        label_img_sra_arr = sitk.GetArrayFromImage(label_img_blessed)

        # Need to convert these lateralized labels to IBL codes (input to their
        # mappings)
        _, im = ismember(label_img_sra_arr, regions.id)
        label = np.reshape(im.astype(np.int16), label_img_sra_arr.shape)

        # Initialize the superclass
        super().__init__(
            intensity_img_sra_arr,
            label,
            dxyz,
            regions,
            iorigin=iorigin,
            dims2xyz=dims2xyz,
            xyz2dims=xyz2dims,
        )

        # Need to account for the anatomical image origin not being at 0,0,0
        # SimpleITK is mm LPS, and IBL wants m RAS
        sitk_origin_ras_m = (
            np.array(intensity_img_blessed.GetOrigin()) * np.array([-1, -1, 1]) * 1e-3
        )
        nxyz = np.array(intensity_img_sra_arr.shape)[dims2xyz]
        self.bc = BrainCoordinates(nxyz=nxyz, xyz0=sitk_origin_ras_m, dxyz=dxyz)
        # Store the SimpleITK intensity image, and the pipeline image for use
        # with CCF transforms
        self.intensity_sitk_image = intensity_img_blessed
        self.pipeline_sitk_image = pipeline_img_blessed
        # Retain pre-rotation versions for the SPIM-native ANTs chain path.
        # If the caller didn't supply them, we weren't rotated — reuse the
        # already-blessed rotated images as both.
        self.intensity_sitk_image_spim_native = (
            intensity_img_spim_native
            if intensity_img_spim_native is not None
            else intensity_img_blessed
        )
        self.pipeline_sitk_image_spim_native = (
            pipeline_img_spim_native
            if pipeline_img_spim_native is not None
            else pipeline_img_blessed
        )
        # Rotation applied upstream to get here (None means identity).
        if (display_rotation is None) != (display_rotation_center is None):
            raise ValueError(
                "display_rotation and display_rotation_center must both be set "
                "or both be None"
            )
        self.display_rotation = (
            None if display_rotation is None else np.asarray(display_rotation, dtype=np.float64)
        )
        self.display_rotation_center = (
            None
            if display_rotation_center is None
            else np.asarray(display_rotation_center, dtype=np.float64)
        )

    def _rotation_ras_m(self) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
        """Return (R_ras, center_ras_m) in the IBL GUI's native frame.

        Derived from the stored LPS-mm (R, c) via
        ``R_ras = S @ R_lps @ S`` and ``c_ras_m = S @ c_lps_mm * 1e-3``
        where ``S = diag(-1, -1, 1)``. Returns ``None`` when no rotation is
        configured (identity case).
        """
        if self.display_rotation is None or self.display_rotation_center is None:
            return None
        S = self._LPS_RAS_FLIP
        R_ras = S @ self.display_rotation @ S
        center_ras_m = (S @ self.display_rotation_center) * 1e-3
        return R_ras, center_ras_m

    def unrotate_to_spim_native(self, points_ras_m: np.ndarray) -> np.ndarray:
        """Map RAS-metre points from the canonical (rotated) frame back to SPIM native.

        For atlases built without rotation this is the identity. Used at I/O
        boundaries: saving xyz_picks back to disk, composing with the
        SPIM-native ANTs CCF transform chain.
        """
        ras = self._rotation_ras_m()
        pts = np.asarray(points_ras_m, dtype=np.float64)
        if ras is None:
            return pts
        R_ras, c_ras = ras
        return (pts - c_ras) @ R_ras + c_ras

    def rotate_to_canonical(self, points_ras_m: np.ndarray) -> np.ndarray:
        """Map SPIM-native RAS-metre points into the canonical (rotated) frame.

        Inverse of :meth:`unrotate_to_spim_native`. Called when loading
        xyz_picks off disk so the rest of the GUI sees them in the rotated
        canonical frame used by the rotated atlas.
        """
        ras = self._rotation_ras_m()
        pts = np.asarray(points_ras_m, dtype=np.float64)
        if ras is None:
            return pts
        R_ras, c_ras = ras
        return (pts - c_ras) @ R_ras.T + c_ras

    def physical_points_to_indices(
        self, channel_ndxs: np.ndarray, round: bool = False, mode: str = "clip"
    ) -> np.ndarray:
        """
        Convert physical points in the atlas space to voxel indices in this atlas.

        Parameters
        ----------
        channel_ndxs : np.ndarray
            An (N, 3) array of physical points in the atlas space.

        Returns
        -------
        np.ndarray
            An (N, 3) array of voxel indices in this atlas.
        """
        return self.bc.xyz2i(channel_ndxs, round=round, mode=mode)[:, self.xyz2dims]

    def indices_to_physical_points(self, channel_ndxs: np.ndarray) -> np.ndarray:
        """
        Convert voxel indices in this atlas to physical points in the atlas space.

        Parameters
        ----------
        channel_ndxs : np.ndarray
            An (N, 3) array of voxel indices in this atlas.

        Returns
        -------
        np.ndarray
            An (N, 3) array of physical points in the atlas space.
        """
        return self.bc.i2xyz(channel_ndxs[:, self.dims2xyz])
