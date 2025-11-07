import logging

import numpy as np
import SimpleITK as sitk
from iblatlas.atlas import BrainAtlas, BrainCoordinates
from iblatlas.regions import BrainRegions
from iblutil.numerical import ismember

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
    """

    # Pipeline image associated with this atlas for CCF conversion
    pipeline_sitk_image: sitk.Image
    # Anatomical intensity image
    intensity_sitk_image: sitk.Image

    def __init__(
        self,
        intensity_img: sitk.Image,
        label_img: sitk.Image,
        pipeline_img: sitk.Image,
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
