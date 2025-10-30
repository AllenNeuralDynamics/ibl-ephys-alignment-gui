import logging

import numpy as np
import SimpleITK as sitk
from iblatlas.atlas import BrainAtlas, BrainCoordinates
from iblatlas.regions import BrainRegions

_logger = logging.getLogger(__name__)


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
        methods_to_check = ["GetOrigin", "GetSpacing", "GetSize", "GetDirection"]
        for m in methods_to_check:
            intensity_val = np.array(getattr(intensity_img, m)())
            label_val = np.array(getattr(label_img, m)())
            if not np.allclose(intensity_val, label_val):
                raise ValueError(
                    f"Intensity and label {m}() mismatch: {intensity_val} != {label_val}"
                )

        # Reorient to SRA if needed
        orientation_code = (
            sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(
                intensity_img.GetDirection()
            )
        )
        if orientation_code == "SRA":
            intensity_img_sra = intensity_img
            label_img_sra = label_img
            pipeline_img_sra = pipeline_img
        else:
            _logger.info(
                f"Reorienting volume from {orientation_code} to SRA for consistency with IBL convention."
            )
            intensity_img_sra = sitk.DICOMOrient(intensity_img, "SRA")
            label_img_sra = sitk.DICOMOrient(label_img, "SRA")
            pipeline_img_sra = sitk.DICOMOrient(pipeline_img, "SRA")

        # SRA to SAR, with C-order. This matches IBL XYZ (RAS with C order)
        # Allows you to have coronal slices be close in memory for visualization
        dims2xyz = np.array([1, 0, 2])
        xyz2dims = np.array([1, 0, 2])

        # Compute dxyz (IBL spacing) from SimpleITK spacing
        # Reverse order to go from SimpleITK fortran order to NP/C order, and
        # convert from mm to m. SimpleITK spacing should be in mm for these
        # images, and is also the physical spacing of the image row-major ijk
        # axes
        spacing_tup_np = np.array(intensity_img_sra.GetSpacing())[::-1] * 1e-3

        # But BrainCoordinates wants it in the IBL XYZ (RAS) world coordinate system
        dxyz = spacing_tup_np[dims2xyz]

        # IBL defines origin by saying which index is at world coordinate 0,0,0
        # We are not using bregma here, so just set to 0,0,0
        iorigin = [0, 0, 0]

        # We can use BrainRegions from iblatlas because the labels are
        # lateralized IBL
        regions = BrainRegions()

        # Get arrays from SimpleITK images. I think there's no mutation risk
        intensity_img_sra_arr = sitk.GetArrayViewFromImage(intensity_img_sra)
        label_img_sra_arr = sitk.GetArrayViewFromImage(label_img_sra)

        # Initialize the superclass
        super().__init__(
            intensity_img_sra_arr,
            label_img_sra_arr,
            dxyz,
            regions,
            iorigin=iorigin,
            dims2xyz=dims2xyz,
            xyz2dims=xyz2dims,
        )

        # Need to account for the anatomical image origin not being at 0,0,0
        # SimpleITK is mm LPS, and IBL wants m RAS
        sitk_origin_ras_m = (
            np.array(intensity_img_sra.GetOrigin()) * np.array([-1, -1, 1]) * 1e-3
        )
        nxyz = np.array(intensity_img_sra_arr.shape)[dims2xyz]
        self.bc = BrainCoordinates(nxyz=nxyz, xyz0=sitk_origin_ras_m, dxyz=dxyz)
        # Store the SimpleITK intensity image, and the pipeline image for use
        # with CCF transforms
        self.intensity_sitk_image = intensity_img_sra
        self.pipeline_sitk_image = pipeline_img_sra
