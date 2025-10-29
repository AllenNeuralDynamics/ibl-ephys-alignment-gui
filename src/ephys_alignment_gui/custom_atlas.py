import logging

import numpy as np
import SimpleITK as sitk
from iblatlas.atlas import BrainAtlas
from iblatlas.regions import BrainRegions

_logger = logging.getLogger(__name__)


class BrainAtlasAnatomical(BrainAtlas):
    pipeline_image: sitk.Image # Pipeline image associated with this atlas for CCF conversion
    def __init__(
        self,
        intensity_img: sitk.Image,
        label_img: sitk.Image,
        pipeline_img: sitk.Image,
    ) -> None:
        methods_to_check = ["GetOrigin", "GetSpacing", "GetSize", "GetDirection"]
        for m in methods_to_check:
            intensity_val = np.array(getattr(intensity_img, m)())
            label_val = np.array(getattr(label_img, m)())
            if not np.allclose(intensity_val, label_val):
                raise ValueError(
                    f"Intensity and label {m}() mismatch: {intensity_val} != {label_val}"
                )
        orientation_code = (
            sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(
                intensity_img.GetDirection()
            )
        )
        if orientation_code == "SRA":
            intensity_img_sra = intensity_img
            label_img_sra = label_img
        else:
            _logger.info(
                f"Reorienting volume from {orientation_code} to SRA for consistency with IBL convention."
            )
            intensity_img_sra = sitk.DICOMOrient(intensity_img, "SRA")
            label_img_sra = sitk.DICOMOrient(label_img, "SRA")
        # SRA to SAR, with C-order. This matches IBL XYZ (RAS with C order)
        # Allows you to have coronal slices be close in memory for visualization
        dims2xyz = np.array([1, 0, 2])
        xyz2dims = np.array([1, 0, 2])

        # Compute dxyz
        # Reverse to go to NP/C order, and convert from μm to m
        # This is in the IMAGE index coordinate system
        spacing_tup_np = np.array(intensity_img_sra.GetSpacing())[::-1] * 1e-3
        # But BrainCoordinates wants it in the XYZ world coordinate system

        spacing_tup_xyz = spacing_tup_np[dims2xyz]
        # IBL defines origin by saying which index is at world coordinate 0,0,0
        # We are not using bregma here, so just set to 0,0,0
        iorigin = [0, 0, 0]
        regions = BrainRegions()
        intensity_img_sra_arr = sitk.GetArrayFromImage(intensity_img_sra)
        label_img_sra_arr = sitk.GetArrayFromImage(label_img_sra)
        super().__init__(
            intensity_img_sra_arr,
            label_img_sra_arr,
            spacing_tup_xyz,
            regions,
            iorigin=iorigin,
            dims2xyz=dims2xyz,
            xyz2dims=xyz2dims,
        )
        self.pipeline_image = pipeline_img
