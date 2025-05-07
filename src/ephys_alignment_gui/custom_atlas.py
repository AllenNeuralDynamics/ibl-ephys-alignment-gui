
from iblatlas.atlas import BrainAtlas,ALLEN_CCF_LANDMARKS_MLAPDV_UM
from iblatlas.atlas import _download_atlas_allen
from iblutil.numerical import ismember
from iblatlas.regions import BrainRegions, FranklinPaxinosRegions


from pathlib import Path, PurePosixPath
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import nrrd

import one.params
import logging

import SimpleITK as sitk

_logger = logging.getLogger(__name__)


class CustomAllenAtlas(BrainAtlas):
    """pathlib.PurePosixPath: The default relative path of the Allen atlas file."""
    atlas_rel_path = PurePosixPath('histology', 'ATLAS', 'Needles', 'Allen')
    image = None
    label = None


    def __init__(self, 
                 res_um=25, 
                 scaling=(1, 1, 1), 
                 mock=False, 
                 template_path=None,
                 label_path = None):
        """
        Instantiates an atlas.BrainAtlas corresponding to the Allen CCF at the given resolution
        using the IBL Bregma and coordinate system.

        This code is existed the IBL AllenAtlas to prevent re-downloading the atlas files

        Parameters
        ----------
        res_um : {10, 25, 50} int
            The Atlas resolution in micrometres; one of 10, 25 or 50um.
        scaling : float, numpy.array
            Scale factor along ml, ap, dv for squeeze and stretch (default: [1, 1, 1]).
        mock : bool
            For testing purposes, return atlas object with image comprising zeros.
        hist_path : str, pathlib.Path
            The location of the image volume. May be a full file path or a directory.

        Examples
        --------
        Instantiate Atlas from a non-default location, in this case the cache_dir of an ONE instance.
        >>> target_dir = one.cache_dir / AllenAtlas.atlas_rel_path
        ... ba = AllenAtlas(hist_path=target_dir)
        """
        LUT_VERSION = 'v01'  # version 01 is the lateralized version
        regions = BrainRegions()
        xyz2dims = np.array([1, 0, 2])  # this is the c-contiguous ordering
        dims2xyz = np.array([1, 0, 2])
        # we use Bregma as the origin
        self.res_um = res_um
        ibregma = (ALLEN_CCF_LANDMARKS_MLAPDV_UM['bregma'] / self.res_um)
        dxyz = self.res_um * 1e-6 * np.array([1, -1, -1]) * scaling
        if mock:
            image, label = [np.zeros((528, 456, 320), dtype=np.int16) for _ in range(2)]
            label[:, :, 100:105] = 1327  # lookup index for retina, id 304325711 (no id 1327)
        else:
            # Hist path may be a full path to an existing image file, or a path to a directory
            template_cache_dir = Path(one.params.get(silent=True).CACHE_DIR)
            template_path = Path(template_path or template_cache_dir.joinpath(self.atlas_rel_path))
            if not template_path.suffix:  # check if folder
                template_path /= f'average_template_{res_um}.nrrd'
            # get the image volume
            if not template_path.exists():
                template_path = _download_atlas_allen(template_path)

            # get the label volume
            label_path = Path(label_path)
            if not label_path.suffix:
                label_path /= f'average_template_{res_um}.nrrd'
            if not label_path.exists():
                label_path = _download_atlas_allen(label_path)

            file_label_remap = Path(label_path).parent/f'annotation_{res_um}_lut_{LUT_VERSION}.npz'
            if not file_label_remap.exists():
                label = self._read_volume(file_label_remap).astype(dtype=np.int32)
                _logger.info("Computing brain atlas annotations lookup table")
                # lateralize atlas: for this the regions of the left hemisphere have primary
                # keys opposite to to the normal ones
                lateral = np.zeros(label.shape[xyz2dims[0]])
                lateral[int(np.floor(ibregma[0]))] = 1
                lateral = np.sign(np.cumsum(lateral)[np.newaxis, :, np.newaxis] - 0.5)
                label = label * lateral.astype(np.int32)
                # the 10 um atlas is too big to fit in memory so work by chunks instead
                if res_um == 10:
                    first, ncols = (0, 10)
                    while True:
                        last = np.minimum(first + ncols, label.shape[-1])
                        _logger.info(f"Computing... {last} on {label.shape[-1]}")
                        _, im = ismember(label[:, :, first:last], regions.id)
                        label[:, :, first:last] = np.reshape(im, label[:, :, first:last].shape)
                        if last == label.shape[-1]:
                            break
                        first += ncols
                    label = label.astype(dtype=np.uint16)
                    _logger.info("Saving npz, this can take a long time")
                else:
                    _, im = ismember(label, regions.id)
                    label = np.reshape(im.astype(np.uint16), label.shape)
                np.savez_compressed(file_label_remap, label)
                _logger.info(f"Cached remapping file {file_label_remap} ...")
            # loads the files
            label = self._read_volume(file_label_remap)
            image = self._read_volume(template_path)
        super().__init__(image, label, dxyz, regions, ibregma, dims2xyz=dims2xyz, xyz2dims=xyz2dims)

    @staticmethod
    def _read_volume(file_volume):
        if file_volume.suffix == '.nrrd':
            volume, _ = nrrd.read(file_volume, index_order='C')  # ml, dv, ap
            # we want the coronal slice to be the most contiguous
            volume = np.transpose(volume, (2, 0, 1))  # image[iap, iml, idv]
        elif file_volume.suffix == '.npz':
            volume = np.load(file_volume)['arr_0']
        return volume

    def xyz2ccf(self, xyz, ccf_order='mlapdv', mode='raise'):
        """
        Converts anatomical coordinates to CCF coordinates.

        Anatomical coordinates are in meters, relative to bregma, which CFF coordinates are
        assumed to be the volume indices multiplied by the spacing in micormeters.

        Parameters
        ----------
        xyz : numpy.array
            An N by 3 array of anatomical coordinates in meters, relative to bregma.
        ccf_order : {'mlapdv', 'apdvml'}, default='mlapdv'
            The order of the CCF coordinates returned. For IBL (the default) this is (ML, AP, DV),
            for Allen MCC vertices, this is (AP, DV, ML).
        mode : {'raise', 'clip', 'wrap'}, default='raise'
            How to behave if the coordinate lies outside of the volume: raise (default) will raise
            a ValueError; 'clip' will replace the index with the closest index inside the volume;
            'wrap' will return the index as is.

        Returns
        -------
        numpy.array
            Coordinates in CCF space (um, origin is the front left top corner of the data
        volume, order determined by ccf_order
        """
        ordre = self._ccf_order(ccf_order)
        ccf = self.bc.xyz2i(xyz, round=False, mode=mode) * float(self.res_um)
        return ccf[..., ordre]


    def ccf2xyz(self, ccf, ccf_order='mlapdv'):
        """
        Convert anatomical coordinates from CCF coordinates.

        Anatomical coordinates are in meters, relative to bregma, which CFF coordinates are
        assumed to be the volume indices multiplied by the spacing in micormeters.

        Parameters
        ----------
        ccf : numpy.array
            An N by 3 array of coordinates in CCF space (atlas volume indices * um resolution). The
            origin is the front left top corner of the data volume.
        ccf_order : {'mlapdv', 'apdvml'}, default='mlapdv'
            The order of the CCF coordinates given. For IBL (the default) this is (ML, AP, DV),
            for Allen MCC vertices, this is (AP, DV, ML).

        Returns
        -------
        numpy.array
            The MLAPDV coordinates in meters, relative to bregma.
        """
        ordre = self._ccf_order(ccf_order, reverse=True)
        return self.bc.i2xyz((ccf[..., ordre] / float(self.res_um)))

    @staticmethod
    def _ccf_order(ccf_order, reverse=False):
        """
        Returns the mapping to go from CCF coordinates order to the brain atlas xyz
        :param ccf_order: 'mlapdv' or 'apdvml'
        :param reverse: defaults to False.
            If False, returns from CCF to brain atlas
            If True, returns from brain atlas to CCF
        :return:
        """
        if ccf_order == 'mlapdv':
            return [0, 1, 2]
        elif ccf_order == 'apdvml':
            if reverse:
                return [2, 0, 1]
            else:
                return [1, 2, 0]
        else:
            ValueError("ccf_order needs to be either 'mlapdv' or 'apdvml'")

    def compute_regions_volume(self, cumsum=False):
        """
        Sums the number of voxels in the labels volume for each region.
        Then compute volumes for all of the levels of hierarchy in cubic mm.
        :param: cumsum: computes the cumulative sum of the volume as per the hierarchy (defaults to False)
        :return:
        """
        nr = self.regions.id.shape[0]
        count = np.bincount(self.label.flatten(), minlength=nr)
        if not cumsum:
            self.regions.volume = count * (self.res_um / 1e3) ** 3
        else:
            self.regions.compute_hierarchy()
            self.regions.volume = np.zeros_like(count)
            for i in np.arange(nr):
                if count[i] == 0:
                    continue
                self.regions.volume[np.unique(self.regions.hierarchy[:, i])] += count[i]
            self.regions.volume = self.regions.volume * (self.res_um / 1e3) ** 3       

# This is a custom atlas class that inherits from BrainAtlas
class CustomAtlas(BrainAtlas):
    image = None
    label = None
    read_string = 'IRP' # Works, but not what it should be.

    def __init__(self,
                 atlas_image_file =None,
                 atlas_labels_file = None,
                 bregma = None,
                 force_um = None,
                 scaling = np.array([1,1,1])):
        self.atlas_image_file = atlas_image_file
        self.atlas_labels_file = atlas_labels_file
        if force_um is None:
            #dxyz = np.array(self.read_atlas_image())*np.array([1, -1, -1])*1e-6
            dxyz = np.array(self.read_atlas_image()) * 1000
            print('Atlas scaling', dxyz)
            self.res_um = dxyz[0] 
            dxyz = self.res_um * 1e-6 * np.array([1, 1, 1]) * scaling
            print('Resolution', self.res_um)
        else:
            _  = self.read_atlas_image()
            self.res_um = force_um
            dxyz = self.res_um * 1e-6 * np.array([1, -1, -1]) * scaling        
        self.read_atlas_labels()
        regions = BrainRegions()
        #_, im = ismember(self.label, regions.id)
        #label = np.reshape(im.astype(np.uint16), self.label.shape)
        self.label[~np.isin(self.label,regions.id)]=997
        self.label = self.label.astype(np.uint64)

        
        xyz2dims = np.array([0, 1, 2])  # this is the c-contiguous ordering
        dims2xyz = np.array([0, 1, 2])
        if bregma is None:
            bregma = [0,0,0]
        elif isinstance(bregma,str) and (bregma.lower() == 'allen'):
            bregma = (ALLEN_CCF_LANDMARKS_MLAPDV_UM['bregma'] / self.res_um)
        super().__init__(self.image, self.label, dxyz, regions, iorigin=list(self.offset), dims2xyz=dims2xyz, xyz2dims=xyz2dims)
        self.label[~np.isin(self.label,regions.id)]=997

    
    def read_atlas_image(self):
        # Reads the 
        IMG = sitk.ReadImage(self.atlas_image_file)
        # Convert sitk to the (ap, ml, dv) np array needed by BrainAtlas
        #IMG2 = sitk.DICOMOrient(IMG,self.read_string) 
        self.original_image = IMG
        self.image = np.flip(sitk.GetArrayFromImage(IMG).T, axis=(0, 2))
        print('Shape', self.image.shape)
        self.offset = IMG.GetOrigin()
        self.spacing = IMG.GetSpacing()[0] * 1000
        return IMG.GetSpacing()
        
    def read_atlas_labels(self):
        IMG = sitk.ReadImage(self.atlas_labels_file)
        # Convert sitk to the (ap, ml, dv) np array needed by BrainAtlas
        #IMG2 = sitk.DICOMOrient(IMG,self.read_string)
        self.label = np.flip(sitk.GetArrayFromImage(IMG).astype(np.int32).T, axis=(0, 2))