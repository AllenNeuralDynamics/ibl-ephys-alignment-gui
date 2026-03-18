import logging

import numpy as np
from iblatlas import atlas
from iblatlas.atlas import BrainAtlas, Trajectory
from numpy.typing import NDArray
from scipy.interpolate import interp1d

import ephys_alignment_gui.histology as histology

logger = logging.getLogger(__name__)

TIP_SIZE_UM = 200


def _cumulative_distance(xyz):
    return np.cumsum(np.r_[0, np.sqrt(np.sum(np.diff(xyz, axis=0) ** 2, axis=1))])


def _get_surface_intersection_override(
    traj: Trajectory,
    brain_atlas: BrainAtlas,
    surface: str = "top",
    mode: str = "raise",
) -> NDArray[np.float64]:
    """
    Override for atlas.Insertion._get_surface_intersection to avoid issues
    with atlases that don't have `res_um`

    Computes the intersection of a trajectory with either the top or the bottom surface of an atlas.

    Parameters
    ----------
    traj: iblatlas.atlas.Trajectory object
    brain_atlas: iblatlas.atlas.BrainAtlas (or descendant) object
    surface: str, optional (defaults to 'top') 'top' or 'bottom'
    mode: str, optional (defaults to 'raise') 'raise' or 'none': raise an error if no intersection
        with the brain surface is found otherwise returns None

    Returns
    -------
    xyz: np.array, 3 elements, x, y, z coordinates of the intersection point with the surface
            None if no intersection is found and mode is not set to 'raise'
    """
    brain_atlas.compute_surface()
    distance = traj.mindist(brain_atlas.srf_xyz)
    dist_sort = np.argsort(distance)
    # In some cases the nearest two intersection points are not the top and bottom of brain
    # So we find all intersection points that fall within one voxel and take the one with
    # highest dV to be entry and lowest dV to be exit
    max_voxel_size_m = np.max(np.abs(brain_atlas.bc.dxyz))
    idx_lim = np.sum(distance[dist_sort] < max_voxel_size_m)
    if idx_lim == 0:  # no intersection found
        if mode == "raise":
            raise ValueError("No intersection found with brain surface")
        else:
            return
    dist_lim = dist_sort[0:idx_lim]
    z_val = brain_atlas.srf_xyz[dist_lim, 2]
    if surface == "top":
        ma = np.argmax(z_val)
        _xyz = brain_atlas.srf_xyz[dist_lim[ma], :]
        _ixyz = brain_atlas.bc.xyz2i(_xyz)
        _ixyz[brain_atlas.xyz2dims[2]] += 1
    elif surface == "bottom":
        ma = np.argmin(z_val)
        _xyz = brain_atlas.srf_xyz[dist_lim[ma], :]
        _ixyz = brain_atlas.bc.xyz2i(_xyz)

    xyz = brain_atlas.bc.i2xyz(_ixyz.astype(float))

    return xyz


def get_brain_exit_override(
    traj: Trajectory, brain_atlas: BrainAtlas, mode: str = "raise"
) -> NDArray[np.float64]:
    """
    Given a Trajectory and a BrainAtlas object, computes the brain exit coordinate as the
    intersection of the trajectory and the brain surface (brain_atlas.surface)
    :param brain_atlas:
    :return: 3 element array x,y,z
    """
    # Find point where trajectory intersects with bottom of brain
    return _get_surface_intersection_override(
        traj, brain_atlas, surface="bottom", mode=mode
    )


def get_brain_entry_override(
    traj: Trajectory, brain_atlas: BrainAtlas, mode: str = "raise"
) -> NDArray[np.float64]:
    """
    Given a Trajectory and a BrainAtlas object, computes the brain entry coordinate as the
    intersection of the trajectory and the brain surface (brain_atlas.surface)
    :param brain_atlas:
    :return: 3 element array x,y,z
    """
    # Find point where trajectory intersects with top of brain
    return _get_surface_intersection_override(
        traj, brain_atlas, surface="top", mode=mode
    )


class EphysAlignment:
    def __init__(
        self,
        track_annotations_ras: NDArray[np.floating],
        chn_depths=None,
        track_prev=None,
        feature_prev=None,
        brain_atlas=None,
        speedy=False,
        track_margin_m=6e-3,
    ) -> None:
        if not brain_atlas:
            self.brain_atlas = atlas.AllenAtlas(25)
        else:
            self.brain_atlas = brain_atlas

        self.track_annos_and_ends_ras, self.track_extent, self.depths_along_trk = (
            self.get_insertion_track(track_annotations_ras, speedy=speedy)
        )

        # Initial depth estimate.
        # If not provided, end of track will be used.
        self.chn_depths = chn_depths
        if np.any(track_prev):
            self.track_init = track_prev
            self.feature_init = feature_prev
        else:
            # Determine required range based on probe geometry
            tip_track_m = -track_margin_m
            if chn_depths is not None and len(chn_depths) > 0:
                probe_span = 1e-6 * np.max(chn_depths)  # meters
                # Add 50% margin for alignment flexibility
                margin_factor = 1.5
                top_track_m = max(track_margin_m, probe_span * margin_factor)
            else:
                # Default to 6mm if no channel information available
                top_track_m = track_margin_m

            self.track_init = np.array([tip_track_m, top_track_m])
            self.feature_init = np.copy(self.track_init)

        # Fit trajectory to the track for voxel-aligned sampling
        # Get DV range of the trajectory
        z_min = np.min(self.track_annos_and_ends_ras[:, 2])
        z_max = np.max(self.track_annos_and_ends_ras[:, 2])
        # Convert to voxel indices and align to voxel boundaries
        i_min, i_max = np.sort(
            self.brain_atlas.bc.z2i(np.array([z_min, z_max]), mode="clip")
        )
        i_min = int(i_min)
        i_max = int(i_max)
        # Sample at every voxel in DV (z) direction
        z_samples = np.sort(self.brain_atlas.bc.i2z(np.arange(i_min, i_max + 1)))

        # Evaluate trajectory at voxel-aligned DV coordinates
        track_cumulative_distance = _cumulative_distance(self.track_annos_and_ends_ras)
        depths_at_z_samples = np.interp(
            z_samples, self.track_annos_and_ends_ras[:, 2], track_cumulative_distance
        )
        self.track_interpolation_ras = histology.interpolate_along_track(
            self.track_annos_and_ends_ras, depths_at_z_samples
        )
        # Compute cumulative distance along trajectory for compatibility
        # (ephys_depths_along_track is used for depth_coords in get_histology_regions)
        first_electrode_dist = track_cumulative_distance[1] + 1e-6 * TIP_SIZE_UM
        self.ephys_depths_along_track = depths_at_z_samples - first_electrode_dist
        # ensure none of the track is outside the y or x lim of atlas
        xlim = np.sort(self.brain_atlas.bc.xlim)
        ylim = np.sort(self.brain_atlas.bc.ylim)
        x_in_range = np.bitwise_and(
            self.track_interpolation_ras[:, 0] >= xlim[0],
            self.track_interpolation_ras[:, 0] <= xlim[1],
        )
        y_in_range = np.bitwise_and(
            self.track_interpolation_ras[:, 1] >= ylim[0],
            self.track_interpolation_ras[:, 1] <= ylim[1],
        )
        rem = np.bitwise_and(x_in_range, y_in_range)
        self.track_interpolation_ras = self.track_interpolation_ras[rem]
        # Also filter ephys_depths_along_track to match filtered track_interpolation_ras
        self.ephys_depths_along_track = self.ephys_depths_along_track[rem]

        self.region, self.region_label, self.region_colour, self.region_id = (
            self.get_histology_regions(
                self.track_interpolation_ras,
                self.ephys_depths_along_track,
                self.brain_atlas,
            )
        )

    def get_insertion_track(self, track_annotations_ras, speedy=False):
        """
        Extends probe trajectory from bottom of brain to upper bound of allen atlas
        :param track_annotations_ras: points defining probe trajectory in 3D space (Right Anterior Superior)
        :type track_annotations_ras: np.array((n, 3)) - n: no. of unique points
        :return track_annos_and_ends_ras: points defining extended trajectory in 3D space (RAS)
        :type track_annos_and_ends_ras: np.array((n+2, 3))
        :return track_extent: cumulative distance between two extremes of track_annos_and_ends_ras (bottom of
        brain and top of atlas) offset by distance to probe tip
        :type track_extent: np.array((2))
        """
        # Use the first and last quarter of track_annotations_ras to estimate
        # the trajectory beyond track_annotations_ras
        n_picks = np.max([4, round(track_annotations_ras.shape[0] / 4)])
        traj_entry = atlas.Trajectory.fit(track_annotations_ras[:n_picks, :])
        traj_exit = atlas.Trajectory.fit(track_annotations_ras[-1 * n_picks :, :])

        # Force the entry to be on the upper z lim of the atlas to account for
        # cases where channels may be located above the surface of the brain
        entry_lims = traj_entry.eval_z(self.brain_atlas.bc.zlim)
        entry_top_lim = np.argmax(entry_lims[:, 2])
        entry = entry_lims[entry_top_lim, :]
        if speedy:
            exit_lims = traj_exit.eval_z(self.brain_atlas.bc.zlim)
            exit_top_lim = np.argmin(entry_lims[:, 2])
            exit = exit_lims[exit_top_lim, :]
        else:
            exit = get_brain_exit_override(traj_exit, self.brain_atlas)
            # The exit is just below the bottom surfacce of the brain
            exit[2] = exit[2] - 200 / 1e6

        # Catch cases where the exit
        if any(np.isnan(exit)):
            exit = (traj_exit.eval_z(self.brain_atlas.bc.zlim))[1, :]

        track_annos_and_ends_ras = np.r_[
            exit[np.newaxis, :], track_annotations_ras, entry[np.newaxis, :]
        ]
        # Sort so that most ventral coordinate is first
        indices = np.argsort(track_annos_and_ends_ras[:, 2])
        track_annos_and_ends_ras = track_annos_and_ends_ras[indices, :]

        # Compute distance to first electrode from bottom coordinate
        cumulative_dist = _cumulative_distance(track_annos_and_ends_ras)
        tip_cumulative_distance = cumulative_dist[1]
        first_electrode_dist = tip_cumulative_distance + 1e-6 * TIP_SIZE_UM
        track_length = cumulative_dist[-1]
        track_extent = np.array([0, track_length]) - first_electrode_dist
        depths_along_track = cumulative_dist - first_electrode_dist
        logger.debug(f"Track extent: {track_extent}")
        return track_annos_and_ends_ras, track_extent, depths_along_track

    def get_track_and_feature(self):
        """
        Return track, feature and track_annos_and_ends_ras variables
        """
        return self.feature_init, self.track_init, self.track_annos_and_ends_ras

    @staticmethod
    def feature2track(feature_new, feature_ref, track_ref):
        """
        Estimate new values of feature_new according to interpolated fit between feature and track space
        :param feature_new: points in FEATURE space to convert TO track space
        :type feature_new: np.array
        :param feature_ref: reference coordinates in feature space (ephys plots)
        :type feature_ref: np.array((n_lines + 2)) n_lines: no. of user reference lines
        :param track_ref: reference coordinates in track space (histology track)
        :type track_ref: np.array((n_lines + 2))
        :return track_new: interpolated values of trk IN TRACK SPACE
        :type track_new: np.array
        """
        fcn = interp1d(feature_ref, track_ref, fill_value="extrapolate")
        return fcn(feature_new)

    @staticmethod
    def track2feature(track_new, feature_ref, track_ref):
        """
        Estimate new values of track_new according to interpolated fit between track and feature space
        :param track_new: points in track space to convert feature space
        :type track_new: np.array
        :param feature_ref: reference coordinates in feature space (ephys plots)
        :type feature_ref: np.array((n_lines + 2)) n_lines: no. of user reference lines
        :param track: reference coordinates in track space (histology track)
        :type track: np.array((n_lines + 2))
        :return fcn(track_new): interpolated values of track_new
        :type fcn(track_new): np.array
        """
        fcn = interp1d(track_ref, feature_ref, fill_value="extrapolate")
        return fcn(track_new)

    @staticmethod
    def feature2track_lin(trk, feature, track):
        """
        Estimate new values of trk according to linear fit between feature and track space, only
        implemented if no. of reference points >= 3
        :param trk: points in track space to convert feature space
        :type trk: np.array
        :param feature: reference coordinates in feature space (ephys plots)
        :type feature: np.array((n_lines + 2)) n_lines: no. of user reference lines
        :param track: reference coordinates in track space (histology track)
        :type track: np.array((n_lines + 2))
        :return fcn(trk): linear fit values of trk
        :type fcn(trk): np.array
        """
        if feature.size >= 5:
            fcn_lin = np.poly1d(np.polyfit(feature[1:-1], track[1:-1], 1))
            lin_fit = fcn_lin(trk)
        else:
            lin_fit = 0
        return lin_fit

    @staticmethod
    def adjust_extremes_uniform(feature, track):
        """
        Change the value of the first and last reference points (non user chosen points) such
        that coordinates outside user picked regions are left unchanged
        :param feature: reference coordinates in feature space (ephys plots)
        :type feature: np.array((n_lines + 2)) n_lines: no. of user reference lines
        :param track: reference coordinates in track space (histology track)
        :type track: np.array((n_lines + 2))
        :return track: reference coordinates in track space with first and last value adjusted
        :type track: np.array((n_lines + 2))
        """
        diff = np.diff(feature - track)
        track[0] -= diff[0]
        track[-1] += diff[-1]
        return track

    def adjust_extremes_linear(self, feature, track, extend_feature=1):
        """
        Change the value of the first and last reference points (non user chosen points) such
        that coordinates outside user picked regions have a linear fit applied
        :param feature: reference coordinates in feature space (ephys plots)
        :type feature: np.array((n_lines + 2)) n_lines: no. of user reference lines
        :param track: reference coordinates in track space (histology track)
        :type track: np.array((n_lines + 2))
        :param extend_feature: amount to extend extreme coordinates before applying linear fit
        :type extend_feature: float
        :return feature: reference coordinates in feature space with first and last value adjusted
        :type feature: np.array((n_lines + 2))
        :return track: reference coordinates in track space with first and last value adjusted
        :type track: np.array((n_lines + 2))
        """

        feature[0] = self.track_init[0] - extend_feature
        feature[-1] = self.track_init[-1] + extend_feature
        extend_track = self.feature2track_lin(feature[[0, -1]], feature, track)
        track[0] = extend_track[0]
        track[-1] = extend_track[-1]
        return feature, track

    def scale_histology_regions(self, feature, track, region=None, region_label=None):
        """
        Recompute locations of brain region boundaries using interpolated fit based on reference
        lines
        :param feature: reference coordinates in feature space (ephys plots)
        :type feature: np.array((n_lines + 2)) n_lines: no. of user reference lines
        :param track: reference coordinates in track space (histology track)
        :type track: np.array((n_lines + 2))
        :return region: new coordinates of histology boundaries after applying interpolation
        :type region: np.array((n_bound, 2)) n_bound: no. of histology boundaries
        :return region_label: new coordinates of histology labels positions after applying
                              interpolation
        :type region_label: np.array((n_bound)) of tuples (coordinate - float, label - str)
        """
        region = np.copy(region) if region is not None else np.copy(self.region)
        region_label = (
            np.copy(region_label)
            if region_label is not None
            else np.copy(self.region_label)
        )
        region = self.track2feature(region, feature, track) * 1e6
        region_label[:, 0] = (
            self.track2feature(np.float64(region_label[:, 0]), feature, track) * 1e6
        )

        return region, region_label

    @staticmethod
    def get_histology_regions(xyz_coords, depth_coords, brain_atlas, mapping=None):
        """
        Find all brain regions and their boundaries along the depth of probe or track
        :param xyz_coords: 3D coordinates of points along probe or track
        :type xyz_coords: np.array((n_points, 3)) n_points: no. of points
        :param depth_coords: depth along probe or track where each xyz_coord is located
        :type depth_coords: np.array((n_points))
        :return region: coordinates bounding each brain region
        :type region: np.array((n_bound, 2)) n_bound: no. of histology boundaries
        :return region_label: label for each brain region and coordinate of where to place label
        :type region_label: np.array((n_bound)) of tuples (coordinate - float, label - str)
        :return region_colour: allen atlas rgb colour for each brain region along track
        :type region_colour: np.array((n_bound, 3))
        :return region_id: allen atlas id for each brain region along track
        :type region_id: np.array((n_bound))
        """
        # Input validation and sanity checks
        if len(xyz_coords) == 0:
            raise ValueError("Empty coordinate array provided")

        if xyz_coords.ndim != 2 or xyz_coords.shape[1] != 3:
            raise ValueError(
                f"Coordinates must be Nx3 array, got shape {xyz_coords.shape}"
            )

        if len(depth_coords) != len(xyz_coords):
            raise ValueError(
                f"Coordinate and depth arrays must have same length: {len(xyz_coords)} vs {len(depth_coords)}"
            )

        if np.any(~np.isfinite(xyz_coords)) or np.any(~np.isfinite(depth_coords)):
            raise ValueError("Coordinates contain NaN or infinite values")

        if len(xyz_coords) < 2:
            raise ValueError(
                f"Need at least 2 trajectory points, got {len(xyz_coords)}"
            )

        # Check for degenerate trajectory (all points at same location)
        coord_range = np.ptp(xyz_coords, axis=0)  # peak-to-peak range
        if np.all(coord_range < 1e-9):  # less than 1 nanometer variation
            raise ValueError(
                "Degenerate trajectory: all coordinates are at the same location"
            )

        # Check for unreasonably large coordinates (likely unit errors)
        max_coord = np.abs(xyz_coords).max()
        if max_coord > 1.0:  # larger than 1 meter
            logger.warning(
                f"Very large coordinates detected (max: {max_coord:.3f}m). Check coordinate units."
            )

        region_ids = brain_atlas.get_labels(xyz_coords, mapping=mapping)

        region_info = brain_atlas.regions.get(region_ids)
        boundaries = np.where(np.diff(region_info.id))[0]

        # Handle single region case (no boundaries)
        # if len(boundaries) == 0:
        #     # Single region spanning entire trajectory
        #     region = np.array([[depth_coords[0], depth_coords[-1]]])
        #     region_label = np.array([[(depth_coords[0] + depth_coords[-1]) / 2, region_info.acronym[0]]], dtype=object)
        #     region_id = np.array([[region_info.id[0]]], dtype=int)
        #     region_colour = np.array([region_info.rgb[0]], dtype=int)

        #     print(f"INFO: Probe trajectory spans single brain region: {region_info.acronym[0]}")
        #     return region, region_label, region_colour, region_id

        # Multiple regions case (original logic)
        region = np.empty((boundaries.size + 1, 2))
        region_label = np.empty((boundaries.size + 1, 2), dtype=object)
        region_id = np.empty((boundaries.size + 1, 1), dtype=int)
        region_colour = np.empty((boundaries.size + 1, 3), dtype=int)

        for bound in np.arange(boundaries.size + 1):
            if bound == 0:
                _region = np.array([0, boundaries[bound]])
            elif bound == boundaries.size:
                _region = np.array([boundaries[bound - 1], region_info.id.size - 1])
            else:
                _region = np.array([boundaries[bound - 1], boundaries[bound]])

            _region_colour = region_info.rgb[_region[1]]
            _region_label = region_info.acronym[_region[1]]
            _region_id = region_info.id[_region[1]]
            _region = depth_coords[_region]
            _region_mean = np.mean(_region)
            region[bound, :] = _region
            region_colour[bound, :] = _region_colour
            region_id[bound, :] = _region_id
            region_label[bound, :] = (_region_mean, _region_label)

        return region, region_label, region_colour, region_id

    @staticmethod
    def get_nearest_boundary(
        xyz_coords, allen, extent=100, steps=8, parent=True, brain_atlas=None
    ):
        """
        Finds distance to closest neighbouring brain region along trajectory. For each point in
        xyz_coords computes the plane passing through point and perpendicular to trajectory and
        finds all brain regions that lie in that plane up to a given distance extent from specified
        point. Additionally, if requested, computes distance between the parents of regions.
        :param xyz_coords: 3D coordinates of points along probe or track
        :type xyz_coords: np.array((n_points, 3)) n_points: no. of points
        :param allen: dataframe containing allen info. Loaded from allen_structure_tree in
        ibllib/atlas
        :type allen: pandas Dataframe
        :param extent: extent of plane in each direction from origin in (um)
        :type extent: float
        :param steps: no. of steps to discretise plane into
        :type steps: int
        :param parent: Whether to also compute nearest distance between parents of regions
        :type parent: bool
        :return nearest_bound: dict containing results
        :type nearest_bound: dict
        """
        if not brain_atlas:
            brain_atlas = atlas.AllenAtlas(25)

        vector = atlas.Insertion.from_track(
            xyz_coords, brain_atlas=brain_atlas
        ).trajectory.vector
        nearest_bound = dict()
        nearest_bound["dist"] = np.zeros(xyz_coords.shape[0])
        nearest_bound["id"] = np.zeros(xyz_coords.shape[0])
        # nearest_bound['adj_id'] = np.zeros((xyz_coords.shape[0]))
        nearest_bound["col"] = []

        if parent:
            nearest_bound["parent_dist"] = np.zeros(xyz_coords.shape[0])
            nearest_bound["parent_id"] = np.zeros(xyz_coords.shape[0])
            # nearest_bound['parent_adj_id'] = np.zeros((xyz_coords.shape[0]))
            nearest_bound["parent_col"] = []

        for iP, point in enumerate(xyz_coords):
            d = np.dot(vector, point)
            x_vals = np.r_[
                np.linspace(point[0] - extent / 1e6, point[0] + extent / 1e6, steps),
                point[0],
            ]
            y_vals = np.r_[
                np.linspace(point[1] - extent / 1e6, point[1] + extent / 1e6, steps),
                point[1],
            ]

            X, Y = np.meshgrid(x_vals, y_vals)
            Z = (d - vector[0] * X - vector[1] * Y) / vector[2]
            XYZ = np.c_[
                np.reshape(X, X.size),
                np.reshape(Y, Y.size),
                np.reshape(Z, Z.size),
            ]
            dist = np.sqrt(np.sum((XYZ - point) ** 2, axis=1))

            try:
                brain_id = brain_atlas.regions.get(brain_atlas.get_labels(XYZ))["id"]
            except Exception as err:
                logger.error(f"Failed to get brain region for boundary: {err}")
                continue

            dist_sorted = np.argsort(dist)
            brain_id_sorted = brain_id[dist_sorted]
            nearest_bound["id"][iP] = brain_id_sorted[0]
            nearest_bound["col"].append(
                allen["color_hex_triplet"][
                    np.where(allen["id"] == brain_id_sorted[0])[0][0]
                ]
            )
            bound_idx = np.where(brain_id_sorted != brain_id_sorted[0])[0]
            if np.any(bound_idx):
                nearest_bound["dist"][iP] = dist[dist_sorted[bound_idx[0]]] * 1e6
                # nearest_bound['adj_id'][iP] = brain_id_sorted[bound_idx[0]]
            else:
                nearest_bound["dist"][iP] = np.max(dist) * 1e6
                # nearest_bound['adj_id'][iP] = brain_id_sorted[0]

            if parent:
                # Now compute for the parents
                brain_parent = np.array(
                    [
                        allen["parent_structure_id"][np.where(allen["id"] == br)[0][0]]
                        for br in brain_id_sorted
                    ]
                )
                brain_parent[np.isnan(brain_parent)] = 0

                nearest_bound["parent_id"][iP] = brain_parent[0]
                nearest_bound["parent_col"].append(
                    allen["color_hex_triplet"][
                        np.where(allen["id"] == brain_parent[0])[0][0]
                    ]
                )

                parent_idx = np.where(brain_parent != brain_parent[0])[0]
                if np.any(parent_idx):
                    nearest_bound["parent_dist"][iP] = (
                        dist[dist_sorted[parent_idx[0]]] * 1e6
                    )
                    # nearest_bound['parent_adj_id'][iP] = brain_parent[parent_idx[0]]
                else:
                    nearest_bound["parent_dist"][iP] = np.max(dist) * 1e6
                    # nearest_bound['parent_adj_id'][iP] = brain_parent[0]

        return nearest_bound

    @staticmethod
    def arrange_into_regions(depth_coords, region_ids, distance, region_colours):
        """
        Arrange output from get_nearest_boundary into a form that can be plot using pyqtgraph or
        matplotlib
        :param depth_coords: depth along probe or track where each point is located
        :type depth_coords: np.array((n_points))
        :param region_ids: brain region id at each depth along probe
        :type regions_ids: np.array((n_points))
        :param distance: distance to nearest boundary in plane at each point
        :type distance: np.array((n_points))
        :param region_colours: allen atlas hex colour for each region id
        :type region_colours: list of strings len(n_points)
        :return all_x: dist values for each region along probe track
        :type all_x: list of np.array
        :return all_y: depth values for each region along probe track
        :type all_y: list of np.array
        :return all_colour: colour assigned to each region along probe track
        :type all_colour: list of str
        """

        boundaries = np.where(np.diff(region_ids))[0]
        bound = np.r_[0, boundaries + 1, region_ids.shape[0]]
        all_y = []
        all_x = []
        all_colour = []
        for iB in np.arange(len(bound) - 1):
            y = depth_coords[bound[iB] : (bound[iB + 1])]
            y = np.r_[y[0], y, y[-1]]
            x = distance[bound[iB] : (bound[iB + 1])]
            x = np.r_[0, x, 0]
            all_y.append(y)
            all_x.append(x)
            col = region_colours[bound[iB]]
            if not isinstance(col, str):
                col = "#FFFFFF"
            else:
                col = "#" + col
            all_colour.append(col)

        return all_x, all_y, all_colour

    def get_scale_factor(self, region, region_orig=None):
        """
        Find how much each brain region has been scaled following interpolation
        :param region: scaled histology boundaries
        :type region: np.array((n_bound, 2)) n_bound: no. of histology boundaries
        :return scaled_region: regions that have unique scaling applied
        :type scaled_region: np.array((n_scale, 2)) n_scale: no. of uniquely scaled regions
        :return scale_factor: scale factor applied to each scaled region
        :type scale_factor: np.array((n_scale))
        """

        region_orig = region_orig if region_orig is not None else self.region
        scale = []
        for iR, (reg, reg_orig) in enumerate(zip(region, region_orig * 1e6)):
            scale = np.r_[scale, (reg[1] - reg[0]) / (reg_orig[1] - reg_orig[0])]
        boundaries = np.where(np.diff(np.around(scale, 3)))[0]
        if boundaries.size == 0:
            scaled_region = np.array([[region[0][0], region[-1][1]]])
            scale_factor = np.unique(scale)
        else:
            scaled_region = np.empty((boundaries.size + 1, 2))
            scale_factor = []
            for bound in np.arange(boundaries.size + 1):
                if bound == 0:
                    _scaled_region = np.array(
                        [region[0][0], region[boundaries[bound]][1]]
                    )
                    _scale_factor = scale[0]
                elif bound == boundaries.size:
                    _scaled_region = np.array(
                        [region[boundaries[bound - 1]][1], region[-1][1]]
                    )
                    _scale_factor = scale[-1]
                else:
                    _scaled_region = np.array(
                        [
                            region[boundaries[bound - 1]][1],
                            region[boundaries[bound]][1],
                        ]
                    )
                    _scale_factor = scale[boundaries[bound]]
                scaled_region[bound, :] = _scaled_region
                scale_factor = np.r_[scale_factor, _scale_factor]
        return scaled_region, scale_factor

    def get_channel_locations(self, feature, track, depths=None):
        """
        Gets 3d coordinates from a depth along the electrophysiology feature. 2 steps
        1) interpolate from the electrophys features depths space to the probe depth space
        2) interpolate from the probe depth space to the true 3D coordinates
        if depths is not provided, defaults to channels local coordinates depths

        feature : reference coordinates in feature space (ephys plots)
        track : reference coordinates in track space (histology track)
        """
        if depths is None:
            depths = self.chn_depths / 1e6
        channel_depths_track = (
            self.feature2track(depths, feature, track) - self.track_extent[0]
        )

        channel_locations_ras = histology.interpolate_along_track(
            self.track_annos_and_ends_ras, channel_depths_track
        )
        return channel_locations_ras

    def get_tip_location(self, feature, track):
        """
        Gets 3D coordinates of the probe tip.
        The tip is TIP_SIZE_UM (200 μm) below the first electrode.
        Uses the same feature-track transformation as channels, so the tip
        position updates dynamically as the user adjusts alignment.

        feature : reference coordinates in feature space (ephys plots)
        track : reference coordinates in track space (histology track)

        Returns
        -------
        tip_location_ras : np.array, shape (3,)
            3D coordinates of the tip in RAS space (x, y, z)
        """
        # Tip is at negative depth (200 μm below first electrode)
        tip_depth = np.array([-TIP_SIZE_UM / 1e6])

        # Transform from feature space to track space using same mapping as channels
        tip_depth_track = (
            self.feature2track(tip_depth, feature, track) - self.track_extent[0]
        )

        # Interpolate 3D position along the trajectory
        tip_location_ras = histology.interpolate_along_track(
            self.track_annos_and_ends_ras, tip_depth_track
        )

        return tip_location_ras[0]  # Return single 3D point

    def get_brain_locations(self, channel_locations_ras):
        """
        Finds the brain regions from 3D coordinates of electrode locations
        :param channel_locations_ras: 3D coordinates of electrodes on probe
        :type channel_locations_ras: np.array((n_elec, 3)) n_elec: no. of electrodes (384)
        :return brain_regions: brain region object for each electrode
        :type dict
        """
        brain_regions = self.brain_atlas.regions.get(
            self.brain_atlas.get_labels(channel_locations_ras)
        )
        return brain_regions

    def get_perp_vector(self, feature, track):
        """
        Finds the perpendicular vector along the trajectory at the depth of reference lines
        :param feature: reference coordinates in feature space (ephys plots)
        :type feature: np.array((n_lines + 2)) n_lines: no. of user reference lines
        :param track: reference coordinates in track space (histology track)
        :type track: np.array((n_line+2))
        :return slice_lines: coordinates of perpendicular lines
        :type slice_lines: np.array((n_lines, 2))
        """

        slice_lines = []
        for line in feature[1:-1]:
            depths = np.array([line, line + 10 / 1e6])
            xyz = self.get_channel_locations(feature, track, depths)

            extent = 500e-6
            vector = np.diff(xyz, axis=0)[0]
            point = xyz[0, :]
            vector_perp = np.array([1, 0, -1 * vector[0] / vector[2]])
            xyz_per = np.r_[
                [point + (-1 * extent * vector_perp)],
                [point + (extent * vector_perp)],
            ]
            slice_lines.append(xyz_per)

        return slice_lines
