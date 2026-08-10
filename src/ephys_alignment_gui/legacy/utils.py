import numpy as np


def _fcn_extrap(x, f, bounds):
    """
    Extrapolates a flat value before and after bounds
    x: array to be filtered
    f: function to be applied between bounds (cf. fcn_cosine below)
    bounds: 2 elements list or np.array
    """
    y = f(x)
    y[x < bounds[0]] = f(bounds[0])
    y[x > bounds[1]] = f(bounds[1])
    return y


def fcn_cosine(bounds):
    """
    Returns a soft thresholding function with a cosine taper:
    values <= bounds[0]: values
    values < bounds[0] < bounds[1] : cosine taper
    values < bounds[1]: bounds[1]
    :param bounds:
    :return: lambda function
    """

    def _cos(x):
        return (1 - np.cos((x - bounds[0]) / (bounds[1] - bounds[0]) * np.pi)) / 2

    func = lambda x: _fcn_extrap(x, _cos, bounds)  # noqa
    return func


def bincount2D(x, y, xbin=0, ybin=0, xlim=None, ylim=None, weights=None):
    """
    Computes a 2D histogram by aggregating values in a 2D array.

    :param x: values to bin along the 2nd dimension (c-contiguous)
    :param y: values to bin along the 1st dimension
    :param xbin:
        scalar: bin size along 2nd dimension
        0: aggregate according to unique values
        array: aggregate according to exact values (count reduce operation)
    :param ybin:
        scalar: bin size along 1st dimension
        0: aggregate according to unique values
        array: aggregate according to exact values (count reduce operation)
    :param xlim: (optional) 2 values (array or list) that restrict range along 2nd dimension
    :param ylim: (optional) 2 values (array or list) that restrict range along 1st dimension
    :param weights: (optional) defaults to None, weights to apply to each value for aggregation
    :return: 3 numpy arrays MAP [ny,nx] image, xscale [nx], yscale [ny]
    """
    # if no bounds provided, use min/max of vectors
    if xlim is None:
        xlim = [np.min(x), np.max(x)]
    if ylim is None:
        ylim = [np.min(y), np.max(y)]

    def _get_scale_and_indices(v, bin, lim):
        # if bin is a nonzero scalar, this is a bin size: create scale and indices
        if np.isscalar(bin) and bin != 0:
            scale = np.arange(lim[0], lim[1] + bin / 2, bin)
            ind = (np.floor((v - lim[0]) / bin)).astype(np.int64)
        # if bin == 0, aggregate over unique values
        else:
            scale, ind = np.unique(v, return_inverse=True)
        return scale, ind

    xscale, xind = _get_scale_and_indices(x, xbin, xlim)
    yscale, yind = _get_scale_and_indices(y, ybin, ylim)
    # aggregate by using bincount on absolute indices for a 2d array
    nx, ny = [xscale.size, yscale.size]
    ind2d = np.ravel_multi_index(np.c_[yind, xind].transpose(), dims=(ny, nx))
    r = np.bincount(ind2d, minlength=nx * ny, weights=weights).reshape(ny, nx)

    # if a set of specific values is requested output an array matching the scale dimensions
    if not np.isscalar(xbin) and xbin.size > 1:
        _, iout, ir = np.intersect1d(xbin, xscale, return_indices=True)
        _r = r.copy()
        r = np.zeros((ny, xbin.size))
        r[:, iout] = _r[:, ir]
        xscale = xbin

    if not np.isscalar(ybin) and ybin.size > 1:
        _, iout, ir = np.intersect1d(ybin, yscale, return_indices=True)
        _r = r.copy()
        r = np.zeros((ybin.size, r.shape[1]))
        r[iout, :] = _r[ir, :]
        yscale = ybin

    return r, xscale, yscale


def probes_description(ses_path, one):
    """
    Aggregate probes information into ALF files
    Register alyx probe insertions and Micro-manipulator trajectories
    Input:
        raw_ephys_data/probeXX/
    Output:
        alf/probes.description.npy
    """

    eid = one.path2eid(ses_path, query_type="remote")
    ses_path = Path(ses_path)
    meta_files = spikeglx.glob_ephys_files(ses_path, ext="meta")
    ap_meta_files = [(ep.ap.parent, ep.label, ep) for ep in meta_files if ep.get("ap")]
    # If we don't detect any meta files exit function
    if len(ap_meta_files) == 0:
        return

    subdirs, labels, efiles_sorted = zip(*sorted(ap_meta_files))

    def _create_insertion(md, label, eid):
        # create json description
        description = {
            "label": label,
            "model": md.neuropixelVersion,
            "serial": int(md.serial),
            "raw_file_name": md.fileName,
        }

        # create or update probe insertion on alyx
        alyx_insertion = {
            "session": eid,
            "model": md.neuropixelVersion,
            "serial": md.serial,
            "name": label,
        }
        pi = one.alyx.rest("insertions", "list", session=eid, name=label)
        if len(pi) == 0:
            qc_dict = {"qc": "NOT_SET", "extended_qc": {}}
            alyx_insertion.update({"json": qc_dict})
            insertion = one.alyx.rest("insertions", "create", data=alyx_insertion)
        else:
            insertion = one.alyx.rest(
                "insertions",
                "partial_update",
                data=alyx_insertion,
                id=pi[0]["id"],
            )

        return description, insertion

    # Ouputs the probes description file
    probe_description = []
    alyx_insertions = []
    for label, ef in zip(labels, efiles_sorted):
        md = spikeglx.read_meta_data(ef.ap.with_suffix(".meta"))
        if md.neuropixelVersion == "NP2.4":
            # NP2.4 meta that hasn't been split
            if md.get("NP2.4_shank", None) is None:
                geometry = spikeglx.read_geometry(ef.ap.with_suffix(".meta"))
                nshanks = np.unique(geometry["shank"])
                for shank in nshanks:
                    label_ext = f"{label}{chr(97 + int(shank))}"
                    description, insertion = _create_insertion(md, label_ext, eid)
                    probe_description.append(description)
                    alyx_insertions.append(insertion)
            # NP2.4 meta that has already been split
            else:
                description, insertion = _create_insertion(md, label, eid)
                probe_description.append(description)
                alyx_insertions.append(insertion)
        else:
            description, insertion = _create_insertion(md, label, eid)
            probe_description.append(description)
            alyx_insertions.append(insertion)

    alf_path = ses_path.joinpath("alf")
    alf_path.mkdir(exist_ok=True, parents=True)
    probe_description_file = alf_path.joinpath("probes.description.json")
    with open(probe_description_file, "w+") as fid:
        fid.write(json.dumps(probe_description))

    return [probe_description_file]
