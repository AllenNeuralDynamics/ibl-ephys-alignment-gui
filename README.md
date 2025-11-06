# IBL Ephys Alignment GUI

![GUI Screenshot](src/ephys_alignment_gui/ephys_atlas_image.png)

GUI developed by the International Brain Laboratory for aligning electrophysiology data with histology data.

Usage instructions can be found on the [`iblapps` wiki](https://github.com/int-brain-lab/iblapps/wiki)


## Allen Institute for Neural Dynamics fork

This version of the GUI includes the following modifications:
- Only include code for the alignment app (and rename to `ephys_alignment_gui`)
- Restrict dependencies to `iblatlas`, `PyQt5`, and `pyqtgraph`
- Use separate directories for loading and saving, to allow input data to live in a read-only filesystem


## Installing

This version can be installed from GitHub via `pip`:

```
pip install git+https://github.com/AllenNeuralDynamics/ibl-ephys-alignment-gui.git
```

If you're running an Ubuntu workstation on Code Ocean, just add that line to the post-install script.

Once the package has been installed in your environment, you can run the GUI with the following command:

```
launch
```

