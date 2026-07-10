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

You can prefill the save-root directory by setting `EPHYS_ALIGNMENT_OUTPUT_ROOT`.
For example, on Code Ocean set `EPHYS_ALIGNMENT_OUTPUT_ROOT=/results`. The GUI
still lets users choose or edit a different output root after startup.


## Keyboard shortcuts

Every action below is also available from the GUI's menu bar, where each item
shows its shortcut — the menu bar is the authoritative source. This list is
taken from `src/ephys_alignment_gui/ephys_gui_setup.py`.

### Alignment (Fit Options menu)

| Shortcut | Action |
|---|---|
| `Enter` | Fit — apply the interpolation from the reference lines (commits the move) |
| `O` | Offset |
| `Shift+Up` / `Shift+Down` | Offset by ±50 µm |
| `Shift+D` | Delete a reference line (hover the line first) |
| `←` | Previous move (undo) |
| `→` | Next move (redo) |
| `Ctrl+R` | Reset to the loaded/original state |
| `Ctrl+S` | Save |

### Display (Display Options menu)

| Shortcut | Action |
|---|---|
| `Alt+1` / `Alt+2` / `Alt+3` / `Alt+4` | Toggle Image / Line / Probe / Slice plot |
| `Alt+Ctrl+1…4` | Toggle to the *previous* Image / Line / Probe / Slice plot |
| `Shift+1` / `Shift+2` / `Shift+3` | Switch panel layout (View 1 / 2 / 3) |
| `Shift+A` | Reset axes |
| `Shift+L` | Hide/show region labels |
| `Shift+H` | Hide/show reference lines |
| `Shift+C` | Hide/show channels |
| `Shift+N` | Hide/show nearby boundaries |
| `Shift+M` | Change histology map (Allen ↔ Franklin–Paxinos) |
| `Alt+M` / `Alt+X` | Minimise-show / close cluster popup |
| `Ctrl+Shift+S` | Save plots |
| `Shift+I` | Region info |

### Undoing or deleting an alignment

There is no "delete one alignment" action — committed moves form a linear
undo/redo buffer (per shank):

- **Undo the fit you just applied:** `←` (Previous) steps back one committed
  move; `→` (Next) re-applies it.
- **Discard all alignment work for the current shank:** `Ctrl+R` (Reset).
- **Remove a reference line you placed but have not Fit yet:** hover it and
  press `Shift+D`.

An alignment only enters the buffer/history when you commit it with Fit,
Offset, or a move — placing reference lines alone does not.
