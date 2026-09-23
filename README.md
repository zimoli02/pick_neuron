# PickNeuron: Sequential Suite2p ROI Validation

`pick_neuron.py` is a lightweight Tkinter GUI for reviewing Suite2p ROIs one at a time. For each ROI, it displays a local mean or maximum projection, an optional ROI-mask overlay, and the corresponding `F` and `Fneu` traces.

Manual decisions are stored in a separate `iscell_new.npy` file, so the original Suite2p `iscell.npy` is not changed unless the user explicitly chooses to merge the final labels.

The program does not import Suite2p itself. It only requires Python, NumPy, Matplotlib, and Tkinter.

## 1. Download the script

Download `pick_neuron.py` and place it in a dedicated folder, for example:

```text
~/Desktop/PickNeuron/pick_neuron.py
```

If the script is hosted on GitHub, open the file and use **Download raw file**, or download the repository as a ZIP file. If it is shared through a file link, use the normal Download button.

Make sure the filename remains:

```text
pick_neuron.py
```

Some browsers or text editors may accidentally save it as `pick_neuron.py.txt`. Rename it if necessary.

## 2. Create a Python environment

### Option A: Create a dedicated environment

This is the recommended option because it keeps the GUI dependencies separate from the Suite2p installation.

Open Terminal and run:

```bash
conda create -n pick_neuron_env --override-channels -c conda-forge \
  python=3.11 numpy matplotlib tk
```

Activate the environment:

```bash
conda activate pick_neuron_env
```

Verify that all required packages can be imported:

```bash
python -c "import numpy, matplotlib, tkinter; print('Environment OK')"
```

If the command prints `Environment OK`, the environment is ready.

The script is known to work with Python 3.11, NumPy 1.26, Matplotlib 3.8, and Tk 8.6. Leaving the NumPy and Matplotlib patch versions unpinned usually makes dependency solving easier because Conda can select compatible builds automatically.

### Option B: Use an existing Suite2p environment

The script does not require the `suite2p` Python package. If an existing Suite2p environment already contains NumPy, Matplotlib, and Tkinter, it can be used directly.

Replace `suite2p` below with the actual name of the environment:

```bash
conda activate suite2p
python -c "import numpy, matplotlib, tkinter; print('Environment OK')"
```

If the import test fails, creating the dedicated `pick_neuron_env` is safer than modifying a working Suite2p environment and potentially changing its package versions.

## 3. Run the program

In Terminal, move to the folder containing the script:

```bash
cd ~/Desktop/PickNeuron
conda activate pick_neuron_env
python pick_neuron.py
```

The script can also be launched with its absolute path:

```bash
conda activate pick_neuron_env
python "/Users/YOUR_NAME/Desktop/PickNeuron/pick_neuron.py"
```

Replace `YOUR_NAME` and the rest of the path with the actual location on the computer. If a Suite2p environment is being used, replace `pick_neuron_env` with that environment's name.

### Optional: create a double-click launcher on macOS

Create a plain-text file named `PickNeuron.command` with the following content:

```bash
#!/bin/zsh

source "/Users/YOUR_NAME/miniforge3/etc/profile.d/conda.sh"
conda activate pick_neuron_env
python "/Users/YOUR_NAME/Desktop/PickNeuron/pick_neuron.py"
```

The Conda installation may be inside `miniforge3`, `miniconda3`, or `anaconda3`. To find the correct base directory, run:

```bash
conda info --base
```

Append `/etc/profile.d/conda.sh` to the path returned by that command and use the result in the `source` line.

Make the launcher executable:

```bash
chmod +x "/Users/YOUR_NAME/Desktop/PickNeuron.command"
```

The program can then be opened by double-clicking `PickNeuron.command`. If the Python script is moved later, edit the final `python` line in the `.command` file and replace the old script path with the new absolute path.

## 4. Required data structure

The default data root is:

```text
/Volumes/kermit/Zimo/2_RawData
```

The GUI combines the data root, mouse name, and date to construct this path:

```text
{DATA_ROOT}/{mouse_name}/{date}/2P/suite2p
```

For example:

```text
/Volumes/kermit/Zimo/2_RawData/Mouse01/2026-09-23/2P/suite2p
```

The required Suite2p files may be located directly inside the `suite2p` folder or inside its `plane0` subfolder.

Each dataset must contain:

- `stat.npy`
- `iscell.npy`
- `F.npy`
- `Fneu.npy`
- `ops.npy`

The number of ROIs must agree across `stat.npy`, `iscell.npy`, `F.npy`, and `Fneu.npy`.

`ops.npy` must contain a two-dimensional `meanImg` or `meanImgE`. A `max_proj` entry is optional. If `max_proj` is not present, the Max projection control is disabled and the Mean projection remains available.

The current script automatically searches only the requested `suite2p` folder and its `plane0` subfolder. To review `plane1` or another plane, modify the `candidates` list in `_find_suite2p_plane()` accordingly.

## 5. Change the data path

### Change the data root for the current session

Click **Change** beside **Data root** on the Home screen and enter a new root path.

This change applies only to the current run. Restarting the program restores the default path written in the script.

### Change the default data root permanently

Open `pick_neuron.py` in a code editor and change this line near the top:

```python
DATA_ROOT = Path("/Volumes/kermit/Zimo/2_RawData")
```

For example:

```python
DATA_ROOT = Path("/Users/YOUR_NAME/Data")
```

### Use a different folder structure

If the data are not stored as `{root}/{mouse}/{date}/2P/suite2p`, edit this line in `_load_dataset()`:

```python
requested_path = self.data_root / mouse_name / date / "2P" / "suite2p"
```

For example, if the structure is `{root}/{mouse}/{date}/suite2p`, use:

```python
requested_path = self.data_root / mouse_name / date / "suite2p"
```

## 6. Adjustable parameters

The main display parameters are defined near the top of the script:

```python
CROP_RADIUS = 20
MASK_ALPHA = 0.5
MAX_TRACE_POINTS = 20_000
```

- `CROP_RADIUS = 20` displays a region extending 20 pixels from the ROI center in each direction, producing a `41 x 41` pixel crop.
- `MASK_ALPHA = 0.5` controls the transparency of the green ROI-mask overlay. Typical values range from `0.0` to `1.0`.
- `MAX_TRACE_POINTS = 20_000` limits the number of plotted trace points to keep navigation responsive. This affects display only and never changes `F.npy` or `Fneu.npy`.

For each ROI crop, the initial image limits are automatically calculated from the 1st and 99th percentiles. The **Brightness** and **Contrast** sliders adjust those display limits. **Reset display** restores brightness to `0` and contrast to `1.00`.

### Change the automatic rejection rules

When **Create iscell_new** is clicked, the program first labels every ROI as `-1`. It then automatically labels an ROI as `0` if either of these conditions is true:

1. `Fneu.min() > F.max()` for that ROI.
2. The ROI contains fewer than 5 pixels.

These rules are implemented in `create_iscell_new()`:

```python
neuropil_reject = np.min(self.Fneu, axis=1) > np.max(self.F, axis=1)
small_roi_reject = np.fromiter(
    (len(np.asarray(roi["ypix"])) < 5 for roi in self.stat),
    dtype=bool,
    count=len(self.stat),
)
```

For example, to reject ROIs containing fewer than 8 pixels, change `< 5` to `< 8`.

## 7. Use the GUI

### 7.1 Load a dataset

1. Confirm that **Data root** is correct. Use **Change** if necessary.
2. Enter the **Mouse name** and **Date** exactly as they appear in the folder names.
3. Click **Load dataset**. Pressing Return while the Date field is active also loads the dataset.
4. After loading succeeds, the GUI displays the Suite2p folder that was found and the total number of ROIs.

### 7.2 Create or load `iscell_new.npy`

After loading a dataset, choose one of the following:

- **Create iscell_new** creates a new label file using the automatic rejection rules. If `iscell_new.npy` already exists, the program asks for confirmation before replacing it. Confirming replacement deletes the previous manual labels from that file.
- **Load iscell_new data** loads an existing label file. The program reports an error if the file is missing, has the wrong number of labels, has an unsupported shape, or contains values other than `-1`, `0`, and `1`.

The program saves `iscell_new.npy` as a one-dimensional array with one value per ROI:

| Value | Meaning |
| ---: | --- |
| `1` | Is cell |
| `0` | Not cell |
| `-1` | Not classified |

When a new file is created, automatically rejected ROIs are assigned `0` and all remaining ROIs are assigned `-1`. Positive labels from the original Suite2p `iscell.npy` are not copied into the new file.

### 7.3 Choose a review queue

The Home screen contains three buttons showing the current label counts:

- **Is cell** reviews ROIs currently labeled `1`.
- **Not cell** reviews ROIs currently labeled `0`.
- **Not classified** reviews ROIs currently labeled `-1`.

When a group is opened, the program creates a fixed review queue from the ROIs that belong to that group at that moment. Each ROI in that queue is then shown once during the current pass, even when its label changes.

### 7.4 Inspect an ROI

The review screen contains:

- A projection crop centered on the ROI.
- **Mean** and **Max** controls for switching projections.
- An **ROI mask** checkbox that shows or hides the green ROI overlay.
- **Brightness** and **Contrast** sliders that change only the display.
- A blue `F` trace and a red `Fneu` trace.
- The real ROI index, position within the current queue, current `iscell_new` label, original Suite2p label, and Suite2p probability.

The ROI mask is constructed from `xpix` and `ypix` in `stat.npy`. The ROI center is taken from `stat[roi]["med"]` when available, with the median mask coordinate used as a fallback.

### 7.5 Assign a label

| Action | Button | Keyboard shortcut |
| --- | --- | --- |
| Label as cell | YES - Is cell | Space |
| Label as not cell | NO - Not cell | Z |
| Save and return Home | Return home | Esc |

Every YES or NO decision immediately updates and saves `iscell_new.npy`, then advances to the next ROI in the queue.

### 7.6 Jump within the current queue

Enter a number in **Go to position**, then press Return or click **Go**.

This number refers to a position within the current review queue, not the global ROI index. For example, if the upper-right corner shows:

```text
ROI 817  -  63/120
```

entering `63` goes to the 63rd item in the current group, which is ROI 817 in this example. It does not necessarily go to ROI 63.

During the same application session, pressing Esc and reopening the same group resumes at the current queue position. After the application is closed and restarted, the labels remain saved in `iscell_new.npy`, but the queue position is not retained.

## 8. Saving and merging labels

### Automatic saving

The program saves `iscell_new.npy` when:

- YES or NO is selected.
- Esc or **Return home** is used.
- The application window is closed.
- A new `iscell_new.npy` is created.

Saving is atomic: the program writes a temporary file and then replaces the destination file. This reduces the chance of leaving a partially written label file.

The original Suite2p `iscell.npy` is not modified until **Merge iscell_new and iscell** is explicitly selected and confirmed.

### Merge `iscell_new.npy` into `iscell.npy`

After clicking **Merge iscell_new and iscell** and confirming the warning:

- Every ROI with `iscell_new == 1` is written as `iscell[:, 0] = 1`.
- Every other ROI, including labels `0` and `-1`, is written as `iscell[:, 0] = 0`.
- The Suite2p probability column in `iscell[:, 1]` is preserved.

> **Important:** Any ROI that is still labeled `-1` becomes not cell during the merge. It is safest to confirm that the **Not classified** count is zero before merging.

Before the first merge, the program creates this backup in the same Suite2p folder:

```text
iscell_before_pick_neuron_merge.npy
```

The backup is created only if it does not already exist. It therefore preserves the `iscell.npy` state from before the first PickNeuron merge and is not overwritten by later merges.

## 9. Files created or modified by PickNeuron

| File | Behavior |
| --- | --- |
| `iscell_new.npy` | Created and updated by PickNeuron; contains the independent manual labels. |
| `iscell.npy` | Read on dataset load; modified only after an explicitly confirmed merge. |
| `iscell_before_pick_neuron_merge.npy` | One-time backup created before the first merge. |
| `stat.npy`, `F.npy`, `Fneu.npy`, `ops.npy` | Read only; never modified by this program. |

## 10. Troubleshooting

### The GUI does not open, or Python cannot import Tkinter

Confirm that the intended environment is active:

```bash
conda activate pick_neuron_env
which python
python -c "import tkinter; print(tkinter.TkVersion)"
```

If the import fails, recreating the dedicated environment with the command in Section 2 is usually the simplest solution.

### Conda remains on `Solving environment`

Use the environment command containing `--override-channels -c conda-forge`, and avoid pinning many exact package versions. If Mamba is already installed, the same environment can be created by replacing `conda create` with `mamba create`.

Warnings about version specifications ending in `.*` usually come from old package metadata or an older Conda solver. They do not by themselves mean that NumPy, Matplotlib, and Tkinter are incompatible.

### `Could not find a complete Suite2p output folder`

Check all of the following:

- The data root, mouse name, and date are spelled correctly.
- The path contains the expected `2P/suite2p` components.
- All five required `.npy` files are present directly in `suite2p` or in `suite2p/plane0`.
- The external or network drive has been mounted under `/Volumes`.

### The Max projection option is disabled

The loaded `ops.npy` does not contain `max_proj`. The Mean projection can still be used normally.

### Saving fails

Confirm that the Suite2p folder is not read-only and that the current user has permission to create and replace files in it. PickNeuron must be able to write `iscell_new.npy` and, during a merge, update `iscell.npy`.

### The application does not resume at the same position after restarting

ROI labels are saved, but the current queue position exists only in memory. Restarting the program rebuilds a review queue from the labels currently stored in `iscell_new.npy`.

### The application takes a long time to open a dataset

Check that the volume under `/Volumes` is mounted and responsive. Loading from a slow external or network drive can delay access to `stat.npy`, `ops.npy`, and the trace arrays.

## 11. Recommended safety practices

- Test the program on a copied dataset before using it on important data for the first time.
- Do not click Merge if the independent `iscell_new.npy` labels are sufficient for the analysis.
- Confirm that the **Not classified** count is zero before merging.
- Keep `iscell_before_pick_neuron_merge.npy` as a recovery copy.
- Avoid editing `.npy` files manually unless their shape and data type are understood.
