#!/usr/bin/env python3
"""Review Suite2p ROIs one at a time.

The GUI builds a dataset path as

    {data_root}/{mouse_name}/{date}/2P/suite2p

and accepts Suite2p files either directly in that directory or in its
``plane0`` subdirectory.

Required files
--------------
stat.npy, iscell.npy, F.npy, Fneu.npy, ops.npy

The original Suite2p ``iscell.npy`` is read-only. Manual labels are loaded from
and saved to a separate one-dimensional ``iscell_new.npy`` containing only
``1`` (cell), ``0`` (not cell), or ``-1`` (not classified). Every Yes, No,
Home, and window-close action atomically saves ``iscell_new.npy``.

Run
---
    python pick_neuron.py

Dependencies (compatible with suite2p environment): numpy, matplotlib, and tkinter (normally included with Python).
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure


# Change this one line if the data root moves.
DATA_ROOT = Path("/Volumes/kermit/Zimo/2_RawData")

CROP_RADIUS = 20  # radius 20 -> a 41 x 41 pixel view
MASK_ALPHA = 0.5
MAX_TRACE_POINTS = 20_000  # display decimation only; saved data are untouched

LABEL_NAMES = {
    1: "Is cell",
    0: "Not cell",
    -1: "Not classified",
}


def _load_object_dict(path: Path) -> dict[str, Any]:
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.shape == ():
        obj = obj.item()
    if not isinstance(obj, dict):
        raise ValueError(f"{path.name} does not contain a dictionary.")
    return obj


def _find_suite2p_plane(suite2p_path: Path) -> Path:
    """Return the directory that directly contains the five required files."""
    required = ("stat.npy", "iscell.npy", "F.npy", "Fneu.npy", "ops.npy")
    candidates = [suite2p_path, suite2p_path / "plane0"]

    for candidate in candidates:
        if all((candidate / name).is_file() for name in required):
            return candidate

    missing_messages = []
    for candidate in candidates:
        missing = [name for name in required if not (candidate / name).is_file()]
        missing_messages.append(f"{candidate}: missing {', '.join(missing)}")
    raise FileNotFoundError(
        "Could not find a complete Suite2p output folder.\n\n"
        + "\n".join(missing_messages)
    )


def _paste_projection_into_full_frame(
    image: np.ndarray,
    full_shape: tuple[int, int],
    ops: dict[str, Any],
) -> np.ndarray:
    """Align Suite2p's sometimes-cropped max projection to meanImg coordinates."""
    image = np.asarray(image, dtype=float).squeeze()
    if image.ndim != 2:
        raise ValueError(f"Projection must be 2-D, got shape {image.shape}.")
    if image.shape == full_shape:
        return image

    full = np.full(full_shape, np.nan, dtype=float)
    yrange = np.asarray(ops.get("yrange", []), dtype=int).ravel()
    xrange = np.asarray(ops.get("xrange", []), dtype=int).ravel()

    # Some Suite2p versions store each retained coordinate explicitly.
    if len(yrange) == image.shape[0] and len(xrange) == image.shape[1]:
        valid_y = (yrange >= 0) & (yrange < full_shape[0])
        valid_x = (xrange >= 0) & (xrange < full_shape[1])
        if valid_y.all() and valid_x.all():
            full[np.ix_(yrange, xrange)] = image
            return full

    # Other versions store endpoints or an array whose first entry is the crop
    # origin. In both cases the image dimensions determine the pasted extent.
    y0 = int(yrange[0]) if len(yrange) else 0
    x0 = int(xrange[0]) if len(xrange) else 0
    y1 = min(y0 + image.shape[0], full_shape[0])
    x1 = min(x0 + image.shape[1], full_shape[1])
    src_h = max(0, y1 - y0)
    src_w = max(0, x1 - x0)
    if y0 < 0 or x0 < 0 or src_h == 0 or src_w == 0:
        raise ValueError(
            f"Cannot align projection shape {image.shape} to frame {full_shape}."
        )
    full[y0:y1, x0:x1] = image[:src_h, :src_w]
    return full


def _crop_centered(
    image: np.ndarray,
    center_y: int,
    center_x: int,
    radius: int = CROP_RADIUS,
    fill_value: float | bool = np.nan,
) -> np.ndarray:
    """Return a fixed-size crop, padding near image borders."""
    size = 2 * radius + 1
    dtype = bool if isinstance(fill_value, (bool, np.bool_)) else float
    crop = np.full((size, size), fill_value, dtype=dtype)

    src_y0 = max(0, center_y - radius)
    src_y1 = min(image.shape[0], center_y + radius + 1)
    src_x0 = max(0, center_x - radius)
    src_x1 = min(image.shape[1], center_x + radius + 1)

    dst_y0 = src_y0 - (center_y - radius)
    dst_x0 = src_x0 - (center_x - radius)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    crop[dst_y0:dst_y1, dst_x0:dst_x1] = image[
        src_y0:src_y1, src_x0:src_x1
    ]
    return crop


def _trace_for_display(trace: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Limit plotted points so moving between ROIs remains responsive."""
    trace = np.asarray(trace, dtype=float).ravel()
    step = max(1, int(np.ceil(trace.size / MAX_TRACE_POINTS)))
    x = np.arange(0, trace.size, step)
    return x, trace[::step]


def _auto_display_limits(image: np.ndarray) -> tuple[float, float]:
    """Robust default brightness/contrast limits for one image crop."""
    finite = np.asarray(image, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return 0.0, 1.0

    vmin, vmax = np.percentile(finite, [1, 99])
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return 0.0, 1.0
    if vmin == vmax:
        padding = max(abs(float(vmin)) * 0.01, 1.0)
        return float(vmin - padding), float(vmax + padding)
    return float(vmin), float(vmax)


def _adjust_display_limits(
    base_vmin: float,
    base_vmax: float,
    brightness: float,
    contrast: float,
) -> tuple[float, float]:
    """Convert GUI brightness/contrast values into Matplotlib color limits."""
    base_span = max(float(base_vmax - base_vmin), np.finfo(float).eps)
    contrast = max(float(contrast), 0.05)
    center = (base_vmin + base_vmax) / 2.0

    # Lower color limits make a fixed pixel value appear brighter.
    center -= (float(brightness) / 100.0) * 0.5 * base_span
    adjusted_span = base_span / contrast
    return center - adjusted_span / 2.0, center + adjusted_span / 2.0


@dataclass
class ReviewSession:
    label: int
    roi_indices: np.ndarray
    position: int = 0

    @property
    def finished(self) -> bool:
        return self.position >= len(self.roi_indices)

    @property
    def current_roi(self) -> int:
        return int(self.roi_indices[self.position])

    def jump_to(self, one_based_position: int) -> None:
        """Move to a 1-based position within this fixed review queue."""
        if not 1 <= one_based_position <= len(self.roi_indices):
            raise ValueError(
                f"Position must be between 1 and {len(self.roi_indices)}."
            )
        self.position = one_based_position - 1


class Suite2PDataset:
    """Loaded Suite2p arrays plus independent manual labels and image helpers."""

    def __init__(self, suite2p_path: Path):
        self.folder = _find_suite2p_plane(suite2p_path)
        self.stat = np.load(self.folder / "stat.npy", allow_pickle=True)
        self.suite2p_iscell = np.load(self.folder / "iscell.npy")
        self.F = np.load(self.folder / "F.npy", mmap_mode="r")
        self.Fneu = np.load(self.folder / "Fneu.npy", mmap_mode="r")
        self.ops = _load_object_dict(self.folder / "ops.npy")
        self.iscell_path = self.folder / "iscell.npy"
        self.iscell_new_path = self.folder / "iscell_new.npy"
        self.iscell_merge_backup_path = (
            self.folder / "iscell_before_pick_neuron_merge.npy"
        )
        self.iscell_new: np.ndarray | None = None

        self._validate_shapes()
        self.mean_projection = self._load_mean_projection()
        self.max_projection = self._load_max_projection()
        self.frame_shape = self.mean_projection.shape

    def _validate_shapes(self) -> None:
        n_rois = len(self.stat)
        if self.suite2p_iscell.ndim != 2 or self.suite2p_iscell.shape[1] < 2:
            raise ValueError(
                "iscell.npy should have shape (n_ROIs, 2); "
                f"got {self.suite2p_iscell.shape}."
            )
        shapes = {
            "iscell": self.suite2p_iscell.shape[0],
            "F": self.F.shape[0],
            "Fneu": self.Fneu.shape[0],
        }
        wrong = {name: n for name, n in shapes.items() if n != n_rois}
        if wrong:
            raise ValueError(
                f"stat.npy contains {n_rois} ROIs, but row counts differ: {wrong}."
            )

    def _load_mean_projection(self) -> np.ndarray:
        for key in ("meanImg", "meanImgE"):
            if key in self.ops:
                image = np.asarray(self.ops[key], dtype=float).squeeze()
                if image.ndim == 2:
                    return image
        raise KeyError("ops.npy contains neither a 2-D meanImg nor meanImgE.")

    def _load_max_projection(self) -> np.ndarray | None:
        if "max_proj" not in self.ops:
            return None
        return _paste_projection_into_full_frame(
            self.ops["max_proj"], self.mean_projection.shape, self.ops
        )

    def count(self, label: int) -> int:
        if self.iscell_new is None:
            return 0
        return int(np.count_nonzero(self.iscell_new == label))

    def load_iscell_new(self) -> None:
        """Load and validate an existing independent label file."""
        if not self.iscell_new_path.exists():
            raise FileNotFoundError(
                "No iscell_new.npy exists for this dataset. "
                "Click Create iscell_new first."
            )
        raw = np.load(self.iscell_new_path)
        if raw.ndim == 1:
            labels = raw
        elif raw.ndim == 2 and raw.shape[1] == 1:
            labels = raw[:, 0]
        else:
            raise ValueError(
                "iscell_new.npy must be a one-dimensional array or have "
                f"shape (n_ROIs, 1); got {raw.shape}."
            )
        if len(labels) != len(self.stat):
            raise ValueError(
                f"iscell_new.npy contains {len(labels)} labels, but this "
                f"dataset contains {len(self.stat)} ROIs."
            )
        if not np.all(np.isin(labels, (-1, 0, 1))):
            invalid = np.unique(labels[~np.isin(labels, (-1, 0, 1))])
            raise ValueError(
                "iscell_new.npy may contain only -1, 0, and 1; found "
                f"{invalid.tolist()}."
            )
        self.iscell_new = labels.astype(np.int8, copy=True)

    def create_iscell_new(self) -> dict[str, int]:
        """Create fresh labels using the two conservative automatic rejects."""
        labels = np.full(len(self.stat), -1, dtype=np.int8)
        neuropil_reject = np.min(self.Fneu, axis=1) > np.max(self.F, axis=1)
        small_roi_reject = np.fromiter(
            (len(np.asarray(roi["ypix"])) < 5 for roi in self.stat),
            dtype=bool,
            count=len(self.stat),
        )
        auto_reject = neuropil_reject | small_roi_reject
        labels[auto_reject] = 0
        self.iscell_new = labels
        self.save_iscell_new()
        return {
            "neuropil": int(np.count_nonzero(neuropil_reject)),
            "small_roi": int(np.count_nonzero(small_roi_reject)),
            "total": int(np.count_nonzero(auto_reject)),
        }

    def roi_center(self, roi_index: int) -> tuple[int, int]:
        roi = self.stat[roi_index]
        ypix = np.asarray(roi["ypix"], dtype=int)
        xpix = np.asarray(roi["xpix"], dtype=int)
        if not len(ypix) or not len(xpix):
            raise ValueError(f"ROI {roi_index} has an empty pixel mask.")

        med = np.asarray(roi.get("med", []), dtype=float).ravel()
        if len(med) >= 2 and np.isfinite(med[:2]).all():
            return int(round(med[0])), int(round(med[1]))
        return int(round(np.median(ypix))), int(round(np.median(xpix)))

    def roi_mask(self, roi_index: int) -> np.ndarray:
        mask = np.zeros(self.frame_shape, dtype=bool)
        roi = self.stat[roi_index]
        ypix = np.asarray(roi["ypix"], dtype=int)
        xpix = np.asarray(roi["xpix"], dtype=int)
        valid = (
            (ypix >= 0)
            & (ypix < self.frame_shape[0])
            & (xpix >= 0)
            & (xpix < self.frame_shape[1])
        )
        mask[ypix[valid], xpix[valid]] = True
        return mask

    def crop_for_roi(
        self, roi_index: int, projection_name: str
    ) -> tuple[np.ndarray, np.ndarray]:
        center_y, center_x = self.roi_center(roi_index)
        if projection_name == "Max" and self.max_projection is not None:
            projection = self.max_projection
        else:
            projection = self.mean_projection
        image_crop = _crop_centered(projection, center_y, center_x)
        mask_crop = _crop_centered(
            self.roi_mask(roi_index),
            center_y,
            center_x,
            fill_value=False,
        )
        return image_crop, mask_crop

    def save_iscell_new(self) -> None:
        """Atomically save the independent one-dimensional label array."""
        if self.iscell_new is None:
            raise RuntimeError("iscell_new data have not been loaded or created yet.")
        temp_path = self.folder / ".iscell_new_cell_validator.tmp"
        try:
            with temp_path.open("wb") as handle:
                np.save(handle, np.asarray(self.iscell_new, dtype=np.int8))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self.iscell_new_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def merge_iscell_new_into_iscell(self) -> tuple[int, int]:
        """Replace iscell[:, 0] from manual positives; preserve probabilities."""
        if self.iscell_new is None:
            raise RuntimeError("iscell_new data have not been loaded or created yet.")

        if not self.iscell_merge_backup_path.exists():
            shutil.copy2(self.iscell_path, self.iscell_merge_backup_path)

        merged_labels = (self.iscell_new == 1).astype(
            self.suite2p_iscell[:, 0].dtype,
            copy=False,
        )
        previous_labels = self.suite2p_iscell[:, 0].copy()
        self.suite2p_iscell[:, 0] = merged_labels

        temp_path = self.folder / ".iscell_pick_neuron_merge.tmp"
        try:
            with temp_path.open("wb") as handle:
                np.save(handle, self.suite2p_iscell)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self.iscell_path)
        except Exception:
            self.suite2p_iscell[:, 0] = previous_labels
            raise
        finally:
            if temp_path.exists():
                temp_path.unlink()

        n_cell = int(np.count_nonzero(merged_labels == 1))
        return n_cell, len(merged_labels) - n_cell


class CellValidatorApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Suite2p Cell Validator")
        self.geometry("1200x850")
        self.minsize(900, 650)

        self.dataset: Suite2PDataset | None = None
        self.sessions: dict[int, ReviewSession] = {}
        self.active_session: ReviewSession | None = None
        self.data_root = DATA_ROOT
        self.data_root_var = tk.StringVar(value=str(self.data_root))
        self.projection_var = tk.StringVar(value="Mean")
        self.brightness_var = tk.DoubleVar(value=0.0)
        self.contrast_var = tk.DoubleVar(value=1.0)
        self.brightness_text_var = tk.StringVar(value="0")
        self.contrast_text_var = tk.StringVar(value="1.00")
        self.mask_visible_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Enter mouse name and date, then load.")
        self.home_dataset_var = tk.StringVar(value="No dataset loaded")
        self.review_header_var = tk.StringVar(value="")
        self.roi_info_var = tk.StringVar(value="")
        self.jump_position_var = tk.StringVar(value="1")
        self.jump_total_var = tk.StringVar(value="/ 0")
        self._base_vmin = 0.0
        self._base_vmax = 1.0
        self._background_artist = None
        self._mask_artist = None

        self._build_home()
        self._build_review()
        self._show_home()

        self.bind("<space>", self._space_pressed)
        self.bind("<KeyPress-z>", self._z_pressed)
        self.bind("<KeyPress-Z>", self._z_pressed)
        self.bind("<Escape>", self._escape_pressed)
        self.protocol("WM_DELETE_WINDOW", self._close_window)

    def _build_home(self) -> None:
        self.home_frame = ttk.Frame(self, padding=28)
        self.home_frame.columnconfigure(1, weight=1)

        ttk.Label(
            self.home_frame,
            text="Suite2p Cell Validator",
            font=("TkDefaultFont", 20, "bold"),
        ).grid(row=0, column=0, columnspan=3, pady=(0, 22))

        ttk.Label(self.home_frame, text="Data root").grid(
            row=1, column=0, sticky="e", padx=(0, 10), pady=5
        )
        ttk.Entry(
            self.home_frame,
            textvariable=self.data_root_var,
            state="readonly",
        ).grid(
            row=1, column=1, sticky="ew", pady=5
        )
        ttk.Button(
            self.home_frame,
            text="Change",
            command=self._change_data_root,
        ).grid(
            row=1, column=2, padx=(12, 0), sticky="ew", pady=5
        )

        ttk.Label(self.home_frame, text="Mouse name").grid(
            row=2, column=0, sticky="e", padx=(0, 10), pady=5
        )
        self.mouse_entry = ttk.Entry(self.home_frame, width=30)
        self.mouse_entry.grid(row=2, column=1, sticky="ew", pady=5)

        ttk.Label(self.home_frame, text="Date").grid(
            row=3, column=0, sticky="e", padx=(0, 10), pady=5
        )
        self.date_entry = ttk.Entry(self.home_frame, width=30)
        self.date_entry.grid(row=3, column=1, sticky="ew", pady=5)
        self.date_entry.bind("<Return>", lambda _event: self._load_dataset())

        ttk.Button(self.home_frame, text="Load dataset", command=self._load_dataset).grid(
            row=2, column=2, rowspan=2, padx=(12, 0), sticky="nsew", pady=5
        )

        ttk.Separator(self.home_frame).grid(
            row=4, column=0, columnspan=3, sticky="ew", pady=20
        )
        ttk.Label(
            self.home_frame,
            textvariable=self.home_dataset_var,
            wraplength=950,
            justify="center",
        ).grid(row=5, column=0, columnspan=3, pady=(0, 18))

        new_file_frame = ttk.Frame(self.home_frame)
        new_file_frame.grid(
            row=6, column=0, columnspan=3, padx=10, pady=(0, 18)
        )
        self.create_new_button = ttk.Button(
            new_file_frame,
            text="Create iscell_new",
            command=self._create_iscell_new,
            state="disabled",
            width=24,
        )
        self.create_new_button.grid(row=0, column=0, padx=7, ipady=7)
        self.load_new_button = ttk.Button(
            new_file_frame,
            text="Load iscell_new data",
            command=self._load_iscell_new,
            state="disabled",
            width=24,
        )
        self.load_new_button.grid(row=0, column=1, padx=7, ipady=7)

        group_frame = ttk.Frame(self.home_frame)
        group_frame.grid(row=7, column=0, columnspan=3)
        self.is_cell_button = ttk.Button(
            group_frame,
            text="Is cell (0)",
            command=lambda: self._start_or_resume(1),
            state="disabled",
            width=20,
        )
        self.is_cell_button.grid(row=0, column=0, padx=10, ipady=13)
        self.not_cell_button = ttk.Button(
            group_frame,
            text="Not cell (0)",
            command=lambda: self._start_or_resume(0),
            state="disabled",
            width=20,
        )
        self.not_cell_button.grid(row=0, column=1, padx=10, ipady=13)
        self.not_classified_button = ttk.Button(
            group_frame,
            text="Not classified (0)",
            command=lambda: self._start_or_resume(-1),
            state="disabled",
            width=20,
        )
        self.not_classified_button.grid(row=0, column=2, padx=10, ipady=13)

        ttk.Label(
            self.home_frame,
            textvariable=self.status_var,
            foreground="#555555",
            wraplength=950,
            justify="center",
        ).grid(row=8, column=0, columnspan=3, pady=(22, 14))

        self.merge_button = ttk.Button(
            self.home_frame,
            text="Merge iscell_new and iscell",
            command=self._merge_labels,
            state="disabled",
        )
        self.merge_button.grid(
            row=9, column=0, columnspan=3, padx=10, pady=(8, 0), ipady=7
        )

    def _build_review(self) -> None:
        self.review_frame = ttk.Frame(self, padding=12)
        self.review_frame.columnconfigure(0, weight=1)
        self.review_frame.rowconfigure(3, weight=1)

        header = ttk.Frame(self.review_frame)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(1, weight=1)
        ttk.Label(
            header, textvariable=self.review_header_var, font=("TkDefaultFont", 13, "bold")
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(header, textvariable=self.roi_info_var).grid(
            row=0, column=1, sticky="e"
        )
        navigation = ttk.Frame(self.review_frame)
        navigation.grid(row=1, column=0, pady=(8, 2))
        ttk.Label(navigation, text="Go to position").grid(
            row=0, column=0, padx=(0, 5)
        )
        self.jump_entry = ttk.Entry(
            navigation,
            textvariable=self.jump_position_var,
            width=5,
            justify="right",
        )
        self.jump_entry.grid(row=0, column=1)
        self.jump_entry.bind("<Return>", lambda _event: self._jump_to_position())
        ttk.Label(navigation, textvariable=self.jump_total_var).grid(
            row=0, column=2, padx=(3, 8)
        )
        ttk.Button(navigation, text="Go", command=self._jump_to_position).grid(
            row=0, column=3
        )

        projection_frame = ttk.Frame(self.review_frame)
        projection_frame.grid(row=2, column=0, pady=(6, 4))
        ttk.Label(projection_frame, text="Projection:").grid(row=0, column=0, padx=5)
        self.mean_radio = ttk.Radiobutton(
            projection_frame,
            text="Mean",
            value="Mean",
            variable=self.projection_var,
            command=self._draw_current_roi,
        )
        self.mean_radio.grid(row=0, column=1, padx=5)
        self.max_radio = ttk.Radiobutton(
            projection_frame,
            text="Max",
            value="Max",
            variable=self.projection_var,
            command=self._draw_current_roi,
        )
        self.max_radio.grid(row=0, column=2, padx=5)
        ttk.Checkbutton(
            projection_frame,
            text="ROI mask",
            variable=self.mask_visible_var,
            command=self._apply_display_controls,
        ).grid(row=0, column=3, padx=(20, 5))

        ttk.Label(projection_frame, text="Brightness").grid(
            row=1, column=0, padx=5, pady=(8, 0), sticky="e"
        )
        ttk.Scale(
            projection_frame,
            from_=-100,
            to=100,
            variable=self.brightness_var,
            command=self._display_slider_changed,
            length=210,
        ).grid(row=1, column=1, columnspan=2, padx=5, pady=(8, 0), sticky="ew")
        ttk.Label(
            projection_frame,
            textvariable=self.brightness_text_var,
            width=5,
            anchor="e",
        ).grid(row=1, column=3, padx=5, pady=(8, 0), sticky="w")

        ttk.Label(projection_frame, text="Contrast").grid(
            row=2, column=0, padx=5, sticky="e"
        )
        ttk.Scale(
            projection_frame,
            from_=0.25,
            to=4.0,
            variable=self.contrast_var,
            command=self._display_slider_changed,
            length=210,
        ).grid(row=2, column=1, columnspan=2, padx=5, sticky="ew")
        ttk.Label(
            projection_frame,
            textvariable=self.contrast_text_var,
            width=5,
            anchor="e",
        ).grid(row=2, column=3, padx=5, sticky="w")
        ttk.Button(
            projection_frame,
            text="Reset display",
            command=self._reset_display_controls,
        ).grid(row=1, column=4, rowspan=2, padx=(14, 0), sticky="ns")

        self.figure = Figure(figsize=(11, 7), dpi=100, constrained_layout=True)
        grid = self.figure.add_gridspec(
            2, 3, height_ratios=(1.0, 1.25), width_ratios=(1.0, 1.15, 1.0)
        )
        self.image_ax = self.figure.add_subplot(grid[0, 1])
        self.trace_ax = self.figure.add_subplot(grid[1, :])
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.review_frame)
        self.canvas.get_tk_widget().grid(row=3, column=0, sticky="nsew")

        actions = ttk.Frame(self.review_frame)
        actions.grid(row=4, column=0, pady=(8, 0))
        ttk.Button(
            actions,
            text="YES — Is cell  [Space]",
            command=lambda: self._mark_current(1),
            width=24,
        ).grid(row=0, column=0, padx=8, ipady=7)
        ttk.Button(
            actions,
            text="NO — Not cell  [Z]",
            command=lambda: self._mark_current(0),
            width=24,
        ).grid(row=0, column=1, padx=8, ipady=7)
        ttk.Button(
            actions,
            text="Return home  [Esc]",
            command=self._return_home,
            width=24,
        ).grid(row=0, column=2, padx=8, ipady=7)

    def _change_data_root(self) -> None:
        new_root = simpledialog.askstring(
            "Change data root",
            "Enter the new data root path:",
            initialvalue=str(self.data_root),
            parent=self,
        )
        if new_root is None:
            return
        new_root = new_root.strip()
        if not new_root:
            messagebox.showwarning("Invalid path", "The data root cannot be empty.")
            return
        self.data_root = Path(new_root).expanduser()
        self.data_root_var.set(str(self.data_root))
        self.status_var.set(f"Data root changed to {self.data_root}")

    def _load_dataset(self) -> None:
        mouse_name = self.mouse_entry.get().strip()
        date = self.date_entry.get().strip()
        if not mouse_name or not date:
            messagebox.showwarning("Missing input", "Enter both mouse name and date.")
            return

        requested_path = self.data_root / mouse_name / date / "2P" / "suite2p"
        try:
            dataset = Suite2PDataset(requested_path)
        except Exception as exc:
            messagebox.showerror("Could not load dataset", str(exc))
            self.status_var.set(f"Load failed: {exc}")
            return

        self.dataset = dataset
        self.sessions.clear()
        self.active_session = None
        self.home_dataset_var.set(str(dataset.folder))
        self.status_var.set(
            f"Loaded {len(dataset.stat)} ROIs. Create or load iscell_new to continue."
        )
        self.create_new_button.configure(state="normal")
        self.load_new_button.configure(state="normal")
        self.merge_button.configure(state="disabled")
        self.is_cell_button.configure(state="disabled", text="Is cell (0)")
        self.not_cell_button.configure(state="disabled", text="Not cell (0)")
        self.not_classified_button.configure(
            state="disabled", text="Not classified (0)"
        )
        self.max_radio.configure(
            state="normal" if dataset.max_projection is not None else "disabled"
        )
        if dataset.max_projection is None:
            self.projection_var.set("Mean")
            self.status_var.set(
                f"Loaded {len(dataset.stat)} ROIs. ops.npy has no max_proj; "
                "only Mean is available. Create or load iscell_new to continue."
            )
        self._update_counts()

    def _create_iscell_new(self) -> None:
        if self.dataset is None:
            return

        if self.dataset.iscell_new_path.exists():
            recreate = messagebox.askyesno(
                "Replace iscell_new.npy?",
                "iscell_new.npy already exists. Creating a new one will replace "
                "all current manual labels. Continue?",
            )
            if not recreate:
                return
        try:
            summary = self.dataset.create_iscell_new()
        except Exception as exc:
            messagebox.showerror("Could not create iscell_new.npy", str(exc))
            self.status_var.set(f"iscell_new creation failed: {exc}")
            return

        self._enable_label_controls()
        self.status_var.set(
            "Created iscell_new.npy. Automatically labeled "
            f"{summary['total']} ROIs as not cell "
            f"(Fneu threshold: {summary['neuropil']}; "
            f"<5 pixels: {summary['small_roi']}; overlap counted once)."
        )

    def _load_iscell_new(self) -> None:
        if self.dataset is None:
            return

        try:
            self.dataset.load_iscell_new()
        except Exception as exc:
            messagebox.showerror("Could not load iscell_new.npy", str(exc))
            self.status_var.set(f"iscell_new load failed: {exc}")
            return

        self._enable_label_controls()
        self.status_var.set("Loaded existing iscell_new.npy.")

    def _enable_label_controls(self) -> None:
        self.sessions.clear()
        self.active_session = None
        self.is_cell_button.configure(state="normal")
        self.not_cell_button.configure(state="normal")
        self.not_classified_button.configure(state="normal")
        self.merge_button.configure(state="normal")
        self._update_counts()

    def _merge_labels(self) -> None:
        if self.dataset is None or self.dataset.iscell_new is None:
            return

        n_positive = self.dataset.count(1)
        n_other = len(self.dataset.iscell_new) - n_positive
        confirmed = messagebox.askyesno(
            "Merge labels into iscell.npy?",
            "This will overwrite the first column of iscell.npy:\n\n"
            f"• {n_positive} ROIs with iscell_new == 1 will become 1\n"
            f"• The other {n_other} ROIs will become 0\n\n"
            "The Suite2p probability column will not change. Continue?",
        )
        if not confirmed:
            return

        try:
            n_cell, n_not_cell = self.dataset.merge_iscell_new_into_iscell()
        except Exception as exc:
            messagebox.showerror("Merge failed", str(exc))
            self.status_var.set(f"Merge failed: {exc}")
            return

        self.status_var.set(
            f"Merged into iscell.npy: {n_cell} cell, {n_not_cell} not cell. "
            "Probability column unchanged. Original labels are backed up in "
            "iscell_before_pick_neuron_merge.npy."
        )
        messagebox.showinfo(
            "Merge complete",
            f"iscell.npy updated: {n_cell} cell and {n_not_cell} not cell.\n\n"
            "The probability column was preserved.",
        )

    def _update_counts(self) -> None:
        if self.dataset is None or self.dataset.iscell_new is None:
            return
        self.is_cell_button.configure(text=f"Is cell ({self.dataset.count(1)})")
        self.not_cell_button.configure(text=f"Not cell ({self.dataset.count(0)})")
        self.not_classified_button.configure(
            text=f"Not classified ({self.dataset.count(-1)})"
        )

    def _start_or_resume(self, label: int) -> None:
        if self.dataset is None or self.dataset.iscell_new is None:
            return

        session = self.sessions.get(label)
        if session is None or session.finished:
            indices = np.flatnonzero(self.dataset.iscell_new == label)
            if not len(indices):
                messagebox.showinfo(
                    "No ROIs", "There are currently no ROIs in this group."
                )
                return
            session = ReviewSession(label=label, roi_indices=indices.copy())
            self.sessions[label] = session

        self.active_session = session
        group_name = LABEL_NAMES[label]
        self.jump_position_var.set(str(session.position + 1))
        self.jump_total_var.set(f"/ {len(session.roi_indices)}")
        self.review_header_var.set(
            f"Reviewing: {group_name}  •  {self.dataset.folder.name}"
        )
        self.home_frame.pack_forget()
        self.review_frame.pack(fill="both", expand=True)
        self._draw_current_roi()
        self.focus_set()

    def _draw_current_roi(self) -> None:
        if (
            self.dataset is None
            or self.active_session is None
            or self.active_session.finished
        ):
            return

        session = self.active_session
        roi_index = session.current_roi
        image_crop, mask_crop = self.dataset.crop_for_roi(
            roi_index, self.projection_var.get()
        )

        self.image_ax.clear()
        self._base_vmin, self._base_vmax = _auto_display_limits(image_crop)
        vmin, vmax = _adjust_display_limits(
            self._base_vmin,
            self._base_vmax,
            self.brightness_var.get(),
            self.contrast_var.get(),
        )
        self._background_artist = self.image_ax.imshow(
            image_crop, cmap="gray", interpolation="nearest", vmin=vmin, vmax=vmax
        )
        masked_roi = np.ma.masked_where(~mask_crop, mask_crop.astype(float))
        self._mask_artist = self.image_ax.imshow(
            masked_roi,
            cmap=ListedColormap(["#00E676"]),
            interpolation="nearest",
            alpha=MASK_ALPHA,
            vmin=0,
            vmax=1,
        )
        self._mask_artist.set_visible(self.mask_visible_var.get())
        self.image_ax.set_title(
            f"{self.projection_var.get()} projection • ROI mask (alpha={MASK_ALPHA:g})"
        )
        self.image_ax.set_xticks([])
        self.image_ax.set_yticks([])

        self.trace_ax.clear()
        x_f, f_display = _trace_for_display(self.dataset.F[roi_index])
        x_n, fneu_display = _trace_for_display(self.dataset.Fneu[roi_index])
        self.trace_ax.plot(x_f, f_display, color="#1565C0", linewidth=0.75, label="F")
        self.trace_ax.plot(
            x_n, fneu_display, color="#D32F2F", linewidth=0.75, label="Fneu"
        )
        self.trace_ax.set_xlabel("Frame")
        self.trace_ax.set_ylabel("Fluorescence")
        self.trace_ax.legend(loc="upper right", frameon=False)
        self.trace_ax.margins(x=0.005)

        probability = float(self.dataset.suite2p_iscell[roi_index, 1])
        suite2p_label = int(self.dataset.suite2p_iscell[roi_index, 0])
        current_label = int(self.dataset.iscell_new[roi_index])
        self.jump_position_var.set(str(session.position + 1))
        self.jump_total_var.set(f"/ {len(session.roi_indices)}")
        self.roi_info_var.set(
            f"ROI {roi_index}  •  {session.position + 1}/{len(session.roi_indices)}"
            f"  •  iscell_new {current_label}  •  Suite2p {suite2p_label}"
            f" (p={probability:.3f})"
        )
        self.canvas.draw_idle()

    def _jump_to_position(self) -> None:
        if self.active_session is None or self.active_session.finished:
            return
        raw_position = self.jump_position_var.get().strip()
        try:
            position = int(raw_position)
            self.active_session.jump_to(position)
        except ValueError:
            messagebox.showwarning(
                "Invalid position",
                f"Enter a whole number between 1 and "
                f"{len(self.active_session.roi_indices)}.",
            )
            self.jump_position_var.set(str(self.active_session.position + 1))
            return
        self._draw_current_roi()
        self.focus_set()

    def _display_slider_changed(self, _value: str | None = None) -> None:
        self.brightness_text_var.set(f"{self.brightness_var.get():.0f}")
        self.contrast_text_var.set(f"{self.contrast_var.get():.2f}")
        self._apply_display_controls()

    def _apply_display_controls(self) -> None:
        if self._background_artist is None:
            return
        vmin, vmax = _adjust_display_limits(
            self._base_vmin,
            self._base_vmax,
            self.brightness_var.get(),
            self.contrast_var.get(),
        )
        self._background_artist.set_clim(vmin, vmax)
        if self._mask_artist is not None:
            self._mask_artist.set_visible(self.mask_visible_var.get())
        self.canvas.draw_idle()

    def _reset_display_controls(self) -> None:
        self.brightness_var.set(0.0)
        self.contrast_var.set(1.0)
        self.brightness_text_var.set("0")
        self.contrast_text_var.set("1.00")
        self._apply_display_controls()

    def _mark_current(self, label: int) -> None:
        if (
            self.dataset is None
            or self.dataset.iscell_new is None
            or self.active_session is None
            or self.active_session.finished
        ):
            return

        roi_index = self.active_session.current_roi
        previous_label = self.dataset.iscell_new[roi_index]
        self.dataset.iscell_new[roi_index] = label
        try:
            self.dataset.save_iscell_new()
        except Exception as exc:
            self.dataset.iscell_new[roi_index] = previous_label
            messagebox.showerror(
                "Save failed",
                f"ROI {roi_index} was not advanced because iscell_new.npy "
                f"could not be saved.\n\n{exc}",
            )
            return

        self.active_session.position += 1
        self._update_counts()
        if self.active_session.finished:
            group_name = LABEL_NAMES[self.active_session.label]
            self.status_var.set(
                f"Finished the {group_name} review queue. iscell_new.npy is saved."
            )
            self._show_home()
        else:
            self._draw_current_roi()

    def _return_home(self) -> None:
        if self.dataset is not None and self.dataset.iscell_new is not None:
            try:
                self.dataset.save_iscell_new()
            except Exception as exc:
                messagebox.showerror("Save failed", str(exc))
                return
            self.status_var.set(
                "iscell_new.npy saved. Re-enter this group to resume at the current ROI."
            )
        self._show_home()

    def _show_home(self) -> None:
        self.review_frame.pack_forget()
        self.home_frame.pack(fill="both", expand=True)
        self._update_counts()
        if self.dataset is None:
            self.mouse_entry.focus_set()

    def _space_pressed(self, _event: tk.Event) -> str | None:
        if self.active_session is not None and self.review_frame.winfo_ismapped():
            self._mark_current(1)
            return "break"
        return None

    def _z_pressed(self, _event: tk.Event) -> str | None:
        if self.active_session is not None and self.review_frame.winfo_ismapped():
            self._mark_current(0)
            return "break"
        return None

    def _escape_pressed(self, _event: tk.Event) -> str | None:
        if self.review_frame.winfo_ismapped():
            self._return_home()
            return "break"
        return None

    def _close_window(self) -> None:
        if self.dataset is not None and self.dataset.iscell_new is not None:
            try:
                self.dataset.save_iscell_new()
            except Exception as exc:
                close_anyway = messagebox.askyesno(
                    "Save failed",
                    f"iscell_new.npy could not be saved:\n\n{exc}\n\nClose anyway?",
                )
                if not close_anyway:
                    return
        self.destroy()


def main() -> None:
    app = CellValidatorApp()
    app.mainloop()


if __name__ == "__main__":
    main()
