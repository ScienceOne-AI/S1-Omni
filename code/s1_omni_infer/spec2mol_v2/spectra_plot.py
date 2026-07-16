from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mast_motif_mplconfig")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from spec2mol_v2.jdx import load_spectra_dir


IR_AXIS = np.linspace(500.0, 4000.0, 3501)
RAMAN_AXIS = np.linspace(500.0, 4000.0, 3501)
UV_AXIS_EV = np.linspace(1.0, 8.0, 701)
UV_AXIS_NM = 1239.841984 / UV_AXIS_EV

SPECTRUM_COLOR = "#1f2937"
PEAK_COLOR = "#ef2b2d"


def tensor_row_to_numpy(row: Any) -> np.ndarray:
    if hasattr(row, "detach"):
        row = row.detach().cpu().float().numpy()
    return np.asarray(row, dtype=np.float32)


def normalize_spectrum(row: Any) -> tuple[np.ndarray, dict[str, float]]:
    raw = tensor_row_to_numpy(row)
    clean = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    raw_min = float(np.min(clean)) if clean.size else 0.0
    raw_max = float(np.max(clean)) if clean.size else 0.0
    denom = raw_max - raw_min
    if denom > 0.0:
        normalized = (clean - raw_min) / denom
    else:
        normalized = np.zeros_like(clean, dtype=np.float32)
    return normalized.astype(np.float32), {"raw_min": raw_min, "raw_max": raw_max}


def peak_metadata(y_norm: np.ndarray, stats: dict[str, float], modality: str) -> dict[str, float | int]:
    peak_index = int(np.argmax(y_norm)) if y_norm.size else 0
    out: dict[str, float | int] = {
        "index": peak_index,
        "normalized_intensity": float(y_norm[peak_index]) if y_norm.size else 0.0,
        "raw_min": stats["raw_min"],
        "raw_max": stats["raw_max"],
    }
    if modality == "ir":
        out["wavenumber_cm1"] = float(IR_AXIS[peak_index])
    elif modality == "raman":
        out["shift_cm1"] = float(RAMAN_AXIS[peak_index])
    elif modality == "uv":
        out["wavelength_nm"] = round(float(UV_AXIS_NM[peak_index]), 4)
        out["energy_eV"] = round(float(UV_AXIS_EV[peak_index]), 4)
    else:
        raise ValueError(f"Unknown modality: {modality}")
    return out


def annotation_offset(x: float, x_min: float, x_max: float, inverted: bool) -> tuple[int, int, str]:
    span = x_max - x_min
    if inverted:
        if x < x_min + 0.22 * span:
            return -92, -28, "right"
        if x > x_max - 0.22 * span:
            return 12, -28, "left"
        return 12, 14, "left"
    if x > x_max - 0.22 * span:
        return -92, -28, "right"
    if x < x_min + 0.22 * span:
        return 12, -28, "left"
    return 12, 14, "left"


def limit_image_size(path: Path, max_image_side: int | None) -> None:
    if max_image_side is None:
        return
    with Image.open(path) as image:
        max_current_side = max(image.size)
        if max_current_side <= max_image_side:
            return
        scale = max_image_side / max_current_side
        resized_size = (
            max(1, int(round(image.width * scale))),
            max(1, int(round(image.height * scale))),
        )
        try:
            resample_filter = Image.Resampling.LANCZOS
        except AttributeError:
            resample_filter = Image.LANCZOS
        image.resize(resized_size, resample=resample_filter).save(path)


def plot_spectrum(
    y_norm: np.ndarray,
    peak: dict[str, float | int],
    idx_label: str,
    modality: str,
    path: Path,
    max_image_side: int | None,
) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 5.5), constrained_layout=True)

    if modality == "ir":
        x = IR_AXIS
        peak_x = float(peak["wavenumber_cm1"])
        title = f"{idx_label} | IR"
        xlabel = r"wavenumber / cm$^{-1}$"
        ylabel = "normalized IR intensity"
        label = f"{peak_x:.1f} cm$^{{-1}}$"
        x_min, x_max = 500.0, 4000.0
        ax.set_xlim(4000.0, 500.0)
        inverted = True
    elif modality == "raman":
        x = RAMAN_AXIS
        peak_x = float(peak["shift_cm1"])
        title = f"{idx_label} | Raman"
        xlabel = r"Raman shift / cm$^{-1}$"
        ylabel = "normalized Raman intensity"
        label = f"{peak_x:.1f} cm$^{{-1}}$"
        x_min, x_max = 500.0, 4000.0
        ax.set_xlim(4000.0, 500.0)
        inverted = True
    elif modality == "uv":
        x = UV_AXIS_NM
        peak_x = float(peak["wavelength_nm"])
        title = f"{idx_label} | UV"
        xlabel = "wavelength / nm"
        ylabel = "normalized UV intensity"
        label = f"{peak_x:.1f} nm"
        x_min, x_max = 100.0, 1300.0
        ax.set_xlim(100.0, 1300.0)
        inverted = False
    else:
        raise ValueError(f"Unknown modality: {modality}")

    peak_y = float(peak["normalized_intensity"])
    ax.plot(x, y_norm, color=SPECTRUM_COLOR, linewidth=1.35)
    ax.axvline(peak_x, color=PEAK_COLOR, linestyle="--", linewidth=1.2, alpha=0.75)
    ax.scatter([peak_x], [peak_y], color=PEAK_COLOR, s=66, zorder=5)
    dx, dy, ha = annotation_offset(peak_x, x_min, x_max, inverted=inverted)
    ax.annotate(
        label,
        xy=(peak_x, peak_y),
        xytext=(dx, dy),
        textcoords="offset points",
        color=PEAK_COLOR,
        fontsize=12,
        ha=ha,
        va="center",
        arrowprops={"arrowstyle": "->", "color": PEAK_COLOR, "linewidth": 1.1},
    )
    ax.set_ylim(-0.03, 1.05)
    ax.set_title(title, fontsize=15)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(alpha=0.25)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    limit_image_size(path, max_image_side)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _bundle_arrays(input_dir: str | Path) -> tuple[Any, dict[str, torch.Tensor]]:
    bundle = load_spectra_dir(input_dir)
    by_kind = {kind: bundle.context[idx].reshape(-1) for idx, kind in enumerate(("uv", "ir", "raman"))}
    return bundle, by_kind


def render_spectra_images(
    input_dir: str | Path,
    output_dir: str | Path,
    max_image_side: int | None = None,
    sample_id: str = "sample",
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle, arrays = _bundle_arrays(input_dir)

    files = {
        "ir": output_dir / "ir.png",
        "raman": output_dir / "raman.png",
        "uv": output_dir / "uv.png",
    }
    metadata: dict[str, Any] = {
        "sample_id": sample_id,
        "input_dir": str(Path(input_dir).resolve()),
        "normalization": "per-spectrum min-max normalization before plotting and peak marking",
        "axes": {
            "ir_cm1": {"start": 500.0, "stop": 4000.0, "points": 3501, "plot_xlim": [4000.0, 500.0]},
            "raman_cm1": {"start": 500.0, "stop": 4000.0, "points": 3501, "plot_xlim": [4000.0, 500.0]},
            "uv_eV": {"start": 1.0, "stop": 8.0, "points": 701},
            "uv_plot_x": "wavelength_nm = 1239.841984 / eV",
        },
        "present_spectra": bundle.present_spectra,
        "zero_filled_spectra": bundle.zero_filled_spectra,
        "image_files": {kind: str(path) for kind, path in files.items()},
        "source_jdx": {
            kind: str(parsed.path)
            for kind, parsed in bundle.parsed.items()
        },
        "strongest_peaks": {},
    }

    for modality in ("ir", "uv", "raman"):
        normalized, stats = normalize_spectrum(arrays[modality])
        peak = peak_metadata(normalized, stats, modality)
        metadata["strongest_peaks"][modality] = peak
        plot_spectrum(normalized, peak, sample_id, modality, files[modality], max_image_side)

    metadata_path = output_dir / "metadata.json"
    write_json(metadata_path, metadata)
    metadata["metadata_path"] = str(metadata_path)
    return metadata
