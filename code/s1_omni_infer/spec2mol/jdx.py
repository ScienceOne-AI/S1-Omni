from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import torch


SPECTRA_LENGTHS = {"uv": 701, "ir": 3501, "raman": 3501}
DATA_TYPE_MAP = {
    "INFRARED SPECTRUM": "ir",
    "RAMAN SPECTRUM": "raman",
    "UV/VIS SPECTRUM": "uv",
    "ULTRAVIOLET SPECTRUM": "uv",
    "UV SPECTRUM": "uv",
}


@dataclass(frozen=True)
class ParsedJdx:
    path: Path
    kind: str
    data_type: str
    x: torch.Tensor
    y: torch.Tensor


@dataclass(frozen=True)
class SpectraBundle:
    context: list[torch.Tensor]
    parsed: dict[str, ParsedJdx]
    present_spectra: list[str]
    zero_filled_spectra: list[str]


def _header_value(line: str, key: str) -> str | None:
    prefix = f"##{key.upper()}="
    if line.upper().startswith(prefix):
        return line[len(prefix) :].strip()
    return None


def classify_data_type(data_type: str) -> str:
    normalized = re.sub(r"\s+", " ", data_type.strip().upper())
    if normalized in DATA_TYPE_MAP:
        return DATA_TYPE_MAP[normalized]
    raise ValueError(f"Unsupported JDX DATA TYPE: {data_type!r}")


def _parse_xy_line(line: str) -> tuple[float, float] | None:
    if not line or line.startswith("##"):
        return None
    parts = line.replace(",", " ").split()
    if len(parts) < 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


def parse_jdx(path: str | Path) -> ParsedJdx:
    path = Path(path)
    data_type: str | None = None
    in_xy = False
    xs: list[float] = []
    ys: list[float] = []

    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        value = _header_value(line, "DATA TYPE")
        if value is not None:
            data_type = value
            continue
        if line.upper().startswith("##XYDATA="):
            in_xy = True
            continue
        if line.upper().startswith("##END"):
            break
        if in_xy:
            xy = _parse_xy_line(line)
            if xy is not None:
                x_value, y_value = xy
                xs.append(x_value)
                ys.append(y_value)

    if data_type is None:
        raise ValueError(f"{path} is missing ##DATA TYPE")
    if not ys:
        raise ValueError(f"{path} does not contain XYDATA points")

    kind = classify_data_type(data_type)
    return ParsedJdx(
        path=path,
        kind=kind,
        data_type=data_type,
        x=torch.tensor(xs, dtype=torch.float32),
        y=torch.tensor(ys, dtype=torch.float32),
    )


def _fit_length(y: torch.Tensor, length: int, kind: str, path: Path) -> torch.Tensor:
    y = y.detach().float().reshape(-1)
    if y.numel() == length:
        return y
    if y.numel() > length:
        return y[:length]
    raise ValueError(f"{path} classified as {kind} has {y.numel()} points, expected at least {length}")


def load_spectra_dir(input_dir: str | Path, lengths: dict[str, int] | None = None) -> SpectraBundle:
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    lengths = lengths or SPECTRA_LENGTHS

    parsed_by_kind: dict[str, ParsedJdx] = {}
    for path in sorted(input_dir.glob("*.jdx")):
        parsed = parse_jdx(path)
        if parsed.kind in parsed_by_kind:
            raise ValueError(
                f"Duplicate {parsed.kind} spectra: {parsed_by_kind[parsed.kind].path} and {parsed.path}"
            )
        parsed_by_kind[parsed.kind] = parsed
    if not parsed_by_kind:
        raise ValueError(f"No .jdx files found in {input_dir}")

    tensors: dict[str, torch.Tensor] = {}
    present = []
    zero_filled = []
    for kind in ("uv", "ir", "raman"):
        length = int(lengths[kind])
        if kind in parsed_by_kind:
            tensors[kind] = _fit_length(parsed_by_kind[kind].y, length, kind, parsed_by_kind[kind].path)
            present.append(kind)
        else:
            tensors[kind] = torch.zeros(length, dtype=torch.float32)
            zero_filled.append(kind)

    context = [tensors[kind].view(1, 1, -1) for kind in ("uv", "ir", "raman")]
    return SpectraBundle(
        context=context,
        parsed=parsed_by_kind,
        present_spectra=present,
        zero_filled_spectra=zero_filled,
    )


def bundle_to_jsonable(bundle: SpectraBundle) -> dict[str, Any]:
    return {
        "present_spectra": bundle.present_spectra,
        "zero_filled_spectra": bundle.zero_filled_spectra,
        "files": {
            kind: {"path": str(parsed.path), "data_type": parsed.data_type, "n_points": int(parsed.y.numel())}
            for kind, parsed in bundle.parsed.items()
        },
    }
