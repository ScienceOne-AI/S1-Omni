#!/usr/bin/env python3
"""Validate spec2mol_v2 HF-inline E2E spectra-case outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

CASES = {
    "only_ir": ["ir"],
    "only_raman": ["raman"],
    "only_uv": ["uv"],
    "ir_raman": ["ir", "raman"],
    "ir_uv": ["ir", "uv"],
    "raman_uv": ["raman", "uv"],
    "ir_raman_uv": ["ir", "raman", "uv"],
}
EXPECTED_PROMPT = "请你详细分析这张谱图，预测可能的分子结构"


def load_stdout_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8").strip()
    return json.loads(text)


def check_case(output_root: Path, case: str, expected_present: list[str], allow_generation_failures: bool) -> dict[str, Any]:
    out_dir = output_root / "results" / case
    stdout_path = output_root / "results" / f"{case}.stdout.json"
    stderr_path = output_root / "results" / f"{case}.stderr.log"
    rc_path = output_root / "results" / f"{case}.returncode"
    sdf_path = out_dir / "sample.sdf"
    errors: list[str] = []
    result: dict[str, Any] = {
        "case": case,
        "output_dir": str(out_dir),
        "stdout_json": str(stdout_path),
        "stderr_log": str(stderr_path),
        "returncode_file": str(rc_path),
        "expected_present": expected_present,
        "validation_errors": errors,
    }

    rc = None
    if rc_path.is_file():
        try:
            rc = int(rc_path.read_text(encoding="utf-8").strip())
        except ValueError:
            errors.append(f"invalid returncode file: {rc_path}")
    else:
        errors.append(f"missing returncode file: {rc_path}")

    payload: dict[str, Any] = {}
    if stdout_path.is_file():
        try:
            payload = load_stdout_json(stdout_path)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"stdout is not valid JSON: {exc!r}")
    else:
        errors.append(f"missing stdout JSON: {stdout_path}")

    keys = set(payload)
    if keys != {"question", "generated_text", "final_text"}:
        errors.append(f"stdout JSON keys mismatch: {sorted(keys)}")
    if payload.get("question") != EXPECTED_PROMPT:
        errors.append(f"question mismatch: {payload.get('question')!r}")
    for key in ("generated_text", "final_text"):
        if key in payload and not isinstance(payload.get(key), str):
            errors.append(f"{key} is not a string")

    files = sorted(path.name for path in out_dir.iterdir()) if out_dir.is_dir() else []
    extra_files = [name for name in files if name != "sample.sdf"]
    if extra_files:
        errors.append(f"output_dir contains non-SDF files: {extra_files}")

    sdf_ok = sdf_path.is_file() and sdf_path.stat().st_size > 0
    final_text = payload.get("final_text") or ""
    if rc == 0 and "<spectra_st>" in (payload.get("generated_text") or ""):
        if not sdf_ok:
            errors.append("rc=0 with <spectra_st> but sample.sdf missing/empty")
        if str(sdf_path.resolve()) not in final_text:
            errors.append("final_text missing absolute sample.sdf path")
    if sdf_ok and str(sdf_path.resolve()) not in final_text:
        errors.append("sample.sdf exists but final_text does not contain its absolute path")
    if not sdf_ok and str(sdf_path.resolve()) in final_text:
        errors.append("final_text contains nonexistent sample.sdf path")
    if rc not in (0, None) and not allow_generation_failures:
        errors.append(f"inference returncode is {rc}")

    result.update(
        {
            "validation_ok": not errors,
            "returncode": rc,
            "sdf_written": sdf_ok,
            "question": payload.get("question"),
            "generated_has_spectra": "<spectra_st>" in (payload.get("generated_text") or ""),
            "final_has_sdf_path": str(sdf_path.resolve()) in final_text,
        }
    )
    return result


def write_summary_md(path: Path, cases: list[dict[str, Any]]) -> None:
    lines = [
        "# spec2mol_v2 HF-inline E2E Spectra Cases",
        "",
        "| case | validation | rc | sdf | generated `<spectra_st>` | final SDF path |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for item in cases:
        lines.append(
            "| {case} | {validation} | {rc} | {sdf} | {spectra} | {path} |".format(
                case=item["case"],
                validation="PASS" if item.get("validation_ok") else "FAIL",
                rc=item.get("returncode"),
                sdf=item.get("sdf_written"),
                spectra=item.get("generated_has_spectra"),
                path=item.get("final_has_sdf_path"),
            )
        )
    lines.append("")
    for item in cases:
        if item.get("validation_errors"):
            lines.append(f"## {item['case']} errors")
            lines.extend(f"- {err}" for err in item["validation_errors"])
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--allow_generation_failures", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root).expanduser().resolve()
    cases = [
        check_case(output_root, case, present, args.allow_generation_failures)
        for case, present in CASES.items()
    ]
    summary = {
        "output_root": str(output_root),
        "num_cases": len(cases),
        "num_validation_ok": sum(1 for c in cases if c.get("validation_ok")),
        "num_sdf_written": sum(1 for c in cases if c.get("sdf_written")),
        "cases": cases,
    }
    summary["validation_ok"] = summary["num_validation_ok"] == len(cases)
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_summary_md(output_root / "summary.md", cases)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["validation_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
