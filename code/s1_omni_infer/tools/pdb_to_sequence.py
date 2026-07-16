#!/usr/bin/env python3
"""
Convert PDB file(s) + question text into JSONL format for downstream ML tasks.

Each output line: {"id": "<pdb_stem>_<chain>", "input": "<question>\nSequence: <seq>"}

Usage:
    python pdb_to_sequence.py PDB/*.pdb -q question.txt -o input.jsonl
    python pdb_to_sequence.py 5DY0.pdb -q question.txt
"""

import argparse
import json
from pathlib import Path
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════════
# PDB / mmCIF sequence extraction (standard 20 amino acids only)
# ═══════════════════════════════════════════════════════════════════════════════

RESN_TO_ONE: dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

NUC_RESNS: set[str] = {"DA", "DC", "DG", "DT", "A", "C", "G", "T", "U", "DU", "DI"}


def _resolve_one_letter(resn: str) -> Optional[str]:
    """Map a 3-letter residue name to single-letter code.

    Only the standard 20 amino acids are returned; nucleic acids and non-standard
    residues (e.g. MSE, HYP, SEC, PYL) are skipped so the output is suitable for
    downstream tools like ESM-2 that expect canonical amino acid sequences.
    """
    if resn in NUC_RESNS:
        return None
    return RESN_TO_ONE.get(resn)


def _sort_and_number(chains: dict[str, list[dict]]) -> dict[str, list[dict]]:
    for c in chains:
        chains[c].sort(key=lambda r: r["resi"])
        for i, res in enumerate(chains[c]):
            res["pos"] = i + 1
    return chains


def _extract_from_pdb(text: str) -> dict[str, list[dict]]:
    chains: dict[str, list[dict]] = {}
    seen: set[tuple[str, int]] = set()

    for line in text.splitlines():
        if len(line) < 54:
            continue
        if line[0:6].strip() not in ("ATOM", "HETATM"):
            continue
        if line[12:16].strip() != "CA":
            continue

        one = _resolve_one_letter(line[17:20].strip())
        if one is None:
            continue

        chain_id = line[21] if len(line) > 21 else " "
        try:
            resi = int(line[22:26].strip())
        except ValueError:
            continue

        key = (chain_id, resi)
        if key not in seen:
            seen.add(key)
            chains.setdefault(chain_id, []).append({"resi": resi, "one": one})

    return _sort_and_number(chains)


def _extract_from_mmcif(text: str) -> dict[str, list[dict]]:
    chains: dict[str, list[dict]] = {}
    seen: dict[str, set[tuple[str, int]]] = {}
    col: dict[str, int] = {}
    in_atom_site = False

    for line in text.splitlines():
        if line.startswith("loop_"):
            continue
        if line.startswith("_atom_site."):
            in_atom_site = True
            col[line.strip().split(".")[-1]] = len(col)
            continue
        if in_atom_site and line.strip() and not line.startswith("_") and not line[0].isdigit() and not line.startswith(" "):
            in_atom_site = False
            col.clear()
            continue
        if not in_atom_site or not line.strip() or line.startswith("_"):
            continue

        cols = line.strip().split()
        needed = max(col.values()) + 1 if col else 0
        if len(cols) < needed:
            continue

        group = cols[col["group_PDB"]] if "group_PDB" in col else "ATOM"
        if group not in ("ATOM", "HETATM"):
            continue
        label_atom = cols[col.get("label_atom_id", -1)] if "label_atom_id" in col else ""
        if label_atom != "CA":
            continue

        one = _resolve_one_letter(cols[col.get("label_comp_id", -1)] if "label_comp_id" in col else "")
        if one is None:
            continue

        chain_id = cols[col.get("label_asym_id", -1)] if "label_asym_id" in col else " "
        try:
            resi = int(cols[col.get("label_seq_id", -1)])
        except (ValueError, IndexError):
            continue

        sv = seen.setdefault(chain_id, set())
        key = (chain_id, resi)
        if key not in sv:
            sv.add(key)
            chains.setdefault(chain_id, []).append({"resi": resi, "one": one})

    return _sort_and_number(chains)


def extract_sequences(path: str | Path) -> dict[str, list[dict]]:
    """Extract per-chain sequences (standard 20 AAs) from a PDB or mmCIF file.

    Returns:
        {chain_id: [{"pos": 1, "resi": 42, "one": "G"}, ...], ...}
    """
    path = Path(path)
    text = path.read_text(errors="replace")
    if path.suffix.lower() in {".cif", ".mmcif"}:
        return _extract_from_mmcif(text)
    return _extract_from_pdb(text)


# ═══════════════════════════════════════════════════════════════════════════════
# JSONL builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_jsonl(pdb_paths: list[Path], question: str) -> list[dict]:
    records: list[dict] = []
    for pdb_path in pdb_paths:
        pdb_id = pdb_path.stem
        chains = extract_sequences(str(pdb_path))
        if not chains:
            print(f"Warning: no standard amino acid chains found in {pdb_path.name}")
            continue
        for chain_id in sorted(chains.keys()):
            seq = "".join(r["one"] for r in chains[chain_id])
            records.append({
                "id": f"{pdb_id}_{chain_id}",
                "input": f"{question}\nSequence: <PROT>{seq}</PROT>",
            })
    return records


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert PDB files + question text to JSONL.",
    )
    parser.add_argument("pdb", type=str, nargs="+",
                        help="PDB / mmCIF file(s).")
    parser.add_argument("-q", "--question", type=str, required=True,
                        help="Path to a text file containing the question.")
    parser.add_argument("-o", "--output", type=str, default="input.jsonl",
                        help="Output JSONL path (default: input.jsonl).")

    args = parser.parse_args()

    question = Path(args.question).read_text(errors="replace").strip()

    pdb_paths: list[Path] = []
    for pattern in args.pdb:
        p = Path(pattern)
        if p.is_file():
            pdb_paths.append(p)
        else:
            matched = sorted(Path().glob(pattern))
            if not matched:
                print(f"Warning: no files matched pattern '{pattern}'")
            pdb_paths.extend(f for f in matched if f.is_file())

    if not pdb_paths:
        print("Error: no PDB files found.")
        return

    records = build_jsonl(pdb_paths, question)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Written {len(records)} chains to {output_path}")
    for r in records:
        seq_len = len(r["input"].split("Sequence: ")[1])
        print(f"  {r['id']}: {seq_len} aa")


if __name__ == "__main__":
    main()
