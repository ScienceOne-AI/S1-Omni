#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

# Started from https://github.com/jwohlwend/boltz, 
# licensed under MIT License, Copyright (c) 2024 Jeremy Wohlwend, Gabriele Corso, Saro Passaro. 

import click
import pickle
import json
import urllib.request
from pathlib import Path
from tqdm import tqdm
from dataclasses import asdict

from s1_omni.protein_folding.boltz_data_pipeline.types import Manifest, Record


CCD_URL = "https://huggingface.co/boltz-community/boltz-1/resolve/main/ccd.pkl"
MODEL_URL = (
    "https://huggingface.co/boltz-community/boltz-1/resolve/main/boltz1_conf.ckpt"
)


from collections.abc import Mapping
from pathlib import Path

from Bio import SeqIO
from rdkit.Chem.rdchem import Mol

from s1_omni.protein_folding.boltz_data_pipeline.parse.yaml import parse_boltz_schema
from s1_omni.protein_folding.boltz_data_pipeline.types import Target


FASTA_SUFFIXES = {".fa", ".fas", ".fasta", ".faa", ".fna", ".ffn", ".fsa"}


def _looks_like_fasta(path: Path) -> bool:
    """Heuristic FASTA sniffing for extensionless files."""
    if not path.is_file():
        return False
    try:
        with path.open("r") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                return s.startswith(">")
    except Exception:
        return False
    return False


def _is_fasta_path(path: Path) -> bool:
    return path.suffix.lower() in FASTA_SUFFIXES or (path.suffix == "" and _looks_like_fasta(path))


def parse_fasta(path: Path, ccd: Mapping[str, Mol]) -> Target:  # noqa: C901
    """Parse a fasta file.

    The name of the fasta file is used as the name of this job.
    We rely on the fasta record id to determine the entity type.

    > CHAIN_ID|Description
    SEQUENCE
    > CHAIN_ID|Description
    ...

    Where CHAIN_ID is the chain identifier, which should be unique.

    Parameters
    ----------
    fasta_file : Path
        Path to the fasta file.
    ccd : Dict
        Dictionary of CCD components.

    Returns
    -------
    Target
        The parsed target.

    """
    # Read fasta file
    # 这两行代码的作用是打开指定的fasta文件，并用Bio.SeqIO库以FASTA格式解析所有序列记录，然后将这些记录转为列表。
    # path.open("r")：以只读模式打开给定的fasta文件路径
    # SeqIO.parse(f, "fasta")：依次读取文件中的每条fasta序列，生成一个可迭代的记录对象
    # list(...)：将所有序列记录装入列表，方便后续遍历处理
    with path.open("r") as f:
        records = list(SeqIO.parse(f, "fasta"))
    sequences = []
    for seq_record in records:
        seq = str(seq_record.seq)
        molecule = {
            "protein": {
                "id": "A", # Set a default chain ID
                "sequence": seq,
                "modifications": [],
                "msa": None,
            },
        }
        sequences.append(molecule)

    data = {
        "sequences": sequences,
        "bonds": [],
        "version": 1,
    }

    name = path.stem
    return parse_boltz_schema(name, data, ccd)


def check_fasta_inputs(data: Path) -> list[Path]:
    click.echo("Checking input data.")

    # Check if data is a directory
    if data.is_dir():
        candidates: list[Path] = list(data.glob("*"))
        data = [p for p in candidates if _is_fasta_path(p)]
    else:
        data = [data]

    print(f"Found {len(data)} examples to process.")
    return data


def download_fasta_utilities(cache: Path) -> None:
    """Download all the required data.

    Parameters
    ----------
    cache : Path
        The cache directory.

    """
    # Download CCD
    ccd = cache / "ccd.pkl"
    if not ccd.exists():
        click.echo(
            f"Downloading the CCD dictionary to {ccd}. You may "
            "change the cache directory with the --cache flag."
        )
        urllib.request.urlretrieve(CCD_URL, str(ccd))

    # Download model
    model = cache / "boltz1_conf.ckpt"
    if not model.exists():
        click.echo(
            f"Downloading the model weights to {model}. You may "
            "change the cache directory with the --cache flag."
        )
        urllib.request.urlretrieve(MODEL_URL, str(model))


def process_fastas(
    data: list[Path],
    out_dir: Path,
    ccd_path: Path,
) -> None:
    """Process the input data and output directory.

    Parameters
    ----------
    data : list[Path]
        The input data.
    out_dir : Path
        The output directory.
    ccd_path : Path
        The path to the CCD dictionary.

    Returns
    -------
    BoltzProcessedInput
        The processed input data.

    """
    click.echo("Processing input data.")

    struct_dir = out_dir / "structures"
    record_dir = out_dir / "records"
    out_dir.mkdir(parents=True, exist_ok=True)
    struct_dir.mkdir(parents=True, exist_ok=True)
    record_dir.mkdir(parents=True, exist_ok=True)

    # Load CCD
    with ccd_path.open("rb") as file:
        ccd = pickle.load(file)  # noqa: S301

    # Parse input data
    records: list[Record] = []
    print('fasta文件数量：', len(data))
    for path in tqdm(data):
        try:
            # Parse data
            if _is_fasta_path(path):
                target = parse_fasta(path, ccd)
            elif path.is_dir():
                msg = f"Found directory {path} instead of a FASTA file, skipping."
                raise RuntimeError(msg)
            else:
                msg = (
                    f"Unable to parse filetype {path.suffix}, "
                    "please provide a FASTA file (e.g. .fasta/.fa)."
                )
                raise RuntimeError(msg)

            for chain in target.record.chains:
                chain.msa_id = -1

            # Keep record
            records.append(target.record)
            #print(f"[FASTA] id={target.record.id} chains={len(target.record.chains)} lens={[c.num_residues for c in target.record.chains]}")

            # Dump structure
            struct_path = struct_dir / f"{target.record.id}.npz"
            target.structure.dump(struct_path)

            # save record
            record_path = record_dir / f"{target.record.id}.json"
            record_path.parent.mkdir(parents=True, exist_ok=True)
            with record_path.open("w") as f:
                json.dump(asdict(target.record), f)

        except Exception as e:
            if len(data) > 1:
                print(f"Failed to process {path}. Skipping. Error: {e}.")
            else:
                raise e

    # Dump manifest
    manifest = Manifest(records)
    manifest.dump(out_dir / "manifest.json")
