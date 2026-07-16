#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import zipfile
import torch
from torch import Tensor
import json
import numpy as np
from typing import Optional
from dataclasses import dataclass
from pathlib import Path

from s1_omni.protein_folding.boltz_data_pipeline.feature.pad import pad_to_max
from s1_omni.protein_folding.boltz_data_pipeline.crop.cropper import Cropper
from s1_omni.protein_folding.boltz_data_pipeline.tokenize.boltz_protein import BoltzTokenizer
from s1_omni.protein_folding.boltz_data_pipeline.feature.featurizer import BoltzFeaturizer
from s1_omni.protein_folding.processor.protein_processor import ProteinDataProcessor
from s1_omni.protein_folding.boltz_data_pipeline.types import Connection, Input, Manifest, Record, Structure
from s1_omni.protein_folding.boltz_data_pipeline import const


restype_1to3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}
restype_3to1 = {v: k for k, v in restype_1to3.items()}
restype_3to1["UNK"] = "X"
restype_3to1["-"] = "X"
restype_3to1["<pad>"] = "X"


@dataclass
class Dataset:
    """Data holder."""
    tokenized_dir: str
    target_dir: Path
    esm_dir: str
    connectivity_dir: str
    cheap_dir: str
    manifest: Manifest
    cropper: Cropper
    tokenizer: BoltzTokenizer
    featurizer: BoltzFeaturizer
    cluster: Optional[str] = None


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    data_name: str
    tokenized_dir: str
    target_dir: Path
    cropper: Cropper
    filters: Optional[list] = None
    # split: Optional[str] = None
    manifest_path: Optional[str] = None
    esm_dir: Optional[str] = None
    connectivity_dir: Optional[str] = None
    cheap_dir: Optional[str] = None
    record_list: Optional[str] = None
    cluster: Optional[str] = None


def load_input(record: Record, target_dir: Path) -> Input:
    """Load the given input data.

    Parameters
    ----------
    record : Record
        The record to load.
    target_dir : Path
        The path to the data directory.

    Returns
    -------
    Input
        The loaded input.

    """
    # Load the structure
    structure = np.load(target_dir / "structures" / f"{record.id}.npz")
    structure = Structure(
        atoms=structure["atoms"],
        bonds=structure["bonds"],
        residues=structure["residues"],
        chains=structure["chains"],
        connections=structure["connections"].astype(Connection),
        interfaces=structure["interfaces"],
        mask=structure["mask"],
    )
    return Input(structure, {})


def collate(data: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Collate the data.

    Parameters
    ----------
    data : list[dict[str, Tensor]]
        The data to collate.

    Returns
    -------
    dict[str, Tensor]
        The collated data.

    """
    # Get the keys
    keys = data[0].keys()

    # Collate the data
    collated = {}
    for key in keys:
        values = [d[key] for d in data]

        if key not in [
            "all_coords",
            "all_resolved_mask",
            "crop_to_all_atom_map",
            "chain_symmetries",
            "amino_acids_symmetries",
            "ligand_symmetries",
            "record",
            "aa_seq",
            "text",
        ]:
            # Check if all have the same shape
            shape = values[0].shape
            if not all(v.shape == shape for v in values):
                values, _ = pad_to_max(values, 0)
            else:
                values = torch.stack(values, dim=0)

        # Stack the values
        collated[key] = values

    return collated


def extract_sequence_from_tokens(tokenized):
    """
    从tokenized数据中提取蛋白质序列，跳过非蛋白质残基（DNA/RNA）
    
    对于包含DNA/RNA的样本，只提取蛋白质部分
    """
    seq = []
    sequence = []
    current_entity = 0
    for i, t in enumerate(tokenized.tokens):
        entity = t[7]
        if entity != current_entity:
            if seq:  # 只有当seq不为空时才添加
                sequence.append("".join(seq))
            seq = []
            current_entity = entity

        res_type = t[4]
        try:
            res_type_name = const.tokens[res_type]
        except (IndexError, KeyError):
            # 跳过无效的残基类型
            continue
        
        # 只处理蛋白质残基，跳过DNA/RNA残基
        if res_type_name not in restype_3to1:
            # 这是DNA/RNA残基或其他非蛋白质残基，跳过
            print(f"Non-protein residue {res_type_name}. Skipping.")
            continue
        
        res_name = restype_3to1[res_type_name]
        seq.append(res_name)
        
        if i == len(tokenized.tokens) - 1:
            if seq:  # 只有当seq不为空时才添加
                sequence.append("".join(seq))
            seq = []
            current_entity = entity
    
    # 如果没有任何蛋白质序列，返回空字符串
    if not sequence:
        return ""
    
    return ":".join(sequence)


def process_one_inference_structure(
    structure_path,
    record_path,
    tokenizer: BoltzTokenizer,
    featurizer: BoltzFeaturizer,
    processor: ProteinDataProcessor,
    esm_model=None,
    esm_dict=None,
    af2_to_esm=None,
    *,
    use_dssp: bool = False,
    dssp_dir: Optional[str | Path] = None,
    fasta_path: Optional[str | Path] = None,
) -> Optional[tuple]:
    """FASTA/结构推理单样本处理；结构损坏、DSSP 缺失或对齐失败时返回 None（调用方应跳过）。"""
    try:
        structure: Structure = Structure.load(structure_path)
    except (EOFError, OSError, zipfile.BadZipFile, ValueError) as e:
        print(
            f"结构 npz 损坏或不完整，跳过: {structure_path}（{type(e).__name__}: {e}）。"
            f"若上次磁盘满导致，请删除 output_dir/structures 后重新推理。"
        )
        return None
    input_data = Input(structure, {})
    record = json.load(open(record_path))
    record_obj = Record(**record)

    tokenized = tokenizer.tokenize(input_data)
    max_num_tokens = len(tokenized.tokens)

    ss8_feature = None
    ss8_feature_key = None
    if use_dssp:
        if dssp_dir is None:
            raise ValueError("已开启 use_dssp，但未指定 dssp_dir。")
        from utils.dssp_align import load_ssp_feature_for_inference, read_fasta_header_id

        fp = Path(fasta_path) if fasta_path is not None else None
        hdr = read_fasta_header_id(fp)
        ss8_feature, matched_id, tried, ss8_feature_key = load_ssp_feature_for_inference(
            dssp_dir, record_obj.id, fp
        )
        if ss8_feature is None:
            print(
                f"未找到二级结构特征，跳过: record_id={record_obj.id}, fasta_header={hdr}, "
                f"已尝试 id={tried}"
            )
            return None
        if matched_id != record_obj.id:
            print(
                f"二级结构特征命中 id={matched_id}, key={ss8_feature_key}（record_id={record_obj.id}"
                f"{f', fasta头={hdr}' if hdr else ''}）"
            )

    sequence = extract_sequence_from_tokens(tokenized)
    if use_dssp and (not sequence or not sequence.strip()):
        print(f"无蛋白质序列，跳过: {record_obj.id}")
        return None

    features = featurizer.process(tokenized)

    if use_dssp:
        from utils.dssp_align import attach_dssp_onehot_to_features, attach_ssp_prob_to_features

        if ss8_feature_key == "ssp_prob":
            attached = attach_ssp_prob_to_features(tokenized, features, ss8_feature)
        else:
            attached = attach_dssp_onehot_to_features(tokenized, features, ss8_feature)
        if attached is None:
            print(
                f"二级结构特征与 token 对齐失败，跳过: {record_obj.id}, "
                f"key={ss8_feature_key}, n_token={len(tokenized.tokens)}, ss8_L={ss8_feature.shape[0]}"
            )
            return None

    features["aa_seq"] = sequence
    features["record"] = record
    features["num_repeats"] = torch.tensor(1)
    features["max_num_tokens"] = torch.tensor(max_num_tokens, dtype=torch.long)
    protein_seq_length = len(sequence.replace(":", "")) if sequence else 0
    features["cropped_num_tokens"] = torch.tensor(protein_seq_length, dtype=torch.long)

    batch = collate([features])
    batch = processor.preprocess_inference(
        batch,
        esm_model=esm_model,
        esm_dict=esm_dict,
        af2_to_esm=af2_to_esm,
    )

    return batch, structure, record_obj
