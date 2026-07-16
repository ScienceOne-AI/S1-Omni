import json
import logging
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

import transformers

from . import data_list

IGNORE_INDEX = -100
LINEAR_CLA_TOKEN = "<linear_cla>"
LINEAR_PRE_TOKEN = "<linear_pre>"
ROUTE_CLS = 0
ROUTE_REG = 1
local_rank = None
PROTEIN_TAG_RE = re.compile(r"<PROT>(.*?)</PROT>", flags=re.DOTALL)


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def load_pooled_hidden_cache(path: str) -> torch.Tensor:
    try:
        cache = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        cache = torch.load(path, map_location="cpu")

    if isinstance(cache, torch.Tensor):
        pooled_hidden = cache
    elif isinstance(cache, dict) and "pooled_hidden" in cache:
        pooled_hidden = cache["pooled_hidden"]
    else:
        raise ValueError(f"expected tensor or dict with 'pooled_hidden' in cache file: {path}")

    if not isinstance(pooled_hidden, torch.Tensor) or pooled_hidden.ndim != 2:
        raise ValueError(
            f"pooled hidden cache must be a 2D tensor, got {type(pooled_hidden)} "
            f"with shape {getattr(pooled_hidden, 'shape', None)}"
        )
    return pooled_hidden.contiguous()


def _make_abs_paths(base: Path, files: str) -> str:
    return f"{(base / files).resolve()}"


def update_processor_pixels(processor, data_args):
    return processor


def _extract_route_from_text(text: str) -> Optional[int]:
    if text is None:
        return None
    normalized_text = text.lower()
    if LINEAR_CLA_TOKEN in normalized_text:
        return ROUTE_CLS
    if LINEAR_PRE_TOKEN in normalized_text:
        return ROUTE_REG
    return None


def _infer_route_and_label(label: Any, assistant_text: str):
    route_from_text = _extract_route_from_text(assistant_text)

    if isinstance(label, bool):
        label = int(label)

    if route_from_text == ROUTE_CLS:
        if float(label) not in (0.0, 1.0):
            raise ValueError("assistant classification route token conflicts with non-binary label")
        return "classification", ROUTE_CLS, torch.tensor(float(int(label)), dtype=torch.float32)

    if route_from_text == ROUTE_REG:
        return "regression", ROUTE_REG, torch.tensor(float(label), dtype=torch.float32)

    if isinstance(label, int) and label in (0, 1):
        return "classification", ROUTE_CLS, torch.tensor(float(label), dtype=torch.float32)

    if isinstance(label, float) and float(label).is_integer() and int(label) in (0, 1):
        return "classification", ROUTE_CLS, torch.tensor(float(int(label)), dtype=torch.float32)

    return "regression", ROUTE_REG, torch.tensor(float(label), dtype=torch.float32)


def _compute_regression_label_stats(data: Sequence[Dict[str, Any]]) -> tuple[float, float, int]:
    reg_labels = []
    for item in data:
        messages = _normalize_dialogue(item)
        assistant_text = _extract_assistant_text(messages)
        _, route_id, label = _infer_route_and_label(item["label"], assistant_text)
        if route_id == ROUTE_REG:
            reg_labels.append(float(label.item()))

    if not reg_labels:
        return 0.0, 1.0, 0

    labels = np.asarray(reg_labels, dtype=np.float64)
    mean = float(labels.mean())
    std = float(labels.std())
    if std < 1e-6:
        std = 1.0
    return mean, std, int(labels.size)


def _compute_head_label_counts(data: Sequence[Dict[str, Any]]) -> tuple[list[int], list[int]]:
    route_counts = [0, 0]
    class_counts = [0, 0]
    for item in data:
        messages = _normalize_dialogue(item)
        assistant_text = _extract_assistant_text(messages)
        _, route_id, label = _infer_route_and_label(item["label"], assistant_text)
        route_counts[route_id] += 1
        if route_id == ROUTE_CLS:
            class_counts[int(label.item())] += 1
    return route_counts, class_counts


def _extract_assistant_text(messages: List[Dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(item.get("text", "") for item in content if item.get("type") == "text")
            return str(content)
    raise ValueError("missing assistant message")


def _normalize_dialogue(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "messages" in item:
        raw_messages = item["messages"]
    elif "conversations" in item:
        role_map = {
            "human": "user",
            "user": "user",
            "gpt": "assistant",
            "assistant": "assistant",
            "system": "system",
        }
        raw_messages = [
            {
                "role": role_map.get(msg.get("from"), msg.get("from")),
                "content": msg.get("value", ""),
            }
            for msg in item["conversations"]
        ]
    else:
        raise KeyError("expected either 'messages' or 'conversations' in sample")

    if not raw_messages:
        raise ValueError("empty dialogue in sample")
    return raw_messages


def _build_messages(item: Dict[str, Any], base_path: Path):
    images = item.get("image") or []
    if isinstance(images, str):
        images = [images]
    videos = item.get("video") or []
    if isinstance(videos, str):
        videos = [videos]

    image_pool = [{"type": "image", "image": _make_abs_paths(base_path, img)} for img in images]
    video_pool = [{"type": "video", "video": _make_abs_paths(base_path, vid)} for vid in videos]

    messages = []
    for msg in _normalize_dialogue(item):
        role = msg["role"]
        content = msg["content"]
        if role == "assistant":
            if isinstance(content, str):
                messages.append({"role": role, "content": [{"type": "text", "text": content}]})
            else:
                messages.append({"role": role, "content": content})
            continue

        if isinstance(content, list):
            messages.append({"role": role, "content": content})
            continue

        parts = re.split(r"(<image>|<video>)", content)
        normalized = []
        for seg in parts:
            if seg == "<image>":
                if not image_pool:
                    raise ValueError("Number of <image> placeholders exceeds the number of provided images")
                normalized.append(image_pool.pop(0))
            elif seg == "<video>":
                if not video_pool:
                    raise ValueError("Number of <video> placeholders exceeds the number of provided videos")
                normalized.append(video_pool.pop(0))
            elif seg.strip():
                normalized.append({"type": "text", "text": seg.strip()})
        messages.append({"role": role, "content": normalized})

    if image_pool:
        raise ValueError(f"{len(image_pool)} image(s) remain unused (not consumed by placeholders)")
    if video_pool:
        raise ValueError(f"{len(video_pool)} video(s) remain unused (not consumed by placeholders)")

    return messages


def _load_data_dict(data_args, annotation_path: Optional[str] = None):
    list_data_dict = []
    annotation_path = annotation_path or data_args.annotation_path
    if annotation_path:
        file_format = str(annotation_path).split(".")[-1]
        if file_format == "jsonl":
            annotations = read_jsonl(annotation_path)
        else:
            annotations = json.load(open(annotation_path, "r"))
        for ann in annotations:
            ann["data_path"] = data_args.data_path or ""
            list_data_dict.append(ann)
        return list_data_dict

    dataset = data_args.dataset_use.split(",")
    dataset_list = data_list(dataset)
    rank0_print(f"Loading datasets: {dataset_list}")
    for data in dataset_list:
        file_format = data["annotation_path"].split(".")[-1]
        if file_format == "jsonl":
            annotations = read_jsonl(data["annotation_path"])
        else:
            annotations = json.load(open(data["annotation_path"], "r"))
        sampling_rate = data.get("sampling_rate", 1.0)
        if sampling_rate < 1.0:
            annotations = random.sample(annotations, int(len(annotations) * sampling_rate))
        for ann in annotations:
            ann["data_path"] = data["data_path"]
        list_data_dict += annotations
    return list_data_dict


def _extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("type", "text") == "text"
        )
    return str(content)


def _build_protein_messages(item: Dict[str, Any]):
    if "input" in item:
        text = item["input"]
    elif "text" in item:
        text = item["text"]
    elif "question" in item:
        text = item["question"]
    elif "messages" in item or "conversations" in item:
        messages = []
        for msg in _normalize_dialogue(item):
            role = msg["role"]
            if role == "assistant":
                continue
            text = _extract_text_content(msg.get("content", ""))
            if text.strip():
                messages.append({"role": role, "content": [{"type": "text", "text": text}]})
        if not messages:
            raise ValueError("protein sample has no non-assistant text message")
        return messages
    else:
        raise KeyError("expected 'input', 'text', 'question', 'messages', or 'conversations' in protein sample")

    return [{"role": "user", "content": [{"type": "text", "text": str(text)}]}]


def _extract_protein_sequence(text: str) -> str:
    match = PROTEIN_TAG_RE.search(text)
    if match is None:
        raise ValueError("protein sample text must contain <PROT>...</PROT>")
    sequence = re.sub(r"\s+", "", match.group(1)).upper()
    if not sequence:
        raise ValueError("protein sequence inside <PROT>...</PROT> is empty")
    return sequence


def _extract_protein_sequence_and_qwen_text(text: str) -> tuple[str, str]:
    matches = list(PROTEIN_TAG_RE.finditer(text))
    if len(matches) != 1:
        raise ValueError("protein sample text must contain exactly one <PROT>...</PROT>")
    match = matches[0]
    sequence = re.sub(r"\s+", "", match.group(1)).upper()
    if not sequence:
        raise ValueError("protein sequence inside <PROT>...</PROT> is empty")
    qwen_text = text[: match.start()] + "<PROT></PROT>" + text[match.end() :]
    return qwen_text, sequence


def _space_protein_sequence(text: str) -> tuple[str, str]:
    def replace_match(match: re.Match) -> str:
        sequence = re.sub(r"\s+", "", match.group(1)).upper()
        if not sequence:
            raise ValueError("protein sequence inside <PROT>...</PROT> is empty")
        return "<PROT> " + " ".join(sequence) + " </PROT>"

    normalized_text, count = PROTEIN_TAG_RE.subn(replace_match, text, count=1)
    if count != 1:
        raise ValueError("protein sample text must contain exactly one <PROT>...</PROT>")
    return normalized_text, _extract_protein_sequence(normalized_text)


def _normalize_protein_messages_and_sequence(
    messages: List[Dict[str, Any]],
    remove_protein_sequence: bool = False,
) -> tuple[List[Dict[str, Any]], str]:
    normalized_messages = []
    protein_sequence = None
    protein_tag_count = 0

    for msg in messages:
        normalized_content = []
        for item in msg.get("content", []):
            if not isinstance(item, dict) or item.get("type") != "text":
                normalized_content.append(item)
                continue

            text = str(item.get("text", ""))
            tag_count = len(PROTEIN_TAG_RE.findall(text))
            if tag_count:
                protein_tag_count += tag_count
                if protein_sequence is not None:
                    raise ValueError("protein sample contains multiple <PROT>...</PROT> spans")
                if remove_protein_sequence:
                    text, protein_sequence = _extract_protein_sequence_and_qwen_text(text)
                else:
                    text, protein_sequence = _space_protein_sequence(text)
            normalized_content.append({**item, "text": text})
        normalized_messages.append({**msg, "content": normalized_content})

    if protein_tag_count != 1 or protein_sequence is None:
        raise ValueError("protein sample must contain exactly one <PROT>...</PROT> span")
    return normalized_messages, protein_sequence


def _normalize_protein_label(label: Any) -> torch.Tensor:
    if isinstance(label, str):
        label = label.strip()
        if any(ch not in "01" for ch in label):
            raise ValueError("protein label string must contain only 0/1")
        values = [float(ch) for ch in label]
        return torch.tensor(values, dtype=torch.float32)

    if isinstance(label, Sequence):
        values = [float(x) for x in label]
        if any(value not in (0.0, 1.0) for value in values):
            raise ValueError("protein label list must contain only 0/1 values")
        return torch.tensor(values, dtype=torch.float32)

    raise TypeError(f"unsupported protein label type: {type(label)}")


def _find_subsequence(sequence: Sequence[int], subsequence: Sequence[int]) -> int:
    if not subsequence:
        raise ValueError("cannot locate an empty token subsequence")
    last_start = len(sequence) - len(subsequence)
    for start in range(last_start + 1):
        if list(sequence[start : start + len(subsequence)]) == list(subsequence):
            return start
    return -1


class OmniSupervisedDataset(Dataset):
    def __init__(self, processor, data_args):
        super().__init__()
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.data_args = data_args
        self.list_data_dict = []
        self.pooled_hidden_cache = None

        self.list_data_dict = _load_data_dict(data_args)

        self.regression_label_mean, self.regression_label_std, self.regression_label_count = (
            _compute_regression_label_stats(self.list_data_dict)
        )
        self.route_counts, self.class_counts = _compute_head_label_counts(self.list_data_dict)
        if getattr(data_args, "pooled_hidden_cache_path", None):
            self.pooled_hidden_cache = load_pooled_hidden_cache(data_args.pooled_hidden_cache_path)
            if self.pooled_hidden_cache.size(0) != len(self.list_data_dict):
                raise ValueError(
                    "pooled hidden cache length mismatch: "
                    f"{self.pooled_hidden_cache.size(0)} cached rows for "
                    f"{len(self.list_data_dict)} samples"
                )
            print(
                "S1Omni pooled hidden cache loaded:",
                {
                    "path": data_args.pooled_hidden_cache_path,
                    "shape": tuple(self.pooled_hidden_cache.shape),
                    "dtype": str(self.pooled_hidden_cache.dtype),
                },
                flush=True,
            )
        print(
            "S1Omni regression label normalization:",
            {
                "count": self.regression_label_count,
                "mean": self.regression_label_mean,
                "std": self.regression_label_std,
                "route_counts": self.route_counts,
                "class_counts": self.class_counts,
            },
            flush=True,
        )
        rank0_print(f"Total training samples: {len(self.list_data_dict)}")

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, idx):
        num_base_retries = 3
        for attempt_idx in range(num_base_retries):
            try:
                return self._get_item(self.list_data_dict[idx], idx)
            except Exception as e:
                print(f"[Try #{attempt_idx}] Failed to fetch sample {idx}. Exception:", e)
                time.sleep(1)
        raise RuntimeError(f"Failed to fetch sample {idx}")

    def _get_item(self, source, idx: int):
        base_path = Path(source.get("data_path", ""))
        messages = _build_messages(source, base_path)
        assistant_text = _extract_assistant_text(messages)
        task_type, route_id, label = _infer_route_and_label(source["label"], assistant_text)
        if route_id == ROUTE_REG:
            label = (label - self.regression_label_mean) / self.regression_label_std

        model_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        input_ids = model_inputs["input_ids"]
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids).unsqueeze(0)

        labels = torch.full_like(input_ids, IGNORE_INDEX)
        labels[:, -1] = IGNORE_INDEX

        result = dict(model_inputs)
        result["input_ids"] = input_ids
        result["labels"] = labels
        result["route_id"] = torch.tensor(route_id, dtype=torch.long)
        result["label"] = label
        result["task_type"] = task_type
        result["attention_mask"] = input_ids.ne(self.tokenizer.pad_token_id)
        if self.pooled_hidden_cache is not None:
            result["pooled_hidden"] = self.pooled_hidden_cache[idx]
        if getattr(self.data_args, "include_cache_index", False):
            result["cache_index"] = torch.tensor(idx, dtype=torch.long)
        return result


def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)
    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)
    return torch.cat(padded_tensors, dim=1)


@dataclass
class OmniDataCollator(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"].squeeze(0) for instance in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "route_id": torch.stack([instance["route_id"] for instance in instances]),
            "label": torch.stack([instance["label"].to(torch.float32) for instance in instances]),
            "task_type": [instance["task_type"] for instance in instances],
        }
        if "pixel_values" in instances[0]:
            batch["pixel_values"] = torch.cat([instance["pixel_values"] for instance in instances], dim=0)
        if "image_grid_thw" in instances[0]:
            batch["image_grid_thw"] = torch.cat([instance["image_grid_thw"] for instance in instances], dim=0)
        if "pixel_values_videos" in instances[0]:
            batch["pixel_values_videos"] = torch.cat([instance["pixel_values_videos"] for instance in instances], dim=0)
        if "video_grid_thw" in instances[0]:
            batch["video_grid_thw"] = torch.cat([instance["video_grid_thw"] for instance in instances], dim=0)
        if "pooled_hidden" in instances[0]:
            batch["pooled_hidden"] = torch.stack([instance["pooled_hidden"] for instance in instances])
        if "cache_index" in instances[0]:
            batch["cache_index"] = torch.stack([instance["cache_index"] for instance in instances])
        return batch


class ProteinSupervisedDataset(Dataset):
    def __init__(self, processor, data_args, annotation_path: Optional[str] = None, split_name: str = "training"):
        super().__init__()
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.data_args = data_args
        self.list_data_dict = _load_data_dict(data_args, annotation_path=annotation_path)
        rank0_print(f"Total protein {split_name} samples: {len(self.list_data_dict)}")

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, idx):
        num_base_retries = 3
        for attempt_idx in range(num_base_retries):
            try:
                return self._get_item(self.list_data_dict[idx], idx)
            except Exception as e:
                print(f"[Try #{attempt_idx}] Failed to fetch protein sample {idx}. Exception:", e)
                time.sleep(1)
        raise RuntimeError(f"Failed to fetch protein sample {idx}")

    def _get_item(self, source, idx: int):
        use_esm2 = getattr(self.data_args, "use_esm2", True)
        messages, protein_sequence = _normalize_protein_messages_and_sequence(
            _build_protein_messages(source),
            remove_protein_sequence=use_esm2,
        )
        raw_label = source.get("label", source.get("ground_truth"))
        if raw_label is None:
            raise KeyError("protein sample must contain 'label' or 'ground_truth'")
        protein_labels = _normalize_protein_label(raw_label)
        if protein_labels.numel() != len(protein_sequence):
            raise ValueError(
                "protein label length must match protein sequence length, "
                f"got label length {protein_labels.numel()} and sequence length {len(protein_sequence)}"
            )

        model_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        input_ids = model_inputs["input_ids"]
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids).unsqueeze(0)

        result = dict(model_inputs)
        result["input_ids"] = input_ids
        result["attention_mask"] = input_ids.ne(self.tokenizer.pad_token_id)
        result["protein_labels"] = protein_labels
        result["protein_label_mask"] = torch.ones_like(protein_labels, dtype=torch.bool)
        result["protein_sequence"] = protein_sequence
        if not use_esm2:
            residue_token_ids = self.tokenizer.encode(
                " " + " ".join(protein_sequence),
                add_special_tokens=False,
            )
            flat_input_ids = input_ids.squeeze(0).tolist()
            residue_start = _find_subsequence(flat_input_ids, residue_token_ids)
            if residue_start < 0:
                raise ValueError("failed to locate spaced protein residue tokens in tokenized input")

            protein_token_mask = torch.zeros(input_ids.size(1), dtype=torch.bool)
            protein_token_mask[residue_start : residue_start + len(residue_token_ids)] = True
            result["protein_token_mask"] = protein_token_mask
        if "output" in source:
            result["output"] = source["output"]
        if getattr(self.data_args, "include_cache_index", False):
            result["cache_index"] = torch.tensor(idx, dtype=torch.long)
        return result


@dataclass
class ProteinDataCollator(object):
    tokenizer: transformers.PreTrainedTokenizer
    use_esm2: bool = True
    esm_model_name: str = "facebook/esm2_t33_650M_UR50D"

    def __post_init__(self):
        self.esm_tokenizer = None
        if self.use_esm2:
            self.esm_tokenizer = transformers.AutoTokenizer.from_pretrained(self.esm_model_name)

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"].squeeze(0) for instance in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        protein_labels = torch.nn.utils.rnn.pad_sequence(
            [instance["protein_labels"].to(torch.float32) for instance in instances],
            batch_first=True,
            padding_value=0.0,
        )
        protein_label_mask = torch.nn.utils.rnn.pad_sequence(
            [instance["protein_label_mask"] for instance in instances],
            batch_first=True,
            padding_value=False,
        )

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "protein_labels": protein_labels,
            "protein_label_mask": protein_label_mask,
        }
        if self.use_esm2:
            protein_sequences = [instance["protein_sequence"] for instance in instances]
            esm_inputs = self.esm_tokenizer(
                protein_sequences,
                padding=True,
                return_tensors="pt",
            )
            batch["esm_input_ids"] = esm_inputs["input_ids"]
            batch["esm_attention_mask"] = esm_inputs["attention_mask"]
        else:
            protein_token_mask = torch.nn.utils.rnn.pad_sequence(
                [instance["protein_token_mask"] for instance in instances],
                batch_first=True,
                padding_value=False,
            )
            batch["protein_token_mask"] = protein_token_mask
        if "cache_index" in instances[0]:
            batch["cache_index"] = torch.stack([instance["cache_index"] for instance in instances])
        return batch


def make_supervised_data_module(processor, data_args) -> Dict:
    if getattr(data_args, "model_architecture", "").lower() == "s1-protein":
        train_dataset = ProteinSupervisedDataset(processor, data_args=data_args, split_name="training")
        eval_dataset = None
        if getattr(data_args, "eval_annotation_path", None):
            eval_dataset = ProteinSupervisedDataset(
                processor,
                data_args=data_args,
                annotation_path=data_args.eval_annotation_path,
                split_name="eval",
            )
        data_collator = ProteinDataCollator(
            processor.tokenizer,
            use_esm2=getattr(data_args, "use_esm2", True),
            esm_model_name=getattr(data_args, "esm_model_name", "facebook/esm2_t33_650M_UR50D"),
        )
    else:
        train_dataset = OmniSupervisedDataset(processor, data_args=data_args)
        eval_dataset = None
        data_collator = OmniDataCollator(processor.tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)
