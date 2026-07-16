#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_URL = "http://localhost:8009/v1/chat/completions"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Sample:
    name: str
    description: str
    payload: dict[str, Any]


def project_relative(*parts: str) -> str:
    return str(PROJECT_ROOT.joinpath(*parts).relative_to(PROJECT_ROOT))


def build_samples(include_route_hints: bool = True) -> list[Sample]:

    return [
        Sample(
            name="protein_site",
            description="蛋白/RNA 结合位点预测，期望 final_text 替换为阳性位点索引列表。",
            payload={
                "model": "s1-omni",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Given protein sequence <PROT>KETAAAKFERQHMDSSTSAASSSNYCNQMMKSRNLTKDRCKPVNTFVHESLADVQAVCSQKNVACKNGQTNCYQSYSTMSITDCRETGSSKYPNCAYKTTQANKHIIVACEGNPYVPVHFDASV</PROT>, predict the residues that form protein–protein interaction (PPI) binding sites."
                            }
                        ],
                    },
                ],
                "max_new_tokens": 8192,
                "temperature": 0.2,
            },
        ),
        Sample(
            name="protein_fold",
            description="蛋白结构预测，期望 final_text 替换为生成的 cif/pdb 文件路径。",
            payload={
                "model": "s1-omni",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "你是一名面向蛋白质建模的助手。用户给出序列后，请说明这些功能和结构域线索如何影响结构预测。\n\n<PROT>MLNGISNAASTLGRQLVGIASRVSSAGGTGFSVAPQAVRLTPVRVHSPFSPGSSNVNARTIFNVSSQVTSFTPSRPAPPPPTSGQASGASRPLPPIAQALKDHLAAYELSKASETVNFKPTRPAPPPPTSGQASGASRPLPPIAQALKDHLAAYELSKASETVSFKPTRQAPPPPTSGQASGPGGLPPLAQALKDHLAAYEQSKKG</PROT>"
                                ),
                            }
                        ],
                    },
                ],
                "max_new_tokens": 8192,
            },
        ),
        Sample(
            name="property_classification",
            description="小分子性质分类，期望 final_text 替换为 0/1。",
            payload={
                "model": "s1-omni",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "<SMILES>CCOP(=O)(OCC)C12CC(C)C=CC1COC2=O</SMILES> 是否可被认为具有抗 HIV 复制活性？"
                            }
                        ],
                    },
                ],
                "max_new_tokens": 8192,
            },
        ),
        Sample(
            name="property_regression",
            description="小分子性质回归，期望 final_text 替换为数值。",
            payload={
                "model": "s1-omni",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "请评估<SMILES>C1=CC2=CC=C3C=CC4=CC=C5C=CC6=CC=C1C1=C2C3=C4C5=C61</SMILES>在水中的溶解能力，并报告ESOL log S数值。",
                            }
                        ],
                    },
                ],
                "max_new_tokens": 8192,
            },
        ),
        Sample(
            name="image_generation",
            description="文生图，期望 final_text 替换为生成图片路径。",
            payload={
                "model": "s1-omni",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "人体细胞内部剖面，线粒体、DNA 双螺旋、核糖体漂浮，水润半透明细胞膜，淡青、薄荷绿、浅粉柔和配色，液态流动细胞质，柔和柔光，纯白极简实验室背景，干净通透，生物医学科普插画，扁平细腻质感，高清晰度，文字采用英文，8K高清",
                            }
                        ],
                    },
                ],
                "max_new_tokens": 8192,
            },
        ),
        Sample(
            name="image_edit",
            description="图像编辑，messages 中包含单图，期望 final_text 替换为编辑后图片路径。",
            payload={
                "model": "s1-omni",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": project_relative("pred_out", "images", "image_gen_1.png"),
                            },
                            {
                                "type": "text",
                                "text": (
                                    "请把图片背景颜色改为浅棕色"
                                ),
                            },
                        ],
                    },
                ],
                "max_new_tokens": 8192,
            },
        ),
        Sample(
            name="spectra",
            description="JDX 谱图到分子生成，期望 final_text 替换为 sdf 文件路径或 smiles。",
            payload={
                "model": "s1-omni",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "jdx_files",
                                "jdx_files": [
                                    project_relative("s1_omni_infer", "spec2mol", "dataset", "test_data", "000010", "000010_ir.jdx"),
                                    project_relative("s1_omni_infer", "spec2mol", "dataset", "test_data", "000010", "000010_raman.jdx"),
                                    project_relative("s1_omni_infer", "spec2mol", "dataset", "test_data", "000010", "000010_uv.jdx"),
                                ],
                            },
                            {
                                "type": "text",
                                "text": "请你基于给出的红外、拉曼和UV谱，预测可能的分子结构",
                            },
                        ],
                    },
                ],
                "max_new_tokens": 8192,
                "force_spectra": True,
            },
        )
    ]


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def format_http_error(exc: urllib.error.HTTPError) -> str:
    body = exc.read().decode("utf-8", errors="replace")
    if not body:
        return str(exc)
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return f"{exc}\n{body}"
    error = payload.get("error")
    if isinstance(error, dict):
        lines = [
            f"{exc.code} {exc.reason}",
            f"type: {error.get('type', '')}",
            f"message: {error.get('message', '')}",
        ]
        if error.get("hint"):
            lines.append(f"hint: {error['hint']}")
        if error.get("request_id"):
            lines.append(f"request_id: {error['request_id']}")
        causes = error.get("causes")
        if causes:
            lines.append(f"causes: {causes}")
        return "\n".join(lines)
    return f"{exc}\n{json.dumps(payload, ensure_ascii=False, indent=2)}"


def iter_sse(url: str, payload: dict[str, Any], timeout: float):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        event_lines: list[str] = []
        for raw_line in response:
            line = raw_line.decode("utf-8").rstrip("\n")
            if not line:
                if event_lines:
                    yield "\n".join(event_lines)
                    event_lines = []
                continue
            if line.startswith("data: "):
                event_lines.append(line[len("data: ") :])
        if event_lines:
            yield "\n".join(event_lines)


def extract_final_text(response: dict[str, Any]) -> str:
    if "final_text" in response:
        return str(response["final_text"])
    result = response.get("result")
    if isinstance(result, dict) and "final_text" in result:
        return str(result["final_text"])
    choices = response.get("choices") or []
    if choices:
        message = choices[0].get("message") or {}
        return str(message.get("content", ""))
    return ""


def print_response(sample: Sample, response: dict[str, Any], elapsed: float) -> None:
    final_text = extract_final_text(response)
    result = response.get("result") if isinstance(response.get("result"), dict) else response
    selected_by = result.get("selected_by") if isinstance(result, dict) else None
    token = result.get("final_special_token") if isinstance(result, dict) else None

    print(f"\n=== {sample.name} ===")
    print(sample.description)
    print(f"elapsed: {elapsed:.2f}s")
    if selected_by or token:
        print(f"route: selected_by={selected_by!r}, final_special_token={token!r}")
    print("final_text:")
    print(final_text)


def run_non_stream(sample: Sample, url: str, timeout: float) -> bool:
    start = time.time()
    try:
        response = post_json(url, sample.payload, timeout)
    except urllib.error.HTTPError as exc:
        print(f"\n=== {sample.name} FAILED ===\n{format_http_error(exc)}", file=sys.stderr)
        return False
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"\n=== {sample.name} FAILED ===\n{exc}", file=sys.stderr)
        return False
    print_response(sample, response, time.time() - start)
    return True


def run_stream(sample: Sample, url: str, timeout: float) -> bool:
    payload = dict(sample.payload)
    payload["stream"] = True
    final_parts: list[str] = []
    start = time.time()
    print(f"\n=== {sample.name} [stream] ===")
    print(sample.description)
    try:
        for event in iter_sse(url, payload, timeout):
            if event == "[DONE]":
                break
            data = json.loads(event)
            if "error" in data:
                raise RuntimeError(data["error"])
            text = data.get("final_text", "")
            if not text:
                choices = data.get("choices") or []
                if choices:
                    text = (choices[0].get("delta") or {}).get("final_text", "")
            if text:
                final_parts.append(str(text))
                print(str(text), end="", flush=True)
    except urllib.error.HTTPError as exc:
        print(f"\nFAILED: {format_http_error(exc)}", file=sys.stderr)
        return False
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr)
        return False
    print(f"\nelapsed: {time.time() - start:.2f}s")
    print("final_text:")
    print("".join(final_parts))
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--task", action="append", default=None, help="Run one or more named samples. Default: all.")
    parser.add_argument("--list", action="store_true", help="List available sample names and exit.")
    parser.add_argument("--stream", action="store_true", help="Use stream=true and print SSE chunks.")
    parser.add_argument("--no-route-hints", action="store_true", help="Remove route hint text from sample prompts.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    samples = build_samples(include_route_hints=not args.no_route_hints)
    sample_by_name = {sample.name: sample for sample in samples}

    if args.list:
        for sample in samples:
            print(f"{sample.name}\t{sample.description}")
        return 0

    selected_names = args.task or list(sample_by_name)
    unknown = [name for name in selected_names if name not in sample_by_name]
    if unknown:
        print(f"Unknown task(s): {', '.join(unknown)}", file=sys.stderr)
        print("Use --list to see available samples.", file=sys.stderr)
        return 2

    ok = True
    for name in selected_names:
        sample = sample_by_name[name]
        ok = (run_stream(sample, args.url, args.timeout) if args.stream else run_non_stream(sample, args.url, args.timeout)) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
