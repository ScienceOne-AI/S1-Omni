#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QWEN_ROOT = Path(__file__).resolve().parent
GRADIO_TEMP_DIR = PROJECT_ROOT / ".gradio_tmp"
GRADIO_TEMP_DIR.mkdir(exist_ok=True)
os.environ.setdefault("GRADIO_TEMP_DIR", str(GRADIO_TEMP_DIR))

import gradio as gr


DEFAULT_URL = "http://localhost:8009/v1/chat/completions"
DEFAULT_MODEL = "s1-omni"
DEFAULT_SPEC2MOL_JDX_FILES = [
    Path("s1_omni_infer") / "spec2mol" / "dataset" / "test_data" / "000010" / "000010_ir.jdx",
    Path("s1_omni_infer") / "spec2mol" / "dataset" / "test_data" / "000010" / "000010_raman.jdx",
    Path("s1_omni_infer") / "spec2mol" / "dataset" / "test_data" / "000010" / "000010_uv.jdx",
]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
STRUCTURE_SUFFIXES = {".pdb", ".cif", ".mmcif", ".sdf"}
PROTEIN_TASKS = {"蛋白结构预测", "蛋白位点预测"}
PATH_RE = re.compile(
    r"(?P<path>(?:/|\.{1,2}/|[A-Za-z]:\\)[^\s\"'<>]+?\.(?:png|jpg|jpeg|webp|bmp|gif|pdb|cif|mmcif|sdf))"
    r"(?=$|[\s\"'<>，。；;,.])",
    flags=re.IGNORECASE,
)
PROT_TAG_RE = re.compile(r"<PROT>(?P<sequence>.*?)</PROT>", flags=re.IGNORECASE | re.DOTALL)


TASK_PROMPTS = {
    "通用对话": "",
    "蛋白位点预测": "Given protein sequence <PROT>KETAAAKFERQHMDSSTSAASSSNYCNQMMKSRNLTKDRCKPVNTFVHESLADVQAVCSQKNVACKNGQTNCYQSYSTMSITDCRETGSSKYPNCAYKTTQANKHIIVACEGNPYVPVHFDASV</PROT>, predict the residues that form protein–protein interaction (PPI) binding sites.",
    "蛋白结构预测": (
        "你是一名面向蛋白质建模的助手。用户给出序列后，请说明这些功能和结构域线索如何影响结构预测。\n\n<PROT>MLNGISNAASTLGRQLVGIASRVSSAGGTGFSVAPQAVRLTPVRVHSPFSPGSSNVNARTIFNVSSQVTSFTPSRPAPPPPTSGQASGASRPLPPIAQALKDHLAAYELSKASETVNFKPTRPAPPPPTSGQASGASRPLPPIAQALKDHLAAYELSKASETVSFKPTRQAPPPPTSGQASGPGGLPPLAQALKDHLAAYEQSKKG</PROT>"
    ),
    "性质分类": "<SMILES>CCOP(=O)(OCC)C12CC(C)C=CC1COC2=O</SMILES> 是否可被认为具有抗 HIV 复制活性？",
    "性质回归": "请评估<SMILES>C1=CC2=CC=C3C=CC4=CC=C5C=CC6=CC=C1C1=C2C3=C4C5=C61</SMILES>在水中的溶解能力，并报告ESOL log S数值。",
    "文生图": "人体细胞内部剖面，线粒体、DNA 双螺旋、核糖体漂浮，水润半透明细胞膜，淡青、薄荷绿、浅粉柔和配色，液态流动细胞质，柔和柔光，纯白极简实验室背景，干净通透，生物医学科普插画，扁平细腻质感，高清晰度，文字采用英文，8K高清",
    "图像编辑": "请编辑上传的图片：保持主体不变，把背景改成浅蓝色实验室场景。",
    "谱图生成分子": "请你基于给出的红外、拉曼和 UV 谱，预测可能的分子结构。",
}

def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        detail = body
        try:
            error_payload = json.loads(body)
            detail = str(error_payload.get("detail", body))
        except json.JSONDecodeError:
            pass
        raise RuntimeError(f"HTTP {exc.code} {exc.reason}: {detail}") from exc


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


def uploaded_path(file_obj: Any) -> str | None:
    if file_obj is None:
        return None
    if isinstance(file_obj, (str, Path)):
        return str(Path(file_obj).resolve())
    name = getattr(file_obj, "name", None)
    if name:
        return str(Path(name).resolve())
    path = getattr(file_obj, "path", None)
    if path:
        return str(Path(path).resolve())
    return None


def normalize_uploads(files: Any) -> list[str]:
    if files is None:
        return []
    if not isinstance(files, list):
        files = [files]
    paths = []
    for file_obj in files:
        path = uploaded_path(file_obj)
        if path:
            paths.append(path)
    return paths


def default_spec2mol_jdx_paths() -> list[str]:
    return [str((PROJECT_ROOT / path).resolve()) for path in DEFAULT_SPEC2MOL_JDX_FILES]


def has_tagged_protein_sequence(prompt: str) -> bool:
    return any(match.group("sequence").strip() for match in PROT_TAG_RE.finditer(prompt or ""))


def protein_prompt_error(task_name: str, prompt: str) -> str | None:
    if task_name not in PROTEIN_TASKS or has_tagged_protein_sequence(prompt):
        return None
    return (
        "蛋白结构预测和蛋白位点预测任务需要将氨基酸序列包裹在 "
        "<PROT>...</PROT> 中，例如：<PROT>ACDEFGHIKLMNPQRSTVWY</PROT>。"
    )


def build_messages(
    chat_state: list[dict[str, str]],
    prompt: str,
    image_paths: list[str],
    jdx_paths: list[str],
    pdb_paths: list[str],
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for item in chat_state:
        role = item.get("role")
        content = item.get("content", "")
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": [{"type": "text", "text": content}]})

    user_content: list[dict[str, Any]] = []
    for path in image_paths:
        user_content.append({"type": "image_url", "image_url": path})
    if jdx_paths:
        user_content.append({"type": "jdx_files", "jdx_files": jdx_paths})
    if pdb_paths:
        user_content.append({"type": "pdb_files", "pdb_files": pdb_paths})
    if prompt.strip():
        user_content.append({"type": "text", "text": prompt.strip()})
    if not user_content:
        user_content.append({"type": "text", "text": "请根据输入内容完成任务。"})
    messages.append({"role": "user", "content": user_content})
    return messages


def compact_user_display(prompt: str, image_paths: list[str], jdx_paths: list[str], pdb_paths: list[str]) -> str:
    lines = []
    if image_paths:
        lines.append("Images: " + ", ".join(Path(path).name for path in image_paths))
    if jdx_paths:
        lines.append("JDX: " + ", ".join(Path(path).name for path in jdx_paths))
    if pdb_paths:
        lines.append("PDB/mmCIF: " + ", ".join(Path(path).name for path in pdb_paths))
    if prompt.strip():
        lines.append(prompt.strip())
    return "\n".join(lines) or "请根据输入内容完成任务。"


def prediction_paths(response: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    result = response.get("result") if isinstance(response.get("result"), dict) else response
    prediction = result.get("prediction") if isinstance(result, dict) else None
    if isinstance(prediction, dict):
        for key in ("image_path", "structure_path", "sdf_path"):
            value = prediction.get(key)
            if value:
                paths.append(str(value))
    final_text = extract_final_text(response)
    paths.extend(match.group("path") for match in PATH_RE.finditer(final_text))

    stripped = final_text.strip().strip("\"'`")
    if Path(stripped).suffix.lower() in IMAGE_SUFFIXES | STRUCTURE_SUFFIXES:
        paths.append(stripped)

    deduped = []
    seen = set()
    for path in paths:
        resolved = resolve_output_path(path)
        if resolved and resolved not in seen:
            seen.add(resolved)
            deduped.append(resolved)
    return deduped


def prediction_payload(response: dict[str, Any]) -> dict[str, Any] | None:
    result = response.get("result") if isinstance(response.get("result"), dict) else response
    prediction = result.get("prediction") if isinstance(result, dict) else None
    return prediction if isinstance(prediction, dict) else None


def protein_viewer_html(response: dict[str, Any]) -> str:
    prediction = prediction_payload(response)
    if not prediction:
        return ""
    value = prediction.get("visualization_html")
    return str(value) if value else ""


def prediction_artifact_path(response: dict[str, Any]) -> str | None:
    prediction = prediction_payload(response)
    if not prediction:
        return None
    for key in ("prediction_jsonl", "structure_path", "sdf_path", "image_path"):
        value = prediction.get(key)
        if value:
            resolved = resolve_output_path(str(value))
            if resolved:
                return resolved
    return None


def resolve_output_path(path: str) -> str | None:
    raw = Path(path.strip().strip("\"'`"))
    candidates = [raw] if raw.is_absolute() else [Path.cwd() / raw, PROJECT_ROOT / raw, QWEN_ROOT / raw]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return None


def guess_structure_format(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in {".cif", ".mmcif"}:
        return "cif"
    if suffix == ".sdf":
        return "sdf"
    return "pdb"


def render_structure_path(path: str | None) -> str:
    if not path:
        return ""
    file_path = Path(path)
    if not file_path.exists():
        return f"<pre>File not found: {html.escape(str(file_path))}</pre>"

    structure_text = file_path.read_text(errors="replace")
    fmt = guess_structure_format(file_path)
    viewer_html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
  <style>
    html, body, #viewer {{
      margin: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      background: white;
    }}
  </style>
</head>
<body>
  <div id="viewer"></div>
  <script>
    const structure = {json.dumps(structure_text)};
    const format = {json.dumps(fmt)};
    const viewer = $3Dmol.createViewer("viewer", {{ backgroundColor: "white" }});
    viewer.addModel(structure, format);
    const model = viewer.getModel();
    const atoms = model.selectedAtoms({{}});
    const hasProteinBackbone = atoms.some(atom => atom.atom === "CA");
    if (hasProteinBackbone) {{
      viewer.setStyle({{}}, {{ cartoon: {{ color: "spectrum" }} }});
    }} else {{
      viewer.setStyle({{}}, {{ stick: {{ radius: 0.18 }}, sphere: {{ scale: 0.25 }} }});
    }}
    viewer.zoomTo();
    viewer.render();
  </script>
</body>
</html>
"""
    return f"""
<iframe
  srcdoc="{html.escape(viewer_html, quote=True)}"
  style="width:100%; height:650px; border:0;"
></iframe>
"""


def select_outputs(response: dict[str, Any]) -> tuple[str | None, str | None]:
    image_path = None
    structure_path = None
    for path in prediction_paths(response):
        suffix = Path(path).suffix.lower()
        if suffix in IMAGE_SUFFIXES and image_path is None:
            image_path = path
        elif suffix in STRUCTURE_SUFFIXES and structure_path is None:
            structure_path = path
    return image_path, structure_path


def pretty_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def submit(
    task_name: str,
    endpoint_url: str,
    model_name: str,
    prompt: str,
    image_files: Any,
    jdx_files: Any,
    pdb_files: Any,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    force_spectra: bool,
    timeout: float,
    chatbot_format: str,
    chat_display: list[dict[str, str]] | None,
    chat_state: list[dict[str, str]] | None,
):
    chat_display = list(chat_display or [])
    chat_state = list(chat_state or [])
    all_image_paths = normalize_uploads(image_files)
    all_jdx_paths = normalize_uploads(jdx_files)
    all_pdb_paths = normalize_uploads(pdb_files)
    image_paths = all_image_paths if task_name in {"图像编辑", "多图理解"} else []
    if task_name == "谱图生成分子":
        jdx_paths = all_jdx_paths or default_spec2mol_jdx_paths()
    else:
        jdx_paths = []
    pdb_paths = all_pdb_paths if task_name == "蛋白位点预测" else []
    effective_force_spectra = bool(force_spectra) and task_name == "谱图生成分子"

    user_text = compact_user_display(prompt, image_paths, jdx_paths, pdb_paths)
    uses_messages = chatbot_format == "messages"
    validation_error = protein_prompt_error(task_name, prompt)
    if validation_error:
        gr.Warning(html.escape(validation_error))
        raw = pretty_json({"warning": validation_error, "task": task_name})
        yield chat_display, chat_state, "", None, "", None, raw
        return

    messages = build_messages(chat_state, prompt, image_paths, jdx_paths, pdb_paths)
    payload = {
        "model": model_name or DEFAULT_MODEL,
        "messages": messages,
        "max_new_tokens": int(max_new_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "return_full_result": True,
    }
    if effective_force_spectra or jdx_paths:
        payload["force_spectra"] = bool(effective_force_spectra or jdx_paths)

    if uses_messages:
        chat_display.append({"role": "user", "content": user_text})
        chat_display.append({"role": "assistant", "content": "请求中..."})
    else:
        chat_display.append([user_text, "请求中..."])

    yield chat_display, chat_state, "", None, "", None, ""

    try:
        start = time.time()
        response = post_json(endpoint_url.strip(), payload, float(timeout))
        elapsed = time.time() - start
        final_text = extract_final_text(response)
        image_path, structure_path = select_outputs(response)
        viewer_html = protein_viewer_html(response) or render_structure_path(structure_path)
        file_path = prediction_artifact_path(response) or structure_path or image_path
        raw = pretty_json(response)
        assistant_text = final_text or "(empty response)"
        assistant_text = assistant_text.replace("<|im_end|>", "")
        if uses_messages:
            chat_display[-1] = {"role": "assistant", "content": assistant_text}
        else:
            chat_display[-1] = [user_text, assistant_text]
        chat_state.append({"role": "user", "content": user_text})
        chat_state.append({"role": "assistant", "content": assistant_text})
        status = f"完成，用时 {elapsed:.2f}s"
        if file_path:
            status += f"\n解析到文件: {file_path}"
        yield chat_display, chat_state, final_text, image_path, viewer_html, file_path, raw + "\n\n" + status
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError, RuntimeError) as exc:
        message = f"请求失败: {exc}"
        if uses_messages:
            chat_display[-1] = {"role": "assistant", "content": message}
        else:
            chat_display[-1] = [user_text, message]
        raw = pretty_json({"error": message, "request": payload})
        yield chat_display, chat_state, message, None, "", None, raw


def clear_history():
    return [], [], "", None, "", None, ""


def apply_task_template(task: str):
    prompt = TASK_PROMPTS.get(task, "")
    jdx_files = default_spec2mol_jdx_paths() if task == "谱图生成分子" else None
    force_spectra = task == "谱图生成分子"
    return prompt, jdx_files, force_spectra


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_name", default="0.0.0.0")
    parser.add_argument("--server_port", type=int, default=7880)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--api_url", default=DEFAULT_URL)
    return parser.parse_args()


def build_demo(default_url: str) -> gr.Blocks:
    with gr.Blocks(title="S1-Omni OpenAI Client") as demo:
        gr.Markdown("## S1-Omni OpenAI Client")

        chat_state = gr.State([])
        with gr.Row():
            with gr.Column(scale=5):
                chatbot = gr.Chatbot(label="对话", height=520)
                chatbot_format = gr.State(getattr(chatbot, "type", "messages") or "messages")
                prompt = gr.Textbox(label="文本输入", lines=6, placeholder="输入问题、编辑指令、生成描述或任务说明")
                with gr.Row():
                    submit_btn = gr.Button("发送", variant="primary")
                    clear_btn = gr.Button("清空历史")
            with gr.Column(scale=4):
                task = gr.Dropdown(
                    label="任务模板",
                    choices=list(TASK_PROMPTS),
                    value="通用对话",
                )
                image_files = gr.Files(
                    label="上传图片（支持单图/多图）",
                    file_types=["image"],
                )
                jdx_files = gr.Files(
                    label="上传 JDX 文件（IR / Raman / UV，可多选）",
                    file_types=[".jdx", ".dx"],
                )
                pdb_files = gr.Files(
                    label="上传 PDB 文件（蛋白位点预测，可多选）",
                    file_types=[".pdb", ".cif", ".mmcif"],
                )
                with gr.Accordion("请求参数", open=False):
                    endpoint_url = gr.Textbox(label="OpenAI API URL", value=default_url)
                    model_name = gr.Textbox(label="model", value=DEFAULT_MODEL)
                    max_new_tokens = gr.Slider(2048, 16384, value=8096, step=2048, label="max_new_tokens")
                    temperature = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="temperature")
                    top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.01, label="top_p")
                    force_spectra = gr.Checkbox(label="force_spectra", value=False)
                    timeout = gr.Number(label="timeout seconds", value=1800)

        with gr.Row():
            final_text = gr.Textbox(label="final_text", lines=6)
        with gr.Row():
            image_output = gr.Image(label="图片输出", type="filepath")
            file_output = gr.File(label="解析到的输出文件")
        viewer = gr.HTML(label="PDB / CIF / SDF 3D Viewer")
        raw_json = gr.Code(label="原始响应 JSON", language="json", lines=18)

        task.change(
            fn=apply_task_template,
            inputs=[task],
            outputs=[prompt, jdx_files, force_spectra],
        )
        submit_btn.click(
            fn=submit,
            inputs=[
                task,
                endpoint_url,
                model_name,
                prompt,
                image_files,
                jdx_files,
                pdb_files,
                max_new_tokens,
                temperature,
                top_p,
                force_spectra,
                timeout,
                chatbot_format,
                chatbot,
                chat_state,
            ],
            outputs=[chatbot, chat_state, final_text, image_output, viewer, file_output, raw_json],
        )
        prompt.submit(
            fn=submit,
            inputs=[
                task,
                endpoint_url,
                model_name,
                prompt,
                image_files,
                jdx_files,
                pdb_files,
                max_new_tokens,
                temperature,
                top_p,
                force_spectra,
                timeout,
                chatbot_format,
                chatbot,
                chat_state,
            ],
            outputs=[chatbot, chat_state, final_text, image_output, viewer, file_output, raw_json],
        )
        clear_btn.click(
            fn=clear_history,
            inputs=None,
            outputs=[chatbot, chat_state, final_text, image_output, viewer, file_output, raw_json],
        )
    return demo


def main() -> None:
    args = parse_args()
    demo = build_demo(args.api_url)
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        allowed_paths=[str(PROJECT_ROOT), str(GRADIO_TEMP_DIR)],
    )


if __name__ == "__main__":
    main()
