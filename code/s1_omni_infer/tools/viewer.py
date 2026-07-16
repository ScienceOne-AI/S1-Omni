#!/usr/bin/env python3
"""
3Dmol.js protein structure viewer — parameter-driven (no manual upload).

Reads a PDB file and a prediction JSONL file, then highlights the predicted
DNA-binding residues for all chains simultaneously.

Usage:
    python viewer.py -p PDB/5DY0.pdb -r output.jsonl [--port 7861]
"""

import argparse
import html
import json
import re
from pathlib import Path

import gradio as gr

RESN_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
NUC_RESNS = {"DA", "DC", "DG", "DT", "A", "C", "G", "T", "U", "DU", "DI"}

# Per-chain colours — 30 distinct, none conflict with highlight red (#e63946)
CHAIN_COLORS = [
    "#9b5de5",  #  1 purple
    "#457b9d",  #  2 steel blue
    "#2a9d8f",  #  3 teal
    "#f4a261",  #  4 sandy orange
    "#e76f51",  #  5 terracotta
    "#00b4d8",  #  6 cyan
    "#6d6875",  #  7 slate
    "#06d6a0",  #  8 mint
    "#ffb703",  #  9 golden yellow
    "#8338ec",  # 10 violet
    "#3a86ff",  # 11 royal blue
    "#bc6c25",  # 12 cinnamon
    "#ff006e",  # 13 magenta
    "#0077b6",  # 14 deep blue
    "#52b788",  # 15 forest green
    "#e36414",  # 16 burnt orange
    "#7209b7",  # 17 deep purple
    "#4cc9f0",  # 18 sky blue
    "#8ac926",  # 19 lime green
    "#f15bb5",  # 20 hot pink
    "#1e6091",  # 21 ocean blue
    "#d4a373",  # 22 tan
    "#9c89b8",  # 23 lavender
    "#386641",  # 24 hunter green
    "#ef476f",  # 25 watermelon
    "#118ab2",  # 26 teal blue
    "#ffd166",  # 27 mustard
    "#073b4c",  # 28 navy
    "#cb997e",  # 29 beige
    "#6a4c93",  # 30 plum
]


# ═══════════════════════════════════════════════════════════════════════════════
#  JSONL prediction loader
# ═══════════════════════════════════════════════════════════════════════════════

def load_predictions(jsonl_path: str, pdb_id: str) -> dict[str, list[int]]:
    """Parse output.jsonl, extract predicted site positions for each chain.

    Returns:
        {chain_id: [pos1, pos2, ...], ...}   e.g. {"A": [1,3,4,...], "B": [...]}
    """
    predictions: dict[str, list[int]] = {}
    pattern = re.compile(r"site is:\s*\[([^\]]+)\]")

    # Handles both standard JSONL (one JSON per line) and pretty-printed JSON
    # objects separated by blank lines.
    text = Path(jsonl_path).read_text(errors="replace")
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(text):
        # Skip whitespace / blank lines
        while idx < len(text) and text[idx] in " \t\n\r":
            idx += 1
        if idx >= len(text):
            break
        try:
            record, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            # Try skipping to next line
            nl = text.find("\n", idx)
            idx = nl + 1 if nl != -1 else len(text)
            continue
        idx = end

        rid: str = record.get("id", "")
        # id format: "5DY0_A"
        if not rid.startswith(pdb_id + "_"):
            continue
        chain = rid[len(pdb_id) + 1:]

        for msg in record.get("messages", []):
            if msg.get("role") != "assistant":
                continue
            m = pattern.search(msg.get("content", ""))
            if m:
                positions = [int(x.strip()) for x in m.group(1).split(",") if x.strip().isdigit()]
                predictions[chain] = positions
                break

    return predictions


# ═══════════════════════════════════════════════════════════════════════════════
#  PDB sequence extraction (standard 20 AAs, CA-only)
# ═══════════════════════════════════════════════════════════════════════════════

def guess_format(path: str) -> str:
    suffix = Path(path).suffix.lower()
    return "cif" if suffix in {".cif", ".mmcif"} else "pdb"


def extract_chain_sequences(file_path: str) -> dict:
    path = Path(file_path)
    suffix = path.suffix.lower()
    chains: dict[str, list[dict]] = {}

    if suffix in {".cif", ".mmcif"}:
        text = path.read_text(errors="replace")
        in_atom_site = False
        col_indices: dict[str, int] = {}
        for line in text.splitlines():
            if line.startswith("loop_"):
                continue
            if line.startswith("_atom_site."):
                in_atom_site = True
                col_indices[line.strip().split(".")[-1]] = len(col_indices)
                continue
            if in_atom_site and line.strip() and not line.startswith("_") and not line[0].isdigit() and not line.startswith(" "):
                in_atom_site = False
                col_indices = {}
                continue
            if in_atom_site and line.strip() and not line.startswith("_"):
                cols = line.strip().split()
                if len(cols) < max(col_indices.values()) + 1:
                    continue
                if cols[col_indices.get("group_PDB", 0)] not in ("ATOM", "HETATM"):
                    continue
                if cols[col_indices.get("label_atom_id", -1)] != "CA":
                    continue
                resn = cols[col_indices.get("label_comp_id", -1)]
                if resn in NUC_RESNS or resn not in RESN_TO_ONE:
                    continue
                chain_id = cols[col_indices.get("label_asym_id", -1)] if "label_asym_id" in col_indices else " "
                try:
                    resi = int(cols[col_indices.get("label_seq_id", -1)])
                except (ValueError, IndexError):
                    continue
                key = (chain_id, resi)
                if chain_id not in chains:
                    chains[chain_id] = []
                    chains[chain_id]._seen = set()  # type: ignore[attr-defined]
                if key not in chains[chain_id]._seen:  # type: ignore[attr-defined]
                    chains[chain_id]._seen.add(key)    # type: ignore[attr-defined]
                    chains[chain_id].append({"resi": resi, "one": RESN_TO_ONE[resn]})
        for c in chains:
            if hasattr(chains[c], '_seen'):
                delattr(chains[c], '_seen')
    else:
        text = path.read_text(errors="replace")
        seen: set[tuple[str, int]] = set()
        for line in text.splitlines():
            if len(line) < 54:
                continue
            if line[0:6].strip() not in ("ATOM", "HETATM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            resn = line[17:20].strip()
            if resn in NUC_RESNS or resn not in RESN_TO_ONE:
                continue
            chain_id = line[21] if len(line) > 21 else " "
            try:
                resi = int(line[22:26].strip())
            except ValueError:
                continue
            key = (chain_id, resi)
            if key in seen:
                continue
            seen.add(key)
            chains.setdefault(chain_id, []).append({"resi": resi, "one": RESN_TO_ONE[resn]})

    for c in chains:
        chains[c].sort(key=lambda r: r["resi"])
        for i, res in enumerate(chains[c]):
            res["pos"] = i + 1

    return chains


# ═══════════════════════════════════════════════════════════════════════════════
#  HTML generation
# ═══════════════════════════════════════════════════════════════════════════════

def build_seq_bar_html(chain_seqs: dict, predictions: dict[str, list[int]]) -> str:
    """Generate sequence bar with predicted positions pre-highlighted per chain."""
    chain_colors_map: dict[str, str] = {}
    for i, c in enumerate(sorted(chain_seqs.keys())):
        chain_colors_map[c] = CHAIN_COLORS[i % len(CHAIN_COLORS)]

    parts: list[str] = []
    for c in sorted(chain_seqs.keys()):
        seq = chain_seqs[c]
        pred_set = set(predictions.get(c, []))
        color = chain_colors_map[c]

        parts.append(
            f'<div class="chain-label">'
            f'<span class="chain-dot" style="background:{color};"></span>'
            f'Chain {html.escape(c)}'
            f'</div>'
        )
        parts.append('<div class="seq-scroll"><div class="seq-inner">')

        # Number row
        parts.append('<div class="num-row">')
        for i, res in enumerate(seq):
            if i % 5 == 0:
                parts.append(f'<div class="num-cell">{res["pos"]}</div>')
            else:
                parts.append('<div class="num-cell"></div>')
        parts.append('</div>')

        # AA row — un-highlighted get per-chain tint, highlighted get dark red
        parts.append('<div class="aa-row">')
        for res in seq:
            if res["pos"] in pred_set:
                parts.append(
                    f'<div class="aa-cell highlighted" '
                    f'style="background:#e63946;color:white;" '
                    f'title="Chain {html.escape(c)} Pos {res["pos"]} (PDB {res["resi"]})">'
                    f'{html.escape(res["one"])}</div>'
                )
            else:
                parts.append(
                    f'<div class="aa-cell chain-tinted" '
                    f'style="color:{color};" '
                    f'title="Chain {html.escape(c)} Pos {res["pos"]} (PDB {res["resi"]})">'
                    f'{html.escape(res["one"])}</div>'
                )
        parts.append('</div>')

        parts.append('</div></div>')

    return "".join(parts)


def build_legend_html(predictions: dict[str, list[int]]) -> str:
    """Build a small colour legend for chains that have predictions."""
    parts = ['<div class="legend">']
    for i, c in enumerate(sorted(predictions.keys())):
        color = CHAIN_COLORS[i % len(CHAIN_COLORS)]
        count = len(predictions[c])
        parts.append(
            f'<span class="legend-item">'
            f'<span class="chain-dot" style="background:{color};"></span>'
            f'Chain {html.escape(c)} ({count} sites)'
            f'</span>'
        )
    parts.append('</div>')
    return "".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main render function
# ═══════════════════════════════════════════════════════════════════════════════

def render_structure_3dmol(pdb_path: str, jsonl_path: str, cdn_url: str) -> str:
    """Build the complete HTML page with 3Dmol.js viewer."""
    pdb_path = Path(pdb_path)
    pdb_id = pdb_path.stem
    fmt = guess_format(str(pdb_path))
    structure_text = pdb_path.read_text(errors="replace")

    # Extract sequences
    chain_seqs = extract_chain_sequences(str(pdb_path))

    # Load predictions
    predictions = load_predictions(jsonl_path, pdb_id)

    # Build position→resi mapping for JS
    pos_to_resi: dict[str, dict[int, int]] = {}
    for c_ids, seq in chain_seqs.items():
        pos_to_resi[c_ids] = {res["pos"]: res["resi"] for res in seq}

    # Build chain colors (for per-chain cartoon) and predictions (for dark-red highlights)
    chain_list = sorted(chain_seqs.keys())
    chain_colors_js: dict[str, str] = {}
    predictions_js: dict[str, list[int]] = {}
    for i, c in enumerate(chain_list):
        chain_colors_js[c] = CHAIN_COLORS[i % len(CHAIN_COLORS)]
        predictions_js[c] = predictions.get(c, [])

    seq_bar_html = build_seq_bar_html(chain_seqs, predictions)
    legend_html = build_legend_html(predictions)

    pos_to_resi_js = json.dumps(pos_to_resi)
    chain_colors_js_str = json.dumps(chain_colors_js)
    predictions_js_str = json.dumps(predictions_js)

    viewer_html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <script src="{cdn_url}"></script>
  <style>
    * {{ box-sizing: border-box; }}
    html, body {{
      margin: 0; padding: 0; width: 100%; height: 840px; background: white;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }}
    #legend-container {{
      padding: 6px 12px;
      border-bottom: 1px solid #e0e0e0;
      background: #fafafa;
    }}
    .legend {{
      display: flex; flex-wrap: wrap; gap: 16px; align-items: center;
      font-size: 13px; color: #333;
    }}
    .legend-item {{
      display: inline-flex; align-items: center; gap: 4px;
    }}
    .chain-dot {{
      display: inline-block; width: 10px; height: 10px; border-radius: 50%;
    }}
    #viewer {{
      width: 100%; height: 560px;
    }}
    #seq-container {{
      border-top: 2px solid #ccc;
      background: #f9f9f9;
      padding: 6px 10px;
    }}
    .chain-label {{
      font-weight: bold; font-size: 13px; margin: 4px 0 2px 0; color: #333;
      display: flex; align-items: center; gap: 5px;
    }}
    .seq-scroll {{
      overflow-x: auto; overflow-y: hidden; margin-bottom: 6px;
    }}
    .seq-inner {{
      display: inline-flex; flex-direction: column;
    }}
    .num-row {{
      display: inline-flex; gap: 1px; height: 16px;
    }}
    .num-cell {{
      width: 16px; flex-shrink: 0; text-align: left;
      font-size: 9px; color: #666; line-height: 14px; padding-left: 0;
    }}
    .aa-row {{
      display: inline-flex; gap: 1px;
    }}
    .aa-cell {{
      width: 16px; height: 20px; flex-shrink: 0;
      display: flex; align-items: center; justify-content: center;
      border: none; background: transparent;
      font-size: 13px; font-weight: bold; color: #333; font-family: monospace;
    }}
  </style>
</head>
<body>
  <div id="legend-container">{legend_html}</div>
  <div id="viewer"></div>
  <div id="seq-container">{seq_bar_html}</div>
  <script>
    var posToResi = {pos_to_resi_js};
    var chainColors = {chain_colors_js_str};
    var predictions = {predictions_js_str};

    // ---- Initialise 3Dmol ----
    try {{
      var structure = {json.dumps(structure_text)};
      var format = {json.dumps(fmt)};
      var v = $3Dmol.createViewer("viewer", {{ backgroundColor: "white" }});

      v.addModel(structure, format);

      // Base cartoon: fixed colour for nucleic acids, then protein chains overridden
      v.setStyle({{}}, {{ cartoon: {{ color: "#b0b0b0" }} }});
      Object.keys(chainColors).forEach(function(ch) {{
        v.setStyle({{ chain: ch }}, {{ cartoon: {{ color: chainColors[ch] }} }});
      }});
      // Small molecules / ligands as stick
      v.setStyle({{ hetflag: true }}, {{ stick: {{ colorscheme: "greenCarbon", radius: 0.15 }} }});
      // Metal ions as spheres
      var metalNames = ["FE","MG","ZN","MN","CU","CO","NI","CA","NA","K","LI","RB","CS","V","CR","MO","W","CD","HG","PB","AL","GA","IN","SB","BI","AS","SE","TE","Y","LA","CE","PR","ND","SM","EU","GD","TB","DY","HO","ER","TM","YB","LU","HF","RE","OS","IR","PT","AU","TL","SN","BA","RA","SR","AG","PD","RH","RU","TC","NB","TA","TI","ZR","SC","BE","GE"];
      v.setStyle({{ resn: metalNames }}, {{ sphere: {{ scale: 0.4 }} }});

      // Highlight predicted sites — uniform dark red for all chains.
      // If no predicted sites are detected across any chain, skip highlighting
      // entirely to avoid showing spurious bright-red markers.
      var totalPreds = 0;
      Object.keys(predictions).forEach(function(ch) {{
        totalPreds += predictions[ch].length;
      }});
      if (totalPreds > 0) {{
        var nucResns = ["DA","DC","DG","DT","A","C","G","T","U","DU","DI"];
        var hlColor = "#e63946";
        Object.keys(predictions).forEach(function(ch) {{
          var preds = predictions[ch];
          var resis = [];
          var posList = [];
          preds.forEach(function(pos) {{
            var resi = posToResi[ch] ? posToResi[ch][pos] : undefined;
            if (resi !== undefined) {{
              resis.push(resi);
              posList.push(pos);
            }}
          }});
          if (resis.length > 0) {{
            v.setStyle(
              {{ chain: ch, resi: resis, not: {{ resn: nucResns }} }},
              {{ cartoon: {{ color: hlColor }} }}
            );
            for (var j = 0; j < resis.length; j++) {{
              var atomSel = {{ chain: ch, resi: resis[j], atom: "CA" }};
              var sel = v.getModel().selectedAtoms(atomSel);
              if (sel.length > 0) {{
                v.addLabel(ch + posList[j], {{
                  position: sel[0],
                  backgroundColor: hlColor,
                  fontColor: "white",
                  fontSize: 8,
                  borderRadius: 2,
                  padding: 1
                }});
              }}
            }}
          }}
        }});
      }}

      v.zoomTo();
      v.render();
      window.viewer3d = v;
    }} catch(e) {{
      document.getElementById("viewer").innerHTML =
        '<p style="color:red;padding:20px;">3Dmol.js failed to load: ' + e.message + '</p>';
    }}
  </script>
</body>
</html>"""

    return f"""<iframe
  srcdoc="{html.escape(viewer_html, quote=True)}"
  style="width:100%; height:840px; border:0; overflow:hidden;"
></iframe>"""


# ═══════════════════════════════════════════════════════════════════════════════
#  Gradio app
# ═══════════════════════════════════════════════════════════════════════════════

def create_app(pdb_path: str, jsonl_path: str, cdn_url: str) -> gr.Blocks:
    pdb_id = Path(pdb_path).stem
    predictions = load_predictions(jsonl_path, pdb_id)

    with gr.Blocks(title=f"Structure Viewer — {pdb_id}") as demo:
        gr.Markdown(
            f"## 🔬 Structure Viewer — `{pdb_id}`\n\n"
            f"**PDB:** `{pdb_path}`  \n"
            f"**Predictions:** `{jsonl_path}`  \n"
            f"**Chains with predictions:** {', '.join(sorted(predictions.keys()))}  \n"
            f"**Total predicted sites:** {sum(len(v) for v in predictions.values())}"
        )
        viewer = gr.HTML(
            value=render_structure_3dmol(pdb_path, jsonl_path, cdn_url)
        )

    return demo


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3Dmol.js structure viewer — parameter-driven.")
    parser.add_argument("-p", "--pdb", type=str, required=True,
                        help="Path to the PDB / mmCIF file.")
    parser.add_argument("-r", "--results", type=str, required=True,
                        help="Path to the prediction JSONL file (output.jsonl).")
    parser.add_argument("--port", type=int, default=7862,
                        help="Server port (default: 7861).")
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio share link.")
    parser.add_argument("--cdn", type=str, default="jsdelivr",
                        choices=["jsdelivr", "3dmol"],
                        help="3Dmol.js CDN: jsdelivr (default, China-friendly), "
                             "3dmol (3Dmol.org).")
    args = parser.parse_args()

    CDN_URLS = {
        "jsdelivr": "https://cdn.jsdelivr.net/npm/3dmol/build/3Dmol-min.js",
        "3dmol": "https://3Dmol.org/build/3Dmol-min.js",
    }
    cdn_url = CDN_URLS[args.cdn]

    demo = create_app(args.pdb, args.results, cdn_url)
    demo.launch(server_name="0.0.0.0", server_port=args.port,
                share=args.share, max_threads=4)
