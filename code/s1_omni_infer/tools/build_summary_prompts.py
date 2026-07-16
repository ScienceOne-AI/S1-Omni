"""
Build summary prompts from LLM-predicted protein binding site outputs (no LLM call).

Reads LLM_output.jsonl, extracts the assistant's answer (excluding <think>...</think>),
and constructs the final system + user prompts that would be sent to the LLM.
Outputs each protein's constructed prompt to a separate .txt file and a combined JSONL.

Usage:
    python build_summary_prompts.py --input_file /path/to/LLM_output.jsonl [--output_dir /path/to/output]
"""

import json
import os
import re
import argparse
from typing import Dict, List


# ============================================================
# Prompt templates (mirrored from summarize_protein_sites.py)
# ============================================================

SUMMARY_SYSTEM_PROMPT = """You are an expert computational biologist specializing in protein function prediction, structural biology, and bioinformatics. Your task is to summarize predicted protein binding sites from multiple chain predictions into a clear, well-structured report.

Rules:
1. Group results by protein (e.g., PDB ID), then analyze each chain in alphabetical order (A, B, C, ...).
2. First, provide a detailed narrative analysis of each chain's prediction in alphabetical chain order.
3. Compare predictions across chains: note consistencies, discrepancies, and patterns in the predicted binding interfaces.
4. After the analysis, for quick reference, format all explicitly predicted residue positions in a compact table.
5. Only report residues that were explicitly listed as predicted binding sites.

Produce the summary with the following structure:

## Summary for [Protein ID]

### Detailed Analysis
Carefully review and analyze the prediction from each chain provided above, proceeding in alphabetical order by chain ID (A → B → C → ...). For each chain, discuss:
- The key reasoning and evidence cited in its prediction.
- The specific residues identified and the rationale behind each one.
- Biological plausibility of the predictions based on the provided reasoning.

This should be a thorough narrative analysis, not just a list of residues.

### Per-chain predictions (Table)
After the analysis, for quick reference, list the explicitly predicted binding-site residues in a compact table format, also in alphabetical chain order. Each chain's positions are reported as-is from its own prediction.

### Overall assessment
A 2-3 sentence takeaway about the predicted binding site, highlighting the most confident or biologically significant findings."""


SUMMARY_USER_PROMPT_TEMPLATE = """Below are the predicted outputs from multiple chains of the same protein complex. Each entry shows the chain ID followed by the prediction.
---
{chain_predictions}
---
Please produce a consolidated summary for this protein complex."""


# ============================================================
# Data processing (mirrored from summarize_protein_sites.py)
# ============================================================

def strip_think_tags(content: str) -> str:
    """Remove ...</think> block from assistant content."""
    if not content:
        return ""
    cleaned = re.sub(r".*?</think>", "", content, flags=re.DOTALL)
    cleaned = re.sub(r"The actual residue index of the site is:\s*<prot_cla>", "", cleaned)
    cleaned = cleaned.strip()
    return cleaned


def extract_protein_id(entry_id: str) -> str:
    """Extract protein base ID from chain-level ID, e.g. '5DY0_C' -> '5DY0'."""
    match = re.match(r"^([A-Za-z0-9]+)_([A-Za-z0-9]+)$", entry_id)
    if match:
        return match.group(1)
    return entry_id


def extract_chain_id(entry_id: str) -> str:
    """Extract chain ID from entry ID, e.g. '5DY0_C' -> 'C'."""
    match = re.match(r"^[A-Za-z0-9]+_([A-Za-z0-9]+)$", entry_id)
    if match:
        return match.group(1)
    return entry_id


def load_and_prepare(input_file: str) -> Dict[str, List[Dict]]:
    """Load the JSONL file, strip thinking, and group by protein ID.

    Returns:
        {protein_id: [{"chain": "C", "content": "...cleaned answer..."}, ...]}
    """
    grouped: Dict[str, List[Dict]] = {}

    with open(input_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: skipping line {line_no}: {e}")
                continue

            entry_id = entry.get("id", f"line_{line_no}")
            messages = entry.get("messages", [])

            # Find the assistant message
            assistant_content = ""
            for msg in messages:
                if msg.get("role") == "assistant":
                    assistant_content = msg.get("content", "")
                    break

            if not assistant_content:
                print(f"Warning: no assistant message for {entry_id}")
                continue

            cleaned = strip_think_tags(assistant_content)
            if not cleaned:
                print(f"Warning: empty content after stripping think for {entry_id}")
                continue

            protein_id = extract_protein_id(entry_id)
            chain_id = extract_chain_id(entry_id)

            if protein_id not in grouped:
                grouped[protein_id] = []
            grouped[protein_id].append({
                "chain": chain_id,
                "entry_id": entry_id,
                "content": cleaned,
            })

    return grouped


def build_chain_predictions_text(chains: List[Dict], protein_id: str) -> str:
    """Build the formatted text of chain predictions for a single protein."""
    parts = [f"### Protein: {protein_id}"]
    # Sort chains alphabetically
    for chain_info in sorted(chains, key=lambda c: c["chain"]):
        parts.append(f"\n**Chain {chain_info['chain']}** ({chain_info['entry_id']}):\n")
        parts.append(chain_info["content"])
        parts.append("\n---\n")
    return "\n".join(parts)


def construct_prompt(protein_id: str, chains: List[Dict]) -> Dict:
    """Construct the full prompt (system + user) for one protein.

    Returns a dict with keys: protein_id, system_prompt, user_prompt, num_chains
    """
    chain_text = build_chain_predictions_text(chains, protein_id)
    user_prompt = SUMMARY_USER_PROMPT_TEMPLATE.format(chain_predictions=chain_text)

    return {
        "protein_id": protein_id,
        "system_prompt": SUMMARY_SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "num_chains": len(chains),
    }


# ============================================================
# Output
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build summary prompts from LLM_output.jsonl (no LLM call)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to LLM_output.jsonl")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: same dir as input_file)")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: input file not found: {args.input_file}")
        return

    # Resolve output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.input_file) or "."
    os.makedirs(args.output_dir, exist_ok=True)

    # Load and prepare data
    print(f"Loading: {args.input_file}")
    grouped = load_and_prepare(args.input_file)

    if not grouped:
        print("Error: no valid entries found")
        return

    # Print overview
    print("\n" + "=" * 70)
    print("Input overview")
    print("=" * 70)
    for protein_id, chains in grouped.items():
        chain_ids = sorted([c["chain"] for c in chains])
        print(f"\nProtein: {protein_id}")
        print(f"  Chains ({len(chains)}): {', '.join(chain_ids)}")
        for c in sorted(chains, key=lambda x: x["chain"]):
            print(f"    Chain {c['chain']}: {len(c['content'])} chars of answer text")

    # Construct prompts for each protein
    print(f"\n{'=' * 70}")
    print(f"Constructing prompts for {len(grouped)} protein(s)...")
    print(f"{'=' * 70}")

    all_prompts = []  # for combined JSONL output

    for protein_id, chains in grouped.items():
        prompt = construct_prompt(protein_id, chains)

        # Write individual .txt file
        txt_path = os.path.join(args.output_dir, f"{protein_id}_prompt.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write(f"Protein: {protein_id}  |  Chains: {prompt['num_chains']}\n")
            f.write("=" * 70 + "\n\n")
            f.write("=== SYSTEM PROMPT ===\n\n")
            f.write(prompt["system_prompt"])
            f.write("\n\n=== USER PROMPT ===\n\n")
            f.write(prompt["user_prompt"])

        print(f"  {protein_id}: {prompt['num_chains']} chains -> {txt_path}")

        all_prompts.append(prompt)

    # Write combined JSONL
    jsonl_path = os.path.join(args.output_dir, "all_prompts.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for p in all_prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\n{'=' * 70}")
    print(f"Done: {len(all_prompts)} protein(s)")
    print(f"Individual prompts: {args.output_dir}/<protein_id>_prompt.txt")
    print(f"Combined JSONL:      {jsonl_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
