#!/usr/bin/env python3
"""End-to-end inference reading from the fused S1-VL + spec2mol directory.

Pipeline (mirrors infer_merge.py but loads everything from one fused dir):
  JDX -> render IR/Raman/UV PNG -> fused VLM route
  -> if <spectra_st> -> fused VLM IR vision hidden -> spectra_omni motif predictor
  -> fused DMT diffusion (external motif) + spcformer context/feat256/atom_count -> SDF
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from spec2mol_v2.fused_runtime import (
    FusedVLMRuntime,
    EXPECTED_VISION_SPAN,
    load_head,
    load_small_modules_config_with_source,
    load_spectra_omni,
    load_dmt,
)
from spec2mol_v2.spectra_omni_predictor import _resolve_thresholds  # noqa: F401
from spec2mol_v2.spectra_plot import render_spectra_images

DEFAULT_FUSED_MODEL_DIR = ""
DEFAULT_INPUT_DIR = str(PROJECT_ROOT / "spec2mol" / "dataset" / "test_data" / "000010")


class FusedSpec2MolModel:
    """Spec2Mol generation using DMT/atomcount loaded from the fused directory.

    Reuses the sampling / post-processing logic of spec2mol.model.Spec2MolModel
    but takes pre-built DMT model + atomcount head instead of loading from a
    ckpt dir. The spcformer context/feat256/atom_count path is identical.
    """

    def __init__(self, dmt_model, dmt_config, atomcount_head, device, sampling_steps: int = 1000,
                 sampling_temperature: float = 1.0, seed: int = 42):
        import spec2mol.model as base_model

        self.base_model = base_model
        self.dmt_model = dmt_model
        self.dmt_config = dmt_config
        self.atomcount_head = atomcount_head
        self.device = torch.device(device)
        self.seed = seed

        # Configure sampling params (mirrors Spec2MolModel._load_dmt_config / _build_sampler).
        self.dmt_config.training.distributed = False
        self.dmt_config.training.num_gpus = 1
        self.dmt_config.training.world_size = 1
        self.dmt_config.training.local_rank = 0
        self.dmt_config.device = self.device
        self.dmt_config.sampling.steps = int(sampling_steps)
        self.dmt_config.eval.batch_size = 1
        self.dmt_config.eval.num_samples = 1
        self.dmt_config.eval.sampling_temperature = float(sampling_temperature)
        self.dmt_config.eval.motif_drop_eval = False

        self._build_sampler()
        # motif vocab (for metadata only; external motif comes from predictor)
        self.motif_vocab = [f"motif_{i}" for i in range(20)]

    def _import_llm_dmt(self):
        global create_model, NoiseScheduleVP, get_data_inverse_scaler, AncestralSampler, get_self_cond_fn
        global post_process, mol_process, sample_combined_position_feature_noise
        global sample_symmetric_edge_feature_noise, assert_mean_zero_with_mask
        global check_2D_stability, mol2smiles
        from spec2mol_v2.llm_dmt_runtime.models import create_model
        from spec2mol_v2.llm_dmt_runtime.diffusion.noise_schedule import NoiseScheduleVP
        from spec2mol_v2.llm_dmt_runtime.utils import get_data_inverse_scaler
        from spec2mol_v2.llm_dmt_runtime.sampling import (
            AncestralSampler,
            get_self_cond_fn,
            post_process,
            mol_process,
        )
        from spec2mol_v2.llm_dmt_runtime.models.utils import (
            sample_combined_position_feature_noise,
            sample_symmetric_edge_feature_noise,
            assert_mean_zero_with_mask,
        )
        from spec2mol_v2.llm_dmt_runtime.evaluation.stability import check_2D_stability
        from spec2mol_v2.llm_dmt_runtime.evaluation.rdkit_metric import mol2smiles

        # Also inject into spec2mol.model globals so base_model.* resolves (mirrors
        # Spec2MolModel._import_llm_dmt which sets them as module-level globals).
        import spec2mol.model as _base
        _base.sample_combined_position_feature_noise = sample_combined_position_feature_noise
        _base.assert_mean_zero_with_mask = assert_mean_zero_with_mask
        _base.sample_symmetric_edge_feature_noise = sample_symmetric_edge_feature_noise
        _base.post_process = post_process
        _base.mol_process = mol_process
        _base.check_2D_stability = check_2D_stability
        _base.mol2smiles = mol2smiles

    def _build_sampler(self):
        self._import_llm_dmt()
        self.noise_scheduler = NoiseScheduleVP(
            self.dmt_config.sde.schedule,
            continuous_beta_0=self.dmt_config.sde.continuous_beta_0,
            continuous_beta_1=self.dmt_config.sde.continuous_beta_1,
        )
        time_steps = torch.linspace(
            self.noise_scheduler.T, 1e-3, int(self.dmt_config.sampling.steps), device=self.device
        )
        self.inverse_scaler = get_data_inverse_scaler(self.dmt_config)
        self.sampler = AncestralSampler(
            self.noise_scheduler, time_steps,
            self.dmt_config.model.pred_data, self.dmt_config.pred_edge,
            self.dmt_config.model.self_cond, get_self_cond_fn(self.dmt_config),
            sampling_temperature=self.dmt_config.eval.sampling_temperature,
        )

    def _context_to_device(self, bundle):
        return [t.to(self.device) for t in bundle.context]

    @torch.no_grad()
    def encode_spectra(self, bundle) -> torch.Tensor:
        context = self._context_to_device(bundle)
        return self._single_device_model().cond_encoder(context)

    def _single_device_model(self):
        return self.dmt_model.module if hasattr(self.dmt_model, "module") else self.dmt_model

    @torch.no_grad()
    def predict_atom_count(self, feat256: torch.Tensor) -> int:
        from spec2mol.heads import MIN_ATOMS
        logits = self.atomcount_head(feat256)
        return int(logits.argmax(dim=-1).item()) + MIN_ATOMS

    @torch.no_grad()
    def generate_with_external_motif(self, bundle, motif_onehot, motif_probabilities=None,
                                     route_metadata=None):
        base_model = self.base_model
        torch.manual_seed(int(self.seed))
        context = self._context_to_device(bundle)
        feat256 = self.encode_spectra(bundle)
        atom_count = self.predict_atom_count(feat256)

        motif = torch.as_tensor(motif_onehot, dtype=torch.float32, device=self.device)
        if motif.ndim == 1:
            motif = motif.view(1, -1)

        node_nf = int(self.dmt_config.data.atom_types) + int(self.dmt_config.model.include_fc_charge)
        edge_nf = int(self.dmt_config.model.edge_ch)
        n_nodes = [atom_count]
        node_mask = torch.ones(1, atom_count, 1, device=self.device)
        edge_mask_square = node_mask.squeeze(-1).unsqueeze(1) * node_mask.squeeze(-1).unsqueeze(2)
        diag_mask = ~torch.eye(atom_count, dtype=torch.bool, device=self.device).unsqueeze(0)
        edge_mask_square = edge_mask_square * diag_mask
        edge_mask = edge_mask_square.reshape(atom_count * atom_count, 1)

        z = base_model.sample_combined_position_feature_noise(1, atom_count, node_nf, node_mask)
        base_model.assert_mean_zero_with_mask(z[:, :, :3], node_mask)
        edge_z = base_model.sample_symmetric_edge_feature_noise(1, atom_count, edge_nf, edge_mask)

        x_node, x_edge = self.sampler.sampling(
            self._single_device_model(), z, node_mask, edge_mask, edge_z, context, motif.to(self.device)
        )
        pos, one_hot, fc, edge_types = base_model.post_process(
            x_node, int(self.dmt_config.data.atom_types),
            bool(self.dmt_config.model.include_fc_charge), node_mask, self.inverse_scaler,
            x_edge, edge_mask, bool(self.dmt_config.data.compress_edge),
        )
        processed = base_model.mol_process(one_hot, pos, fc, n_nodes, edge_types)[0]

        rdkit_mol = None
        smiles = None
        try:
            from spec2mol_v2.llm_dmt_runtime.datasets.datasets_config import get_dataset_info
            dataset_info = get_dataset_info(self.dmt_config.data.info_name)
            pos_i, atom_types_i, edge_types_i, fc_i = processed
            _, _, _, rdkit_mol = base_model.check_2D_stability(pos_i, atom_types_i, fc_i, edge_types_i, dataset_info)
            smiles = base_model.mol2smiles(rdkit_mol) if rdkit_mol is not None else None
        except Exception:  # noqa: BLE001
            rdkit_mol = None
            smiles = None

        motif_list = [int(x) for x in motif.detach().cpu().reshape(-1).tolist()]
        prob_list = [float(x) for x in (motif_probabilities or [0.0] * len(motif_list))]
        meta = {
            "atom_count": atom_count,
            "motif_source": "spectra_omni_ir_vlm_hidden",
            "external_motif": True,
            "specformer_motif_head_used": False,
            "motif_vocab": self.motif_vocab,
            "motif_active": [self.motif_vocab[i] if i < len(self.motif_vocab) else f"motif_{i}"
                             for i, x in enumerate(motif_list) if x],
            "sampling_steps": int(self.dmt_config.sampling.steps),
            "device": str(self.device),
        }
        if route_metadata:
            meta["route_metadata"] = route_metadata

        return {
            "rdkit_mol": rdkit_mol,
            "processed_mol": processed,
            "atom_count": atom_count,
            "motif_onehot": motif_list,
            "motif_probabilities": prob_list,
            "smiles": smiles,
            "metadata": meta,
        }

    def save_sdf(self, result, path) -> bool:
        if result["rdkit_mol"] is None:
            return False
        from rdkit import Chem
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = Chem.SDWriter(str(path))
        writer.write(result["rdkit_mol"])
        writer.close()
        return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fused_model_dir", default=DEFAULT_FUSED_MODEL_DIR)
    p.add_argument("--input_dir", default=DEFAULT_INPUT_DIR)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--attn_implementation", default=None)
    p.add_argument("--max_new_tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_image_side", type=int, default=None)
    p.add_argument("--force_spectra", action="store_true")
    p.add_argument("--skip_vlm_generate", action="store_true")
    p.add_argument("--sampling_steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fused_dir = Path(args.fused_model_dir).expanduser().resolve()

    sample_json = output_dir / "sample.json"
    sdf_path = output_dir / "sample.sdf"

    record: dict[str, Any] = {
        "fused_model_dir": str(fused_dir),
        "input_dir": str(Path(args.input_dir).resolve()),
        "output_dir": str(output_dir),
        "sdf_path": str(sdf_path),
        "ok": False,
        "failure_stage": None,
        "error": None,
        "traceback": None,
    }

    try:
        sm_cfg, sm_cfg_source = load_small_modules_config_with_source(fused_dir)
        record["fused_format"] = sm_cfg.get("format")
        record["small_modules"] = {n: e["prefix"] for n, e in sm_cfg["modules"].items()}
        record["small_modules_config_source"] = sm_cfg_source
        record["dmt_runtime_source"] = "spec2mol_v2.llm_dmt_runtime"
        record["dmt_config_source"] = "spec2mol_v2.dmt_config"

        model_index_path = fused_dir / "model.safetensors.index.json"
        if not model_index_path.is_file():
            raise FileNotFoundError(f"missing {model_index_path}")
        model_index = json.loads(model_index_path.read_text(encoding="utf-8"))
        weight_map = model_index.get("weight_map", {})
        record["model_index"] = {
            "path": str(model_index_path),
            "metadata": model_index.get("metadata", {}),
            "weight_map_entries": len(weight_map),
            "small_module_prefix_counts": {
                name: sum(1 for key in weight_map if key.startswith(entry["prefix"]))
                for name, entry in sm_cfg["modules"].items()
            },
        }

        # --- JDX + PNG ---
        from spec2mol_v2.jdx import bundle_to_jsonable, load_spectra_dir

        bundle = load_spectra_dir(args.input_dir)
        record["present_spectra"] = bundle.present_spectra
        record["zero_filled_spectra"] = bundle.zero_filled_spectra
        record["spectra_files"] = bundle_to_jsonable(bundle)["files"]

        spectra_dir = output_dir / "spectra_images"
        image_meta = render_spectra_images(args.input_dir, spectra_dir, max_image_side=args.max_image_side)
        record["spectra_images"] = image_meta.get("image_files", {})
        ir_png, raman_png, uv_png = spectra_dir / "ir.png", spectra_dir / "raman.png", spectra_dir / "uv.png"

        # --- VLM route ---
        if args.skip_vlm_generate:
            route_info = {"skipped": True, "has_spectra_token": True}
            record["generated_text"] = None
            record["has_spectra_token"] = True
        else:
            vlm = FusedVLMRuntime(fused_dir, device=args.device, dtype=args.dtype,
                                  attn_implementation=args.attn_implementation)
            route = vlm.generate_route(ir_png, raman_png, uv_png,
                                       max_new_tokens=args.max_new_tokens,
                                       temperature=args.temperature, top_p=args.top_p)
            (output_dir / "vlm_generate.txt").write_text(route["generated_text"], encoding="utf-8")
            write_json(output_dir / "route.json", route)
            route_info = route
            record["generated_text"] = route["generated_text"]
            record["has_spectra_token"] = route["has_spectra_token"]

        write_json(sample_json, record)

        hit = bool(route_info.get("has_spectra_token")) or args.force_spectra
        if not hit:
            record["failure_stage"] = "route_no_spectra"
            record["error"] = "VLM route did not emit <spectra_st>; skipping motif + DMT."
            write_json(sample_json, record)
            print(json.dumps({"ok": False, "failure_stage": record["failure_stage"]}, indent=2))
            return 0

        # --- IR hidden ---
        if args.skip_vlm_generate:
            vlm = FusedVLMRuntime(fused_dir, device=args.device, dtype=args.dtype,
                                  attn_implementation=args.attn_implementation)
        hidden = vlm.extract_ir_vision_hidden(ir_png)
        record["ir_vision_span_shape"] = hidden["ir_vision_span_shape"]
        record["image_grid_thw"] = hidden["image_grid_thw"]
        if not hidden["ok"]:
            record["failure_stage"] = "vision_span_shape"
            record["error"] = hidden["error"] or "vision span shape mismatch"
            write_json(sample_json, record)
            print(json.dumps({"ok": False, "failure_stage": record["failure_stage"]}, indent=2))
            return 0

        # --- motif predictor (from fused) ---
        extractor = load_spectra_omni(fused_dir, sm_cfg["modules"]["spectra_omni"], device=args.device)
        ext_dtype = next(extractor.parameters()).dtype
        vision_span = hidden["ir_vision_span"].to(device=args.device, dtype=ext_dtype)
        with torch.no_grad():
            logits, _ = extractor(vision_span)
            probs = torch.sigmoid(logits.float())
        spectra_omni_cfg = sm_cfg["modules"]["spectra_omni"]
        thresholds = spectra_omni_cfg.get("thresholds") or [0.5] * spectra_omni_cfg["num_classes"]
        thr_tensor = torch.tensor(thresholds, dtype=probs.dtype, device=probs.device)
        onehot = (probs >= thr_tensor.view(1, -1)).to(probs.dtype)
        probs_list = [float(v) for v in probs.reshape(-1).detach().cpu().tolist()]
        onehot_list = [int(v) for v in onehot.reshape(-1).detach().cpu().tolist()]
        vocab = spectra_omni_cfg["vocab"]
        active = [vocab[i] if i < len(vocab) else f"motif_{i}" for i, x in enumerate(onehot_list) if x]
        record["motif_vocab"] = vocab
        record["motif_probabilities"] = probs_list
        record["motif_onehot"] = onehot_list
        record["motif_active"] = active
        record["motif_threshold_source"] = spectra_omni_cfg.get("threshold_source", "0.5")

        # --- DMT (from fused) ---
        dmt_model, dmt_config = load_dmt(fused_dir, sm_cfg["modules"]["dmt"], device=args.device)
        atomcount_head = load_head(fused_dir, "atomcount", sm_cfg["modules"]["atomcount"],
                                   device=args.device, out_dim=27)
        spec_model = FusedSpec2MolModel(
            dmt_model, dmt_config, atomcount_head, device=args.device,
            sampling_steps=args.sampling_steps, seed=args.seed,
        )
        result = spec_model.generate_with_external_motif(
            bundle=bundle, motif_onehot=onehot_list, motif_probabilities=probs_list,
            route_metadata=route_info,
        )
        record["atom_count"] = result["atom_count"]
        record["motif_onehot"] = result["motif_onehot"]
        record["motif_probabilities"] = result["motif_probabilities"]
        record["smiles"] = result["smiles"]
        record["motif_metadata"] = result["metadata"]

        sdf_written = spec_model.save_sdf(result, sdf_path)
        record["sdf_written"] = sdf_written
        record["ok"] = sdf_written
        if not sdf_written:
            record["failure_stage"] = "sdf_write"
            record["error"] = "RDKit molecule was not produced"

        write_json(sample_json, record)
        print(json.dumps({"ok": record["ok"], "sdf_path": str(sdf_path), "smiles": result["smiles"]}, indent=2))
        return 0 if record["ok"] else 1

    except Exception as exc:  # noqa: BLE001
        record["failure_stage"] = record.get("failure_stage") or "inference"
        record["error"] = str(exc)
        record["traceback"] = traceback.format_exc()
        write_json(sample_json, record)
        print(json.dumps({"ok": False, "failure_stage": record["failure_stage"], "error": str(exc)}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
