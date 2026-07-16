from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

import spec2mol.model as base_model
from spec2mol.heads import MLPHead, NUM_ATOM_CLASSES
from spec2mol.utils import add_llm_dmt_to_path, load_py_config, load_yaml
from spec2mol.model import Spec2MolModel as _BaseSpec2MolModel
from spec2mol.model import Spec2MolResult
from spec2mol.jdx import SpectraBundle


class Spec2MolModel(_BaseSpec2MolModel):
    """Spec2Mol model with an external motif path for VLM-derived motifs."""

    def _import_llm_dmt(self) -> None:
        """Import the vendored DMT runtime as a package and seed base globals."""
        from spec2mol_v2.llm_dmt_runtime.models import create_model
        from spec2mol_v2.llm_dmt_runtime.models.ema import ExponentialMovingAverage
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

        base_model.create_model = create_model
        base_model.ExponentialMovingAverage = ExponentialMovingAverage
        base_model.NoiseScheduleVP = NoiseScheduleVP
        base_model.get_data_inverse_scaler = get_data_inverse_scaler
        base_model.AncestralSampler = AncestralSampler
        base_model.get_self_cond_fn = get_self_cond_fn
        base_model.post_process = post_process
        base_model.mol_process = mol_process
        base_model.sample_combined_position_feature_noise = sample_combined_position_feature_noise
        base_model.sample_symmetric_edge_feature_noise = sample_symmetric_edge_feature_noise
        base_model.assert_mean_zero_with_mask = assert_mean_zero_with_mask
        base_model.check_2D_stability = check_2D_stability
        base_model.mol2smiles = mol2smiles

    @classmethod
    def from_inline_modules(
        cls,
        config_path: str,
        modules: dict[str, dict[str, Any]],
        device: str | None = None,
    ) -> "Spec2MolModel":
        self = cls.__new__(cls)
        self.config_path = config_path
        self.cfg = load_yaml(config_path)
        runtime_root = str(
            Path(__file__).resolve().parent / "llm_dmt_runtime"
        )
        self.cfg["llm_dmt_root"] = runtime_root
        self.cfg.setdefault("source", {})["config"] = str(Path(__file__).resolve().parent / "dmt_config.py")
        self.llm_dmt_root = add_llm_dmt_to_path(runtime_root)
        self.device = torch.device(device or self.cfg["model"].get("device", "cuda:0"))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")

        self._import_llm_dmt()
        self.dmt_config = self._load_dmt_config_from_inline()
        self.metadata = self._metadata_from_inline(modules)
        self.dmt_model = self._load_dmt_model_from_state(modules["dmt"]["state_dict"])
        self.atomcount_head = self._load_atomcount_head_from_state(
            modules["atomcount"]["state_dict"],
            modules["atomcount"].get("config") or {},
        )
        self.motif_head = self._load_motif_head_from_state(
            modules["motif"]["state_dict"],
            modules["motif"].get("config") or {},
        )
        self._build_sampler()
        return self

    def _load_dmt_config_from_inline(self):
        config = load_py_config(Path(__file__).resolve().parent / "dmt_config.py")
        config.training.distributed = False
        config.training.num_gpus = 1
        config.training.world_size = 1
        config.training.local_rank = 0
        config.device = self.device
        config.data.spectra_version = self.cfg["model"].get("spectra_version", "allspectra")
        config.sampling.steps = int(self.cfg["model"].get("sampling_steps", config.sampling.steps))
        config.eval.batch_size = 1
        config.eval.num_samples = 1
        config.eval.sampling_temperature = float(self.cfg["model"].get("sampling_temperature", 1.0))
        config.eval.motif_drop_eval = False
        return config

    def _metadata_from_inline(self, modules: dict[str, dict[str, Any]]) -> dict[str, Any]:
        atom_cfg = modules["atomcount"].get("config") or {}
        atom_build = atom_cfg.get("build_args") or {}
        motif_cfg = modules["motif"].get("config") or {}
        motif_build = motif_cfg.get("build_args") or {}
        num_classes = int(motif_build.get("num_classes", 20))
        return {
            "atomcount": {
                "in_dim": int(atom_build.get("in_dim", 256)),
                "hidden": atom_build.get("hidden", [512, 256]),
                "dropout": float(atom_build.get("dropout", 0.1)),
            },
            "motif": {
                "in_dim": int(motif_build.get("in_dim", 256)),
                "num_classes": num_classes,
                "hidden": motif_build.get("hidden", [512, 256]),
                "dropout": float(motif_build.get("dropout", 0.1)),
                "vocab": motif_cfg.get("vocab") or motif_build.get("vocab") or [f"motif_{i}" for i in range(num_classes)],
                "thresholds_default": motif_build.get("thresholds_default") or [0.5] * num_classes,
                "thresholds_tuned": motif_build.get("thresholds_tuned") or motif_cfg.get("thresholds") or [],
            },
        }

    def _load_dmt_model_from_state(self, state: dict[str, torch.Tensor]):
        with base_model._SuppressLines("Train SpecFormer from scratch"):
            model = base_model.create_model(self.dmt_config)
        state = {key: value.detach().cpu().contiguous() for key, value in state.items()}
        model.load_state_dict(state, strict=True)
        model.eval()
        return model

    def _load_atomcount_head_from_state(self, state: dict[str, torch.Tensor], config: dict[str, Any]):
        build = dict(config.get("build_args") or {})
        in_dim = int(build.get("in_dim", 256))
        hidden = build.get("hidden", [512, 256])
        dropout = float(build.get("dropout", 0.1))
        model = MLPHead(in_dim, NUM_ATOM_CLASSES, hidden=hidden, dropout=dropout).to(self.device)
        model.load_state_dict(state, strict=True)
        model.eval()
        return model

    def _load_motif_head_from_state(self, state: dict[str, torch.Tensor], config: dict[str, Any]):
        build = dict(config.get("build_args") or {})
        in_dim = int(build.get("in_dim", 256))
        num_classes = int(build.get("num_classes", 20))
        hidden = build.get("hidden", [512, 256])
        dropout = float(build.get("dropout", 0.1))
        thresholds = config.get("thresholds") or build.get("thresholds_tuned") or build.get("thresholds_default") or [0.5] * num_classes
        self.motif_thresholds = torch.tensor(thresholds, dtype=torch.float32, device=self.device)
        self.motif_vocab = config.get("vocab") or build.get("vocab") or [f"motif_{i}" for i in range(num_classes)]
        model = MLPHead(in_dim, num_classes, hidden=hidden, dropout=dropout).to(self.device)
        model.load_state_dict(state, strict=True)
        model.eval()
        return model

    def _dataset_info(self):
        from spec2mol_v2.llm_dmt_runtime.datasets.datasets_config import get_dataset_info

        return get_dataset_info(self.dmt_config.data.info_name)

    def _prepare_external_motif(self, motif_onehot: torch.Tensor | list[int] | list[float]) -> torch.Tensor:
        motif = torch.as_tensor(motif_onehot, dtype=torch.float32, device=self.device)
        if motif.ndim == 1:
            motif = motif.view(1, -1)
        if motif.ndim != 2 or motif.shape[0] != 1:
            raise ValueError(f"expected motif shape [20] or [1,20], got {tuple(motif.shape)}")
        expected = len(getattr(self, "motif_vocab", [])) or int(self.metadata.get("motif", {}).get("num_classes", 20))
        if motif.shape[1] != expected:
            raise ValueError(f"expected motif width {expected}, got {motif.shape[1]}")
        return motif

    def _prob_list(self, motif_probabilities: torch.Tensor | list[float] | None, width: int) -> list[float]:
        if motif_probabilities is None:
            return [0.0] * width
        probs = torch.as_tensor(motif_probabilities, dtype=torch.float32)
        return [float(x) for x in probs.detach().cpu().reshape(-1).tolist()]

    @torch.no_grad()
    def generate_with_external_motif(
        self,
        bundle: SpectraBundle,
        motif_onehot: torch.Tensor | list[int] | list[float],
        motif_probabilities: torch.Tensor | list[float] | None = None,
        route_metadata: dict[str, Any] | None = None,
    ) -> Spec2MolResult:
        """Generate a molecule using spcformer context/atom count and external motif bits."""

        torch.manual_seed(int(self.cfg["model"].get("seed", 42)))
        context = self._context_to_device(bundle)
        feat256 = self.encode_spectra(bundle)
        atom_count = self.predict_atom_count(feat256)
        motif = self._prepare_external_motif(motif_onehot)

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
            self._single_device_model(),
            z,
            node_mask,
            edge_mask,
            edge_z,
            context,
            motif.to(self.device),
        )
        pos, one_hot, fc, edge_types = base_model.post_process(
            x_node,
            int(self.dmt_config.data.atom_types),
            bool(self.dmt_config.model.include_fc_charge),
            node_mask,
            self.inverse_scaler,
            x_edge,
            edge_mask,
            bool(self.dmt_config.data.compress_edge),
        )
        processed = base_model.mol_process(one_hot, pos, fc, n_nodes, edge_types)[0]

        rdkit_mol = None
        smiles = None
        try:
            dataset_info = self._dataset_info()
            pos_i, atom_types_i, edge_types_i, fc_i = processed
            _, _, _, rdkit_mol = base_model.check_2D_stability(pos_i, atom_types_i, fc_i, edge_types_i, dataset_info)
            smiles = base_model.mol2smiles(rdkit_mol) if rdkit_mol is not None else None
        except Exception:
            rdkit_mol = None
            smiles = None

        motif_list = [int(x) for x in motif.detach().cpu().reshape(-1).tolist()]
        prob_list = self._prob_list(motif_probabilities, len(motif_list))
        meta = {
            "atom_count": atom_count,
            "motif_source": "s1_omni_linear_present_modalities",
            "motif_fusion": (route_metadata or {}).get("motif_fusion", "onehot_intersection"),
            "external_motif": True,
            "specformer_motif_head_used": False,
            "motif_vocab": self.motif_vocab,
            "motif_active": [
                self.motif_vocab[i] if i < len(self.motif_vocab) else f"motif_{i}"
                for i, x in enumerate(motif_list)
                if x
            ],
            "sampling_steps": int(self.dmt_config.sampling.steps),
            "device": str(self.device),
        }
        if route_metadata:
            meta["route_metadata"] = route_metadata

        return Spec2MolResult(
            rdkit_mol=rdkit_mol,
            processed_mol=processed,
            atom_count=atom_count,
            motif_onehot=motif_list,
            motif_probabilities=prob_list,
            smiles=smiles,
            metadata=meta,
        )
