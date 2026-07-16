from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from spec2mol.heads import MLPHead, MIN_ATOMS, NUM_ATOM_CLASSES
from spec2mol.jdx import SpectraBundle
from spec2mol.utils import add_llm_dmt_to_path, load_py_config, load_yaml, resolve_ckpt_path


class _SuppressLines:
    def __init__(self, *needles: str):
        self.needles = tuple(needles)
        self._stdout = None

    def __enter__(self):
        import sys

        self._stdout = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc, tb):
        import sys

        sys.stdout = self._stdout

    def write(self, text: str) -> int:
        if not any(needle in text for needle in self.needles):
            self._stdout.write(text)
        return len(text)

    def flush(self) -> None:
        self._stdout.flush()


@dataclass
class Spec2MolResult:
    rdkit_mol: Any | None
    processed_mol: Any | None
    atom_count: int
    motif_onehot: list[int]
    motif_probabilities: list[float]
    smiles: str | None
    metadata: dict[str, Any]


class Spec2MolModel:
    def __init__(self, config_path: str | Path, device: str | None = None, ckpt_dir: str | Path | None = None):
        self.config_path = Path(config_path)
        self.cfg = load_yaml(self.config_path)
        if ckpt_dir is not None:
            self.cfg.setdefault("ckpt", {})["dir"] = str(Path(ckpt_dir).expanduser().resolve())
        self.llm_dmt_root = add_llm_dmt_to_path(self.cfg["llm_dmt_root"])
        self.device = torch.device(device or self.cfg["model"].get("device", "cuda:0"))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")

        self._import_llm_dmt()
        self.dmt_config = self._load_dmt_config()
        self.metadata = self._load_metadata()
        self.dmt_model = self._load_dmt_model()
        self.atomcount_head = self._load_atomcount_head()
        self.motif_head = self._load_motif_head()
        self._build_sampler()

    def _import_llm_dmt(self) -> None:
        global create_model, ExponentialMovingAverage, NoiseScheduleVP
        global get_data_inverse_scaler, AncestralSampler, get_self_cond_fn
        global post_process, mol_process, sample_combined_position_feature_noise
        global sample_symmetric_edge_feature_noise, assert_mean_zero_with_mask
        global check_2D_stability, mol2smiles

        from models import create_model  # type: ignore
        from models.ema import ExponentialMovingAverage  # type: ignore
        from diffusion.noise_schedule import NoiseScheduleVP  # type: ignore
        from utils import get_data_inverse_scaler  # type: ignore
        from sampling import (  # type: ignore
            AncestralSampler,
            get_self_cond_fn,
            post_process,
            mol_process,
        )
        from models.utils import (  # type: ignore
            sample_combined_position_feature_noise,
            sample_symmetric_edge_feature_noise,
            assert_mean_zero_with_mask,
        )
        from evaluation.stability import check_2D_stability  # type: ignore
        from evaluation.rdkit_metric import mol2smiles  # type: ignore

    def _load_dmt_config(self):
        config = load_py_config(self.cfg["source"]["config"])
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

    def _load_metadata(self) -> dict[str, Any]:
        import json

        path = resolve_ckpt_path(self.cfg, "metadata")
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _load_safetensors(self, key: str) -> dict[str, torch.Tensor]:
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError("safetensors is required: pip install safetensors") from exc
        return load_file(str(resolve_ckpt_path(self.cfg, key)), device="cpu")

    def _strip_data_parallel(self, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if not state:
            return state
        if all(key.startswith("module.") for key in state):
            return {key[len("module.") :]: value for key, value in state.items()}
        return state

    def _load_dmt_model(self):
        with _SuppressLines("Train SpecFormer from scratch"):
            model = create_model(self.dmt_config)
        state = self._load_safetensors("dmt")
        model.load_state_dict(state, strict=True)
        model.eval()
        return model

    def _load_atomcount_head(self):
        meta = self.metadata.get("atomcount", {})
        args = meta.get("args", {})
        in_dim = int(meta.get("in_dim", args.get("in_dim", 256)))
        hidden = meta.get("hidden", args.get("hidden", [512, 256]))
        dropout = float(meta.get("dropout", args.get("dropout", 0.1)))
        model = MLPHead(in_dim, NUM_ATOM_CLASSES, hidden=hidden, dropout=dropout).to(self.device)
        model.load_state_dict(self._load_safetensors("atomcount"), strict=True)
        model.eval()
        return model

    def _load_motif_head(self):
        meta = self.metadata.get("motif", {})
        args = meta.get("args", {})
        in_dim = int(meta.get("in_dim", args.get("in_dim", 256)))
        num_classes = int(meta.get("num_classes", 20))
        hidden = meta.get("hidden", args.get("hidden", [512, 256]))
        dropout = float(meta.get("dropout", args.get("dropout", 0.1)))
        thresholds = meta.get("thresholds_tuned") or meta.get("thresholds_default") or [0.5] * num_classes
        self.motif_thresholds = torch.tensor(thresholds, dtype=torch.float32, device=self.device)
        self.motif_vocab = meta.get("vocab", [f"motif_{i}" for i in range(num_classes)])
        model = MLPHead(in_dim, num_classes, hidden=hidden, dropout=dropout).to(self.device)
        model.load_state_dict(self._load_safetensors("motif"), strict=True)
        model.eval()
        return model

    def _build_sampler(self) -> None:
        self.noise_scheduler = NoiseScheduleVP(
            self.dmt_config.sde.schedule,
            continuous_beta_0=self.dmt_config.sde.continuous_beta_0,
            continuous_beta_1=self.dmt_config.sde.continuous_beta_1,
        )
        time_steps = torch.linspace(
            self.noise_scheduler.T,
            1e-3,
            int(self.dmt_config.sampling.steps),
            device=self.device,
        )
        self.inverse_scaler = get_data_inverse_scaler(self.dmt_config)
        self.sampler = AncestralSampler(
            self.noise_scheduler,
            time_steps,
            self.dmt_config.model.pred_data,
            self.dmt_config.pred_edge,
            self.dmt_config.model.self_cond,
            get_self_cond_fn(self.dmt_config),
            sampling_temperature=self.dmt_config.eval.sampling_temperature,
        )

    def _context_to_device(self, bundle: SpectraBundle) -> list[torch.Tensor]:
        return [tensor.to(self.device) for tensor in bundle.context]

    @torch.no_grad()
    def encode_spectra(self, bundle: SpectraBundle) -> torch.Tensor:
        context = self._context_to_device(bundle)
        return self._single_device_model().cond_encoder(context)

    def _single_device_model(self):
        return self.dmt_model.module if hasattr(self.dmt_model, "module") else self.dmt_model

    @torch.no_grad()
    def predict_atom_count(self, feat256: torch.Tensor) -> int:
        logits = self.atomcount_head(feat256)
        return int(logits.argmax(dim=-1).item()) + MIN_ATOMS

    @torch.no_grad()
    def predict_motif(self, feat256: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.motif_head(feat256)
        probs = torch.sigmoid(logits)
        motif = (probs >= self.motif_thresholds.view(1, -1)).float()
        return motif, probs

    @torch.no_grad()
    def generate(self, bundle: SpectraBundle) -> Spec2MolResult:
        torch.manual_seed(int(self.cfg["model"].get("seed", 42)))
        context = self._context_to_device(bundle)
        feat256 = self.encode_spectra(bundle)
        atom_count = self.predict_atom_count(feat256)
        motif_onehot, motif_probs = self.predict_motif(feat256)

        node_nf = int(self.dmt_config.data.atom_types) + int(self.dmt_config.model.include_fc_charge)
        edge_nf = int(self.dmt_config.model.edge_ch)
        n_nodes = [atom_count]
        node_mask = torch.ones(1, atom_count, 1, device=self.device)
        edge_mask_square = node_mask.squeeze(-1).unsqueeze(1) * node_mask.squeeze(-1).unsqueeze(2)
        diag_mask = ~torch.eye(atom_count, dtype=torch.bool, device=self.device).unsqueeze(0)
        edge_mask_square = edge_mask_square * diag_mask
        edge_mask = edge_mask_square.reshape(atom_count * atom_count, 1)

        z = sample_combined_position_feature_noise(1, atom_count, node_nf, node_mask)
        assert_mean_zero_with_mask(z[:, :, :3], node_mask)
        edge_z = sample_symmetric_edge_feature_noise(1, atom_count, edge_nf, edge_mask)

        x_node, x_edge = self.sampler.sampling(
            self._single_device_model(),
            z,
            node_mask,
            edge_mask,
            edge_z,
            context,
            motif_onehot.to(self.device),
        )
        pos, one_hot, fc, edge_types = post_process(
            x_node,
            int(self.dmt_config.data.atom_types),
            bool(self.dmt_config.model.include_fc_charge),
            node_mask,
            self.inverse_scaler,
            x_edge,
            edge_mask,
            bool(self.dmt_config.data.compress_edge),
        )
        processed = mol_process(one_hot, pos, fc, n_nodes, edge_types)[0]

        rdkit_mol = None
        smiles = None
        try:
            dataset_info = self._dataset_info()
            pos_i, atom_types_i, edge_types_i, fc_i = processed
            _, _, _, rdkit_mol = check_2D_stability(pos_i, atom_types_i, fc_i, edge_types_i, dataset_info)
            smiles = mol2smiles(rdkit_mol) if rdkit_mol is not None else None
        except Exception:
            rdkit_mol = None
            smiles = None

        motif_list = [int(x) for x in motif_onehot.detach().cpu().reshape(-1).tolist()]
        prob_list = [float(x) for x in motif_probs.detach().cpu().reshape(-1).tolist()]
        meta = {
            "atom_count": atom_count,
            "motif_vocab": self.motif_vocab,
            "motif_active": [self.motif_vocab[i] if i < len(self.motif_vocab) else f"motif_{i}" for i, x in enumerate(motif_list) if x],
            "sampling_steps": int(self.dmt_config.sampling.steps),
            "device": str(self.device),
        }
        return Spec2MolResult(
            rdkit_mol=rdkit_mol,
            processed_mol=processed,
            atom_count=atom_count,
            motif_onehot=motif_list,
            motif_probabilities=prob_list,
            smiles=smiles,
            metadata=meta,
        )

    def _dataset_info(self):
        from datasets.datasets_config import get_dataset_info  # type: ignore

        return get_dataset_info(self.dmt_config.data.info_name)

    def save_sdf(self, result: Spec2MolResult, path: str | Path) -> bool:
        if result.rdkit_mol is None:
            return False
        from rdkit import Chem

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = Chem.SDWriter(str(path))
        writer.write(result.rdkit_mol)
        writer.close()
        return True
