"""Self-contained DMT config for fused spec2mol inference."""
from __future__ import annotations

import torch


def _config_dict():
    try:
        import ml_collections
        return ml_collections.ConfigDict()
    except Exception:
        class ConfigDict(dict):
            def __getattr__(self, name):
                try:
                    return self[name]
                except KeyError as exc:
                    raise AttributeError(name) from exc

            def __setattr__(self, name, value):
                self[name] = value

        return ConfigDict()


def get_config(device: str | torch.device | None = None):
    config = _config_dict()

    config.exp_type = "diffspectra"
    config.pred_edge = True
    config.only_2D = False

    config.data = data = _config_dict()
    data.root = ""
    data.name = "qm9s_motif"
    data.processed_file = ""
    data.info_name = "qm9_second_half"
    data.num_workers = 0
    data.motif_onehot_path = ""
    data.motif_vocab_csv = ""
    data.compress_edge = True
    data.centered = True
    data.include_aromatic = False
    data.atom_types = 5
    data.bond_types = 4
    data.fc_scale = [-1.0, 1.0]
    data.max_node = 29
    data.spectra_version = "allspectra"
    data.aug_translation_scale = 0.1
    data.transform = "EdgeComSpectra"
    data.use_normalize = True

    config.sde = sde = _config_dict()
    sde.schedule = "cosine"
    sde.continuous_beta_0 = 0.1
    sde.continuous_beta_1 = 20.0

    config.model = model = _config_dict()
    model.name = "MAST_DMT"
    model.pred_data = True
    model.include_fc_charge = True
    model.normalize_factors = "1, 4, 4, 1"
    model.ema_decay = 0.999
    model.edge_ch = 2
    model.nf = 256
    model.n_layers = 8
    model.n_heads = 16
    model.dropout = 0.1
    model.cond_time = True
    model.dist_gbf = True
    model.gbf_name = "CondGaussianLayer"
    model.self_cond = True
    model.self_cond_type = "ori"
    model.edge_quan_th = 0.0
    model.n_extra_heads = 2
    model.CoM = True
    model.mlp_ratio = 2
    model.spatial_cut_off = 2.0
    model.softmax_inf = True
    model.trans_name = "TransMixLayer"
    model.cond_ch = 1
    model.pretrained_specformer_path = ""
    model.pretrained_checkpoint_path = ""
    model.patch_len = [20, 50, 50]
    model.stride = [10, 25, 25]
    model.motif_dim = 20
    model.motif_hidden_dim = 256
    model.motif_dropout = 0.1
    model.motif_drop_prob = 0.0
    model.motif_drop_ratio_min = 0.2
    model.motif_drop_ratio_max = 0.8
    model.loss_weights = "1., 0.25, 0.1"
    model.noise_align = True

    config.training = training = _config_dict()
    training.distributed = False
    training.num_gpus = 1
    training.world_size = 1
    training.local_rank = 0
    training.batch_size = 1
    training.eval_batch_size = 1
    training.eval_samples = 1
    training.reduce_mean = False
    training.log_freq = 500
    training.n_iters = 2000000
    training.snapshot_freq = 50000
    training.snapshot_freq_for_preemption = 10000
    training.snapshot_sampling = False
    training.dataloader_drop_last = False

    config.optim = optim = _config_dict()
    optim.weight_decay = 0
    optim.optimizer = "AdamW"
    optim.lr = 2e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 100000
    optim.grad_clip = 10.0
    optim.disable_grad_log = True

    config.sampling = sampling = _config_dict()
    sampling.method = "ancestral"
    sampling.steps = 1000
    sampling.vis_row = 4
    sampling.vis_col = 4

    config.eval = evaluate = _config_dict()
    evaluate.enable_sampling = True
    evaluate.batch_size = 1
    evaluate.num_samples = 1
    evaluate.begin_ckpt = 40
    evaluate.end_ckpt = 40
    evaluate.ckpts = ""
    evaluate.condition_split = "test"
    evaluate.sub_geometry = True
    evaluate.save_mols = "false"
    evaluate.sampling_temperature = 1.0
    evaluate.motif_drop_eval = False
    evaluate.motif_drop_prob = 0.3
    evaluate.motif_drop_ratio_min = 0.2
    evaluate.motif_drop_ratio_max = 0.8

    config.seed = 42
    if device is None:
        config.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    else:
        config.device = torch.device(device)
    return config
