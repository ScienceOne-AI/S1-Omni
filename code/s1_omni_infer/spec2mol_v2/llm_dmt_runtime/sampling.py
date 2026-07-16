import numpy as np
import torch
from torch.nn import functional as F
import random
from tqdm import tqdm

from torch_sparse import sample
from .models.utils import sample_feature_noise, sample_combined_position_feature_noise, assert_mean_zero_with_mask, \
    sample_symmetric_edge_feature_noise, sample_gaussian_with_mask
from .utils import *

def _update_sampling_pbar(pbar, generated_count, total_count):
    target = min(generated_count, total_count)
    if target > pbar.n:
        pbar.update(target - pbar.n)


def _build_spectra_context(spectra_version, sampled_uv, sampled_ir, sampled_raman):
    if spectra_version == 'uv':
        return torch.stack(sampled_uv)
    if spectra_version == 'ir':
        return torch.stack(sampled_ir)
    if spectra_version == 'raman':
        return torch.stack(sampled_raman)
    if spectra_version == 'allspectra':
        return [torch.stack(sampled_uv), torch.stack(sampled_ir), torch.stack(sampled_raman)]
    raise ValueError(f'Invalid spectra_version: {spectra_version}')


def _stack_motif_onehot(sampled_motif_onehot, current_batch_size, model_name, source_desc):
    if len(sampled_motif_onehot) == current_batch_size:
        return torch.stack(sampled_motif_onehot, dim=0)
    if len(sampled_motif_onehot) == 0 and model_name != 'MAST_DMT':
        return None
    raise ValueError(
        f"{model_name} requires consistent motif_onehot for {source_desc}; "
        f"got {len(sampled_motif_onehot)} motif tensors for batch size {current_batch_size}."
    )


def _get_eval_motif_drop_config(config):
    enabled = bool(getattr(config.eval, 'motif_drop_eval', False))
    prob = float(getattr(config.eval, 'motif_drop_prob', 0.3))
    ratio_min = float(getattr(config.eval, 'motif_drop_ratio_min', 0.2))
    ratio_max = float(getattr(config.eval, 'motif_drop_ratio_max', 0.8))
    if prob < 0. or prob > 1.:
        raise ValueError(f'Invalid eval motif_drop_prob: {prob}')
    if ratio_min < 0. or ratio_max < ratio_min:
        raise ValueError(
            f'Invalid eval motif dropout ratio range: [{ratio_min}, {ratio_max}]'
        )
    return enabled, prob, ratio_min, ratio_max


def _build_eval_motif_drop_generator(config):
    seed = int(getattr(config, 'seed', 42)) + 1009
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _apply_eval_motif_onehot_dropout(motif_onehot, enabled, prob, ratio_min, ratio_max, generator):
    if motif_onehot is None or not enabled or prob <= 0.:
        return motif_onehot

    motif_onehot = motif_onehot.clone()
    for batch_idx in range(motif_onehot.size(0)):
        if torch.rand((), device=motif_onehot.device, generator=generator).item() >= prob:
            continue

        active_idx = torch.nonzero(motif_onehot[batch_idx] == 1, as_tuple=False).flatten()
        n_active = int(active_idx.numel())
        if n_active <= 1:
            continue

        drop_min = int(np.ceil(n_active * ratio_min))
        drop_max = min(int(np.floor(n_active * ratio_max)), n_active - 1)
        if drop_max < drop_min:
            continue

        drop_n = int(torch.randint(
            drop_min,
            drop_max + 1,
            (1,),
            device=motif_onehot.device,
            generator=generator,
        ).item())
        drop_idx = active_idx[
            torch.randperm(n_active, device=motif_onehot.device, generator=generator)[:drop_n]
        ]
        motif_onehot[batch_idx, drop_idx] = 0

    return motif_onehot


def _build_node_edge_masks(current_batch_size, max_n_nodes, n_nodes, device):
    node_mask = torch.zeros(current_batch_size, max_n_nodes)
    for i, num_nodes in enumerate(n_nodes):
        node_mask[i, 0:num_nodes] = 1
    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask
    edge_mask = edge_mask.view(current_batch_size * max_n_nodes * max_n_nodes, 1).to(device)
    node_mask = node_mask.unsqueeze(2).to(device)
    return node_mask, edge_mask


def mol_process(one_hot, x, formal_charges, n_nodes, edge_types=None):
    """Convert tensor to mols"""
    mol_list = []
    bs = one_hot.shape[0]
    for i in range(bs):
        atom_type = one_hot[i].argmax(1).cpu().detach()
        pos = x[i].cpu().detach()

        atom_type = atom_type[0:n_nodes[i]]
        pos = pos[0:n_nodes[i]]
        if edge_types is not None:
            edge_type = edge_types[i][:n_nodes[i], :n_nodes[i]].cpu().detach()
            if formal_charges.shape[-1] != 0:
                fc = formal_charges[i][:n_nodes[i], 0].long().cpu().detach()
            else:
                fc = formal_charges[i][:n_nodes[i]].cpu().detach()
            mol_list.append((pos, atom_type, edge_type, fc))
        else:
            mol_list.append((pos, atom_type))

    return mol_list


def mol_process_2D(one_hot, formal_charges, n_nodes, edge_types=None):
    """Convert tensor to mols, without 3D position."""
    mol_list = []
    bs = one_hot.shape[0]
    for i in range(bs):
        atom_type = one_hot[i].argmax(1).cpu().detach()
        atom_type = atom_type[0:n_nodes[i]]

        edge_type = edge_types[i][:n_nodes[i], :n_nodes[i]].cpu().detach()
        if formal_charges.shape[-1] != 0:
            fc = formal_charges[i][:n_nodes[i], 0].long().cpu().detach()
        else:
            fc = formal_charges[i][:n_nodes[i]].cpu().detach()
        mol_list.append((None, atom_type, edge_type, fc))

    return mol_list


def post_process(xh, atom_types, include_charge, node_mask, inverse_scaler,
                 edge_x=None, edge_mask=None, compress_edge=False):
    """Split the xh [bs, n_nodes, pos_dim+atom_types+fc_charge], unormalize data"""
    pos = xh[:, :, :3]
    if include_charge:
        h_int = xh[:, :, -1:]
        h_cat = xh[:, :, 3:-1]
    else:
        h_int = torch.zeros(0).to(xh.device)
        h_cat = xh[:, :, 3:]

    assert h_cat.shape[-1] == atom_types

    if edge_x is not None:
        pos, h_cat, h_int, h_edge = inverse_scaler(pos, h_cat, h_int, node_mask, edge_x, edge_mask)
    else:
        pos, h_cat, h_int = inverse_scaler(pos, h_cat, h_int, node_mask)
    h_cat = F.one_hot(torch.argmax(h_cat, dim=2), atom_types) * node_mask
    h_int = torch.round(h_int).long() * node_mask
    if edge_x is not None:
        if compress_edge:
            edge_exist = h_edge[:, :, :, 0]
            edge_exist[edge_exist < 0.5] = 0.
            edge_exist[edge_exist >= 0.5] = 1.0
            edge_type = h_edge[:, :, :, 1] * 3.
            edge_type[edge_type >= 2.5] = 3.
            edge_type[torch.bitwise_and(edge_type >= 1.5, edge_type < 2.5)] = 2.
            edge_type[torch.bitwise_and(edge_type >= 0.5, edge_type < 1.5)] = 1.
            edge_type[edge_type < 0.5] = 0.
            edge_type = edge_exist * edge_type
            if h_edge.size(-1) == 3:
                edge_aromatic = h_edge[:, :, :, 2]
                edge_aromatic[edge_aromatic < 0.5] = 0.
                edge_aromatic[edge_aromatic >= 0.5] = 1.
                edge_aromatic = edge_exist * edge_aromatic
                edge_type[torch.bitwise_and(edge_aromatic > 0., edge_type == 0.)] = 4.
            h_edge = edge_type
        else:
            # all 0 set non-exist, others set argmax
            h_edge_exist = torch.sum(h_edge > 0.5, dim=-1) != 0
            h_edge = torch.argmax(h_edge, dim=-1) + 1.0
            h_edge = h_edge_exist * h_edge

        return pos, h_cat, h_int, h_edge
    return pos, h_cat, h_int


def post_process_2D(xh, atom_types, include_charge, node_mask, inverse_scaler,
                    edge_x=None, edge_mask=None, compress_edge=False):
    """Split the xh [bs, n_nodes, pos_dim+atom_types+fc_charge], unormalize data"""
    if include_charge:
        h_int = xh[:, :, -1:]
        h_cat = xh[:, :, :-1]
    else:
        h_int = torch.zeros(0).to(xh.device)
        h_cat = xh[:, :, :]

    assert h_cat.shape[-1] == atom_types
    assert edge_x is not None

    _, h_cat, h_int, h_edge = inverse_scaler(None, h_cat, h_int, node_mask, edge_x, edge_mask)

    h_cat = F.one_hot(torch.argmax(h_cat, dim=2), atom_types) * node_mask
    h_int = torch.round(h_int).long() * node_mask

    if compress_edge:
        edge_exist = h_edge[:, :, :, 0]
        edge_exist[edge_exist < 0.5] = 0.
        edge_exist[edge_exist >= 0.5] = 1.0
        edge_type = h_edge[:, :, :, 1] * 3.
        edge_type[edge_type >= 2.5] = 3.
        edge_type[torch.bitwise_and(edge_type >= 1.5, edge_type < 2.5)] = 2.
        edge_type[torch.bitwise_and(edge_type >= 0.5, edge_type < 1.5)] = 1.
        edge_type[edge_type < 0.5] = 0.
        edge_type = edge_exist * edge_type
        if h_edge.size(-1) == 3:
            edge_aromatic = h_edge[:, :, :, 2]
            edge_aromatic[edge_aromatic < 0.5] = 0.
            edge_aromatic[edge_aromatic >= 0.5] = 1.
            edge_aromatic = edge_exist * edge_aromatic
            edge_type[torch.bitwise_and(edge_aromatic > 0., edge_type == 0.)] = 4.
        h_edge = edge_type
    else:
        # all 0 set non-exist, others set argmax
        h_edge_exist = torch.sum(h_edge > 0.5, dim=-1) != 0
        h_edge = torch.argmax(h_edge, dim=-1) + 1.0
        h_edge = h_edge_exist * h_edge

    return h_cat, h_int, h_edge


def expand_dims(v, dims):
    return v[(...,) + (None,) * (dims - 1)]


def get_sampling_fn(config, noise_scheduler, batch_size, n_samples, inverse_scaler, val_ds, eps=1e-3):
    device = config.device
    sampling_steps = config.sampling.steps
    atom_types = config.data.atom_types
    include_fc = config.model.include_fc_charge
    node_nf = atom_types + int(include_fc)
    pred_edge = config.pred_edge
    edge_nf = config.model.edge_ch
    compress_edge = config.data.compress_edge
    self_cond = config.model.self_cond
    only_2D = config.only_2D
    spectra_version = config.data.spectra_version
    model_name = config.model.name
    motif_drop_eval, motif_drop_prob, motif_drop_ratio_min, motif_drop_ratio_max = _get_eval_motif_drop_config(config)

    num_sampling_rounds = int(np.ceil(n_samples / batch_size))
    if config.sampling.method == 'ancestral':
        time_steps = torch.linspace(noise_scheduler.T, eps, sampling_steps, device=device)
        if only_2D:
            sampler = AncestralSampler_2D(noise_scheduler, time_steps, config.model.pred_data, self_cond)
        else:
            sampler = AncestralSampler(noise_scheduler, time_steps, config.model.pred_data, pred_edge, self_cond,
                                       get_self_cond_fn(config))
    else:
        raise ValueError('Invalid sampling method!')

    def sampling_fn(model):
        model.eval()
        processed_mols = []
        sampled_val_pos = []
        sampled_val_rdkit_mols = []
        with torch.no_grad():            
            # Modified sampling process, sample spectra from test set as condition, sample spectra first, and get corresponding real atom count
            num_mol_val = len(val_ds)
            permute_val_mol_id = torch.randperm(num_mol_val)

            with tqdm(total=n_samples, desc='Sampling (train-3D)', unit='mol') as pbar:
                for r in range(num_sampling_rounds):
                    sampled_val_mol_id= permute_val_mol_id[r * batch_size:(r + 1) * batch_size]
                    current_batch_size = len(sampled_val_mol_id)
                    sampled_val_mol = []
                    sampled_val_uv = []
                    sampled_val_ir = []
                    sampled_val_raman = []
                    sampled_motif_onehot = []
                    n_nodes = []
                    for i in sampled_val_mol_id:
                        mol = val_ds[i]
                        sampled_val_mol.append(mol)
                        if spectra_version=='uv':
                            sampled_val_uv.append(mol.uv)
                        if spectra_version=='ir':
                            sampled_val_ir.append(mol.ir)
                        if spectra_version=='raman':
                            sampled_val_raman.append(mol.raman)
                        if spectra_version=='allspectra':
                            sampled_val_uv.append(mol.uv)
                            sampled_val_ir.append(mol.ir)
                            sampled_val_raman.append(mol.raman)
                        sampled_val_pos.append(mol.pos)
                        sampled_val_rdkit_mols.append(mol.rdmol)
                        if hasattr(mol, 'motif_onehot'):
                            sampled_motif_onehot.append(mol.motif_onehot)
                        n_nodes.append(mol.num_atom.item())
                    context = _build_spectra_context(spectra_version, sampled_val_uv, sampled_val_ir, sampled_val_raman)
                    motif_onehot = _stack_motif_onehot(
                        sampled_motif_onehot, current_batch_size, model_name, 'training sampling'
                    )
                    # sampled_val_num_atoms = torch.stack(sampled_val_num_atoms)
                    
                    max_n_nodes = max(n_nodes)
                    

                    node_mask, edge_mask = _build_node_edge_masks(
                        current_batch_size, max_n_nodes, n_nodes, device
                    )

                    # sample initial noise
                    z = sample_combined_position_feature_noise(current_batch_size, max_n_nodes, node_nf, node_mask)
                    assert_mean_zero_with_mask(z[:, :, :3], node_mask)

                    # sample initial edge noise
                    if pred_edge:
                        edge_z = sample_symmetric_edge_feature_noise(current_batch_size, max_n_nodes, edge_nf, edge_mask)
                        # sampling procedure
                        x_node, x_edge = sampler.sampling(model, z, node_mask, edge_mask, edge_z, context, motif_onehot)
                        # postprocessing
                        pos, one_hot, fc, edge_types = post_process(x_node, atom_types, include_fc, node_mask,
                                                                    inverse_scaler, x_edge, edge_mask, compress_edge)
                    else:
                        # sampling procedure
                        x_node = sampler.sampling(model, z, node_mask, edge_mask)
                        # postprocessing: split features and discretize and checking, and inverse
                        pos, one_hot, fc = post_process(x_node, atom_types, include_fc, node_mask, inverse_scaler)

                    assert_mean_zero_with_mask(pos, node_mask)

                    # process tensors
                    if pred_edge:
                        processed_mols += mol_process(one_hot, pos, fc, n_nodes, edge_types)
                    else:
                        processed_mols += mol_process(one_hot, pos, fc, n_nodes)
                    _update_sampling_pbar(pbar, len(processed_mols), n_samples)

        # shuffle mols and pick n_samples
        # random.shuffle(processed_mols)
        # print('debug:', len(processed_mols), len(sampled_val_pos))
        # Save ground truth mol for coordinate error calculation and visualization
        return processed_mols[:n_samples], sampled_val_pos[:n_samples], sampled_val_rdkit_mols[:n_samples]

    def sampling_fn_2D(model):
        model.eval()
        processed_mols = []
        with torch.no_grad():
            num_mol_val = len(val_ds)
            permute_val_mol_id = torch.randperm(num_mol_val)

            with tqdm(total=n_samples, desc='Sampling (train-2D)', unit='mol') as pbar:
                for r in range(num_sampling_rounds):
                    sampled_val_mol_id= permute_val_mol_id[r * batch_size:(r + 1) * batch_size]
                    current_batch_size = len(sampled_val_mol_id)
                    sampled_val_mol = []
                    sampled_val_uv = []
                    sampled_val_ir = []
                    sampled_val_raman = []
                    sampled_motif_onehot = []
                    sampled_val_rdkit_mols = []
                    n_nodes = []
                    for i in sampled_val_mol_id:
                        mol = val_ds[i]
                        sampled_val_mol.append(mol)
                        if spectra_version=='uv':
                            sampled_val_uv.append(mol.uv)
                        if spectra_version=='ir':
                            sampled_val_ir.append(mol.ir)
                        if spectra_version=='raman':
                            sampled_val_raman.append(mol.raman)
                        if spectra_version=='allspectra':
                            sampled_val_uv.append(mol.uv)
                            sampled_val_ir.append(mol.ir)
                            sampled_val_raman.append(mol.raman)
                            sampled_val_rdkit_mols.append(mol.rdmol)
                        if hasattr(mol, 'motif_onehot'):
                            sampled_motif_onehot.append(mol.motif_onehot)
                        n_nodes.append(mol.num_atom.item())
                    context = _build_spectra_context(spectra_version, sampled_val_uv, sampled_val_ir, sampled_val_raman)
                    motif_onehot = _stack_motif_onehot(
                        sampled_motif_onehot, current_batch_size, model_name, 'training 2D sampling'
                    )
                    # sampled_val_num_atoms = torch.stack(sampled_val_num_atoms)
                    
                    max_n_nodes = max(n_nodes)

                    node_mask, edge_mask = _build_node_edge_masks(
                        current_batch_size, max_n_nodes, n_nodes, device
                    )

                    # sample initial noise
                    # z = sample_feature_noise(batch_size, max_n_nodes, node_nf, node_mask)
                    z = sample_gaussian_with_mask((current_batch_size, max_n_nodes, node_nf), device, node_mask)

                    # sample initial edge noise
                    edge_z = sample_symmetric_edge_feature_noise(current_batch_size, max_n_nodes, edge_nf, edge_mask)
                    # sampling procedure
                    x_node, x_edge = sampler.sampling(model, z, node_mask, edge_mask, edge_z, context, motif_onehot)
                    # postprocessing
                    one_hot, fc, edge_types = post_process_2D(x_node, atom_types, include_fc, node_mask,
                                                                inverse_scaler, x_edge, edge_mask, compress_edge)

                    # process tensors
                    processed_mols += mol_process_2D(one_hot, fc, n_nodes, edge_types)
                    _update_sampling_pbar(pbar, len(processed_mols), n_samples)

        # shuffle mols and pick n_samples
        random.shuffle(processed_mols)
        return processed_mols[:n_samples], sampled_val_rdkit_mols[:n_samples]

    if only_2D:
        return sampling_fn_2D
    else:
        return sampling_fn


def get_cond_sampling_eval_fn(config, noise_scheduler, batch_size, n_samples, inverse_scaler, test_ds, eps=1e-3):
    device = config.device
    sampling_steps = config.sampling.steps
    atom_types = config.data.atom_types
    include_fc = config.model.include_fc_charge
    node_nf = atom_types + int(include_fc)
    pred_edge = config.pred_edge
    edge_nf = config.model.edge_ch
    compress_edge = config.data.compress_edge
    self_cond = config.model.self_cond
    only_2D = config.only_2D
    spectra_version = config.data.spectra_version
    model_name = config.model.name
    motif_drop_eval, motif_drop_prob, motif_drop_ratio_min, motif_drop_ratio_max = _get_eval_motif_drop_config(config)

    num_sampling_rounds = int(np.ceil(n_samples / batch_size))
    if config.sampling.method == 'ancestral':
        time_steps = torch.linspace(noise_scheduler.T, eps, sampling_steps, device=device)
        if only_2D:
            sampler = AncestralSampler_2D(noise_scheduler, time_steps, config.model.pred_data, self_cond)
        else:
            sampler = AncestralSampler(noise_scheduler, time_steps, config.model.pred_data, pred_edge, 
                                        self_cond, get_self_cond_fn(config), 
                                        sampling_temperature=config.eval.sampling_temperature)
    else:
        raise ValueError('Invalid sampling method!')

    def sampling_fn(model):
        model.eval()
        processed_mols = []
        sampled_test_pos = []
        sampled_test_rdkit_mols = []
        motif_drop_generator = _build_eval_motif_drop_generator(config)

        with torch.no_grad():            
            num_mol_test = len(test_ds)
            # Fix random seed to ensure different models in each experiment are based on the same spectral sampling
            torch.manual_seed(42)
            permute_test_mol_id = torch.randperm(num_mol_test)

            with tqdm(total=n_samples, desc='Sampling (eval-3D)', unit='mol') as pbar:
                for r in range(num_sampling_rounds):
                    sampled_test_mol_id= permute_test_mol_id[r * batch_size:(r + 1) * batch_size]
                    current_batch_size = len(sampled_test_mol_id)
                    sampled_test_mol = []
                    sampled_val_uv = []
                    sampled_val_ir = []
                    sampled_val_raman = []
                    sampled_motif_onehot = []
                    n_nodes = []
                    for i in sampled_test_mol_id:
                        mol = test_ds[i]
                        sampled_test_mol.append(mol)
                        if spectra_version=='uv':
                            sampled_val_uv.append(mol.uv)
                        if spectra_version=='ir':
                            sampled_val_ir.append(mol.ir)
                        if spectra_version=='raman':
                            sampled_val_raman.append(mol.raman)
                        if spectra_version=='allspectra':
                            sampled_val_uv.append(mol.uv)
                            sampled_val_ir.append(mol.ir)
                            sampled_val_raman.append(mol.raman)
                        sampled_test_pos.append(mol.pos)
                        sampled_test_rdkit_mols.append(mol.rdmol)
                        if hasattr(mol, 'motif_onehot'):
                            sampled_motif_onehot.append(mol.motif_onehot)
                        n_nodes.append(mol.num_atom.item())
                    context = _build_spectra_context(spectra_version, sampled_val_uv, sampled_val_ir, sampled_val_raman)
                    motif_onehot = _stack_motif_onehot(
                        sampled_motif_onehot, current_batch_size, model_name, 'evaluation sampling'
                    )
                    motif_onehot = _apply_eval_motif_onehot_dropout(
                        motif_onehot,
                        motif_drop_eval,
                        motif_drop_prob,
                        motif_drop_ratio_min,
                        motif_drop_ratio_max,
                        motif_drop_generator,
                    )
                    
                    max_n_nodes = max(n_nodes)

                    node_mask, edge_mask = _build_node_edge_masks(
                        current_batch_size, max_n_nodes, n_nodes, device
                    )

                    # sample initial noise
                    z = sample_combined_position_feature_noise(current_batch_size, max_n_nodes, node_nf, node_mask)
                    assert_mean_zero_with_mask(z[:, :, :3], node_mask)

                    # sample initial edge noise
                    if pred_edge:
                        edge_z = sample_symmetric_edge_feature_noise(current_batch_size, max_n_nodes, edge_nf, edge_mask)
                        # sampling procedure
                        x_node, x_edge = sampler.sampling(model, z, node_mask, edge_mask, edge_z, context, motif_onehot)
                        # postprocessing
                        pos, one_hot, fc, edge_types = post_process(x_node, atom_types, include_fc, node_mask,
                                                                    inverse_scaler, x_edge, edge_mask, compress_edge)
                    else:
                        # sampling procedure
                        x_node = sampler.sampling(model, z, node_mask, edge_mask)
                        # postprocessing: split features and discretize and checking, and inverse
                        pos, one_hot, fc = post_process(x_node, atom_types, include_fc, node_mask, inverse_scaler)

                    assert_mean_zero_with_mask(pos, node_mask)

                    if pred_edge:
                        processed_mols += mol_process(one_hot, pos, fc, n_nodes, edge_types)
                    else:
                        processed_mols += mol_process(one_hot, pos, fc, n_nodes)
                    _update_sampling_pbar(pbar, len(processed_mols), n_samples)

        # Save ground-truth mol for coordinate error calculation and visualization
        return processed_mols[:n_samples], sampled_test_pos[:n_samples], sampled_test_rdkit_mols[:n_samples]

    def sampling_fn_2D(model):
        model.eval()
        processed_mols = []
        sampled_test_rdkit_mols = []
        motif_drop_generator = _build_eval_motif_drop_generator(config)

        with torch.no_grad():            
            num_mol_test = len(test_ds)
            permute_test_mol_id = torch.randperm(num_mol_test)

            with tqdm(total=n_samples, desc='Sampling (eval-2D)', unit='mol') as pbar:
                for r in range(num_sampling_rounds):
                    sampled_test_mol_id= permute_test_mol_id[r * batch_size:(r + 1) * batch_size]
                    current_batch_size = len(sampled_test_mol_id)
                    sampled_test_mol = []
                    sampled_val_uv = []
                    sampled_val_ir = []
                    sampled_val_raman = []
                    sampled_motif_onehot = []
                    n_nodes = []
                    for i in sampled_test_mol_id:
                        mol = test_ds[i]
                        sampled_test_mol.append(mol)
                        if spectra_version=='uv':
                            sampled_val_uv.append(mol.uv)
                        if spectra_version=='ir':
                            sampled_val_ir.append(mol.ir)
                        if spectra_version=='raman':
                            sampled_val_raman.append(mol.raman)
                        if spectra_version=='allspectra':
                            sampled_val_uv.append(mol.uv)
                            sampled_val_ir.append(mol.ir)
                            sampled_val_raman.append(mol.raman)
                        sampled_test_rdkit_mols.append(mol.rdmol)
                        if hasattr(mol, 'motif_onehot'):
                            sampled_motif_onehot.append(mol.motif_onehot)
                        n_nodes.append(mol.num_atom.item())
                    context = _build_spectra_context(spectra_version, sampled_val_uv, sampled_val_ir, sampled_val_raman)
                    motif_onehot = _stack_motif_onehot(
                        sampled_motif_onehot, current_batch_size, model_name, 'evaluation 2D sampling'
                    )
                    motif_onehot = _apply_eval_motif_onehot_dropout(
                        motif_onehot,
                        motif_drop_eval,
                        motif_drop_prob,
                        motif_drop_ratio_min,
                        motif_drop_ratio_max,
                        motif_drop_generator,
                    )
                    # sampled_test_num_atoms = torch.stack(sampled_test_num_atoms)
                    
                    max_n_nodes = max(n_nodes)

                    node_mask, edge_mask = _build_node_edge_masks(
                        current_batch_size, max_n_nodes, n_nodes, device
                    )

                    # sample initial noise
                    z = sample_gaussian_with_mask((current_batch_size, max_n_nodes, node_nf), device, node_mask)

                    # sample initial edge noise
                    edge_z = sample_symmetric_edge_feature_noise(current_batch_size, max_n_nodes, edge_nf, edge_mask)
                    # sampling procedure
                    x_node, x_edge = sampler.sampling(model, z, node_mask, edge_mask, edge_z, context, motif_onehot)
                    # postprocessing
                    one_hot, fc, edge_types = post_process_2D(x_node, atom_types, include_fc, node_mask,
                                                                inverse_scaler, x_edge, edge_mask, compress_edge)

                    processed_mols += mol_process_2D(one_hot, fc, n_nodes, edge_types)
                    _update_sampling_pbar(pbar, len(processed_mols), n_samples)

        # Save ground-truth mol for coordinate error calculation and visualization
        return processed_mols[:n_samples], sampled_test_rdkit_mols[:n_samples]

    if only_2D:
        return sampling_fn_2D
    else:
        return sampling_fn


class AncestralSampler:
    """Ancestral sampling for 2D & 3D joint generation."""
    def __init__(self, noise_scheduler, time_steps, model_pred_data, pred_edge=False, self_cond=False, cond_process_fn=None, sampling_temperature=1.0):
        self.noise_scheduler = noise_scheduler
        self.t_array = time_steps
        self.s_array = torch.cat([time_steps[1:], torch.zeros(1, device=time_steps.device)])
        self.model_pred_data = model_pred_data
        self.pred_edge = pred_edge
        self.self_cond = self_cond
        self.cond_process_fn = cond_process_fn
        self.sampling_temperature = sampling_temperature
        
    def sampling(self, model, z_T, node_mask, edge_mask, edge_z_T=None, context=None, motif_onehot=None):
        x = z_T
        edge_x = edge_z_T
        bs = z_T.shape[0]
        cond_x, cond_edge_x = None, None
        for i in range(len(self.t_array)):
            t = self.t_array[i]
            s = self.s_array[i]
            alpha_t, sigma_t = self.noise_scheduler.marginal_prob(t)
            alpha_s, sigma_s = self.noise_scheduler.marginal_prob(s)

            alpha_t_given_s = alpha_t / alpha_s
            # tmp = (1 - alpha_t_given_s**2) * c
            sigma2_t_given_s = sigma_t ** 2 - alpha_t_given_s ** 2 * sigma_s ** 2
            sigma_t_given_s = torch.sqrt(sigma2_t_given_s)
            sigma = sigma_t_given_s * sigma_s / sigma_t

            vec_t = torch.ones(bs, device=x.device) * t

            noise_level = torch.ones(bs, device=x.device) * torch.log(alpha_t ** 2 / sigma_t ** 2)
            if self.pred_edge:
                if self.self_cond:
                    assert self.model_pred_data
                    pred_t, edge_pred_t = model(vec_t, x, node_mask, edge_mask, edge_x=edge_x, noise_level=noise_level,
                                                cond_x=cond_x, cond_edge_x=cond_edge_x, context=context,
                                                motif_onehot=motif_onehot)
                    cond_x, cond_edge_x = self.cond_process_fn(pred_t, edge_pred_t)
                else:
                    pred_t, edge_pred_t = model(vec_t, x, node_mask, edge_mask, edge_x=edge_x, noise_level=noise_level,
                                                context=context, motif_onehot=motif_onehot)

            else:
                if self.self_cond:
                    assert self.model_pred_data
                    pred_t = model(vec_t, x, node_mask, edge_mask, noise_level=noise_level,
                                   cond_x=cond_x, context=context, motif_onehot=motif_onehot)
                else:
                    pred_t = model(vec_t, x, node_mask, edge_mask, noise_level=noise_level,
                                   context=context, motif_onehot=motif_onehot)

            # node update
            if self.model_pred_data:
                x_mean = expand_dims((alpha_t_given_s * sigma_s ** 2 / sigma_t ** 2).repeat(bs), x.dim()) * x \
                         + expand_dims((alpha_s * sigma2_t_given_s / sigma_t ** 2).repeat(bs), pred_t.dim()) * pred_t
            else:
                x_mean = x / expand_dims(alpha_t_given_s.repeat(bs), x.dim()) \
                         - expand_dims((sigma2_t_given_s / alpha_t_given_s / sigma_t).repeat(bs), pred_t.dim()) * pred_t

            x = x_mean + expand_dims(sigma.repeat(bs), x_mean.dim()) * \
                sample_combined_position_feature_noise(bs, x_mean.shape[1], x_mean.shape[2] - 3, node_mask) * self.sampling_temperature

            # edge update
            if self.pred_edge:
                if self.model_pred_data:
                    edge_x_mean = expand_dims((alpha_t_given_s * sigma_s**2 / sigma_t ** 2).repeat(bs), edge_x.dim()) \
                                  * edge_x + expand_dims((alpha_s * sigma2_t_given_s / sigma_t ** 2).repeat(bs),
                                                         edge_pred_t.dim()) * edge_pred_t
                else:
                    edge_x_mean = edge_x / expand_dims(alpha_t_given_s.repeat(bs), edge_x.dim()) - expand_dims(
                        (sigma2_t_given_s / alpha_t_given_s / sigma_t).repeat(bs), edge_pred_t.dim()) * edge_pred_t
                edge_x = edge_x_mean + expand_dims(sigma.repeat(bs), edge_x_mean.dim()) * \
                         sample_symmetric_edge_feature_noise(bs, edge_x_mean.shape[1], edge_x_mean.shape[-1], edge_mask) * self.sampling_temperature

        assert_mean_zero_with_mask(x_mean[:, :, :3], node_mask)

        if self.pred_edge:
            return x_mean, edge_x_mean
        else:
            return x_mean


class AncestralSampler_2D:
    """Ancestral Sampler without 3D positions."""
    def __init__(self, noise_scheduler, time_steps, model_pred_data, self_cond=False):
        self.noise_scheduler = noise_scheduler
        self.t_array = time_steps
        self.s_array = torch.cat([time_steps[1:], torch.zeros(1, device=time_steps.device)])
        self.model_pred_data = model_pred_data
        self.self_cond = self_cond

    def sampling(self, model, z_T, node_mask, edge_mask, edge_z_T=None, context=None, motif_onehot=None):
        x = z_T
        edge_x = edge_z_T
        bs = z_T.shape[0]
        cond_x, cond_edge_x = None, None
        for i in range(len(self.t_array)):
            t = self.t_array[i]
            s = self.s_array[i]
            alpha_t, sigma_t = self.noise_scheduler.marginal_prob(t)
            alpha_s, sigma_s = self.noise_scheduler.marginal_prob(s)

            alpha_t_given_s = alpha_t / alpha_s
            # tmp = (1 - alpha_t_given_s**2) * c
            sigma2_t_given_s = sigma_t ** 2 - alpha_t_given_s ** 2 * sigma_s ** 2
            sigma_t_given_s = torch.sqrt(sigma2_t_given_s)
            sigma = sigma_t_given_s * sigma_s / sigma_t

            vec_t = torch.ones(bs, device=x.device) * t

            noise_level = torch.ones(bs, device=x.device) * torch.log(alpha_t ** 2 / sigma_t ** 2)
            if self.self_cond:
                assert self.model_pred_data
                pred_t, edge_pred_t = model(vec_t, x, node_mask, edge_mask, edge_x=edge_x, noise_level=noise_level,
                                            cond_x=cond_x, cond_edge_x=cond_edge_x, context=context,
                                            motif_onehot=motif_onehot)
                cond_x, cond_edge_x = pred_t, edge_pred_t
            else:
                pred_t, edge_pred_t = model(vec_t, x, node_mask, edge_mask, edge_x=edge_x, noise_level=noise_level,
                                            context=context, motif_onehot=motif_onehot)

            # node update
            if self.model_pred_data:
                x_mean = expand_dims((alpha_t_given_s * sigma_s ** 2 / sigma_t ** 2).repeat(bs), x.dim()) * x \
                         + expand_dims((alpha_s * sigma2_t_given_s / sigma_t ** 2).repeat(bs), pred_t.dim()) * pred_t
            else:
                x_mean = x / expand_dims(alpha_t_given_s.repeat(bs), x.dim()) \
                         - expand_dims((sigma2_t_given_s / alpha_t_given_s / sigma_t).repeat(bs), pred_t.dim()) * pred_t

            x = x_mean + expand_dims(sigma.repeat(bs), x_mean.dim()) * \
                sample_gaussian_with_mask(x.size(), x.device, node_mask)

            # edge update
            if self.model_pred_data:
                edge_x_mean = expand_dims((alpha_t_given_s * sigma_s**2 / sigma_t ** 2).repeat(bs), edge_x.dim()) \
                              * edge_x + expand_dims((alpha_s * sigma2_t_given_s / sigma_t ** 2).repeat(bs),
                                                     edge_pred_t.dim()) * edge_pred_t
            else:
                edge_x_mean = edge_x / expand_dims(alpha_t_given_s.repeat(bs), edge_x.dim()) - expand_dims(
                        (sigma2_t_given_s / alpha_t_given_s / sigma_t).repeat(bs), edge_pred_t.dim()) * edge_pred_t

            edge_x = edge_x_mean + expand_dims(sigma.repeat(bs), edge_x_mean.dim()) * \
                     sample_symmetric_edge_feature_noise(bs, edge_x_mean.shape[1], edge_x_mean.shape[-1], edge_mask)

        return x_mean, edge_x_mean
