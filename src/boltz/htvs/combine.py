import copy

import torch
import torch.nn.functional as F
from torch import Tensor


def combine_feats(fp: dict[str, Tensor], fl: dict[str, Tensor]) -> dict[str, Tensor]:
    """Combine the protein and ligand feature dictionaries."""
    out = {}

    # Helper to pad spatial or sequence tensors
    # fp generally has shape (B, L_p, ...) and fl has (B, L_l, ...)
    # However, some are (B, K, L, 3) where K is number of frames.

    # Let's handle them specifically.

    # 1. 1D sequence tensors (B, L)
    seq_1d = [
        "token_index", "residue_index", "asym_id", "entity_id", "sym_id",
        "mol_type", "res_type", "disto_center", "token_pad_mask", "token_resolved_mask",
        "token_disto_mask", "method_feature", "modified", "cyclic_period", "affinity_token_mask"
    ]

    for key in seq_1d:
        if key in fp and key in fl:
            if key in ["token_index", "residue_index", "asym_id", "entity_id", "sym_id", "disto_center"]:
                # Offset the ligand indices
                max_val = fp[key].max() if fp[key].numel() > 0 else -1
                fl_offset = fl[key].clone()
                if key == "disto_center":
                    fl_offset = fl_offset + fp["atom_pad_mask"].shape[1] # num_atoms
                else:
                    # if they are zero-indexed, add max_val + 1
                    mask = fl[key] > 0 if key == "disto_center" else torch.ones_like(fl[key], dtype=torch.bool)
                    fl_offset = torch.where(mask, fl_offset + max_val + 1, fl_offset)
                out[key] = torch.cat([fp[key], fl_offset], dim=1)
            else:
                out[key] = torch.cat([fp[key], fl[key]], dim=1)

    # 2. Token bonds (B, L, L)
    for key in ("token_bonds", "type_bonds", "contact_conditioning", "contact_threshold"):
        if key in fp and key in fl:
            fp_tensor = fp[key]
            fl_tensor = fl[key]
            B, Lp, _ = fp_tensor.shape
            _, Ll, _ = fl_tensor.shape
            tb = torch.zeros((B, Lp + Ll, Lp + Ll), dtype=fp_tensor.dtype, device=fp_tensor.device)
            tb[:, :Lp, :Lp] = fp_tensor
            tb[:, Lp:, Lp:] = fl_tensor
            out[key] = tb

    # 3. Atom features (B, A)
    atom_1d = [
        "atom_index", "atom_element", "atom_charge", "atom_is_present",
        "atom_chirality", "atom_pad_mask", "atom_resolved_mask", "atom_bfactor"
    ]
    for key in atom_1d:
        if key in fp and key in fl:
            if key == "atom_index":
                max_val = fp[key].max() if fp[key].numel() > 0 else -1
                out[key] = torch.cat([fp[key], fl[key] + max_val + 1], dim=1)
            else:
                out[key] = torch.cat([fp[key], fl[key]], dim=1)

    if "atom_bonds" in fp and "atom_bonds" in fl:
        max_val_atom = fp["atom_index"].max() if ("atom_index" in fp and fp["atom_index"].numel() > 0) else -1
        out["atom_bonds"] = torch.cat([fp["atom_bonds"], fl["atom_bonds"] + max_val_atom + 1], dim=1)

    if "coords" in fp and "coords" in fl:
        out["coords"] = torch.cat([fp["coords"], fl["coords"]], dim=1)

    # 4. MSA features (B, N, L)
    msa_keys = [
        "msa", "msa_paired", "deletion_value", "has_deletion", "deletion_mean",
        "profile", "msa_mask"
    ]
    for key in msa_keys:
        if key in fp and key in fl:
            fp_tensor = fp[key]
            fl_tensor = fl[key]
            B, N_p, Lp = fp_tensor.shape[:3]
            _, N_l, Ll = fl_tensor.shape[:3]
            N_max = max(N_p, N_l)

            if N_p == N_max and N_l == N_max:
                pass
            else:
                if N_p < N_max:
                    pad_p = [0] * (2 * (fp_tensor.ndim - 1))
                    pad_idx = 2 * (fp_tensor.ndim - 2) + 1
                    pad_p[pad_idx] = N_max - N_p
                    fp_tensor = F.pad(fp_tensor, pad_p)
                if N_l < N_max:
                    pad_l = [0] * (2 * (fl_tensor.ndim - 1))
                    pad_idx = 2 * (fl_tensor.ndim - 2) + 1
                    pad_l[pad_idx] = N_max - N_l
                    fl_tensor = F.pad(fl_tensor, pad_l)
            out[key] = torch.cat([fp_tensor, fl_tensor], dim=2)

    # 5. Templates (B, N, L, ...)
    tmpl_keys = [
        "template_restype", "template_frame_rot", "template_frame_t",
        "template_cb", "template_ca", "template_mask_cb", "template_mask_frame",
        "template_mask", "query_to_template", "visibility_ids"
    ]
    for key in tmpl_keys:
        if key in fp and key in fl:
            fp_tensor = fp[key]
            fl_tensor = fl[key]
            B, N_p, Lp = fp_tensor.shape[:3]
            _, N_l, Ll = fl_tensor.shape[:3]
            N_max = max(N_p, N_l)

            if N_p == N_max and N_l == N_max:
                pass
            else:
                if N_p < N_max:
                    pad_p = [0] * (2 * (fp_tensor.ndim - 1))
                    pad_idx = 2 * (fp_tensor.ndim - 2) + 1
                    pad_p[pad_idx] = N_max - N_p
                    fp_tensor = F.pad(fp_tensor, pad_p)
                if N_l < N_max:
                    pad_l = [0] * (2 * (fl_tensor.ndim - 1))
                    pad_idx = 2 * (fl_tensor.ndim - 2) + 1
                    pad_l[pad_idx] = N_max - N_l
                    fl_tensor = F.pad(fl_tensor, pad_l)
            out[key] = torch.cat([fp_tensor, fl_tensor], dim=2)

    # Copy other keys from fp that don't need concatenation
    for key in fp:
        if key not in out:
            out[key] = fp[key]

    if "all_coords" in fp and "all_coords" in fl:
        out["all_coords"] = []
        for p_c, l_c in zip(fp["all_coords"], fl["all_coords"]):
            # p_c is a list/array of coords. Concatenate if possible
            if isinstance(p_c, list):
                out["all_coords"].append(p_c + l_c)
            elif isinstance(p_c, torch.Tensor):
                out["all_coords"].append(torch.cat([p_c, l_c], dim=0))
            else:
                out["all_coords"].append(p_c)

    if "crop_to_all_atom_map" in fp and "crop_to_all_atom_map" in fl:
        out["crop_to_all_atom_map"] = []
        for p_m, l_m in zip(fp["crop_to_all_atom_map"], fl["crop_to_all_atom_map"]):
            max_val = p_m.max() if (isinstance(p_m, torch.Tensor) and p_m.numel() > 0) else -1
            if isinstance(p_m, torch.Tensor):
                out["crop_to_all_atom_map"].append(torch.cat([p_m, l_m + max_val + 1], dim=0))
            else:
                out["crop_to_all_atom_map"].append(p_m)

    if "record" in fp and "record" in fl:
        new_records = []
        for p_rec, l_rec in zip(fp["record"], fl["record"]):
            c_rec = copy.deepcopy(p_rec)
            c_rec.id = p_rec.id + "_" + l_rec.id
            c_rec.chains.extend(l_rec.chains)
            c_rec.structure.num_chains = len(c_rec.chains)
            new_records.append(c_rec)
        out["record"] = new_records

    # Handle affinity specifically
    if "affinity_mw" in fl:
        out["affinity_mw"] = fl["affinity_mw"]
    if "affinity_token_mask" in fl and "affinity_token_mask" in fp:
        pass # Already handled in seq_1d

    # Fix symmetry and constraints!
    # symmetric_chain_index, connected_chain_index
    # We leave them as in fp, since ligands usually don't have constraints.

    return out
