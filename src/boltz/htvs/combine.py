import torch
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
    if "token_bonds" in fp and "token_bonds" in fl:
        B, Lp, _ = fp["token_bonds"].shape
        _, Ll, _ = fl["token_bonds"].shape
        tb = torch.zeros((B, Lp + Ll, Lp + Ll), dtype=fp["token_bonds"].dtype, device=fp["token_bonds"].device)
        tb[:, :Lp, :Lp] = fp["token_bonds"]
        tb[:, Lp:, Lp:] = fl["token_bonds"]
        out["token_bonds"] = tb

    if "type_bonds" in fp and "type_bonds" in fl:
        B, Lp, _ = fp["type_bonds"].shape
        _, Ll, _ = fl["type_bonds"].shape
        tb = torch.zeros((B, Lp + Ll, Lp + Ll), dtype=fp["type_bonds"].dtype, device=fp["type_bonds"].device)
        tb[:, :Lp, :Lp] = fp["type_bonds"]
        tb[:, Lp:, Lp:] = fl["type_bonds"]
        out["type_bonds"] = tb

    if "contact_conditioning" in fp and "contact_conditioning" in fl:
        B, Lp, _ = fp["contact_conditioning"].shape
        _, Ll, _ = fl["contact_conditioning"].shape
        tb = torch.zeros((B, Lp + Ll, Lp + Ll), dtype=fp["contact_conditioning"].dtype, device=fp["contact_conditioning"].device)
        tb[:, :Lp, :Lp] = fp["contact_conditioning"]
        tb[:, Lp:, Lp:] = fl["contact_conditioning"]
        out["contact_conditioning"] = tb

    if "contact_threshold" in fp and "contact_threshold" in fl:
        B, Lp, _ = fp["contact_threshold"].shape
        _, Ll, _ = fl["contact_threshold"].shape
        tb = torch.zeros((B, Lp + Ll, Lp + Ll), dtype=fp["contact_threshold"].dtype, device=fp["contact_threshold"].device)
        tb[:, :Lp, :Lp] = fp["contact_threshold"]
        tb[:, Lp:, Lp:] = fl["contact_threshold"]
        out["contact_threshold"] = tb

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
            # Fl MSA is usually empty (1, L_lig) and Fp is (N, L_prot). We must pad Fl.
            B, N_p, Lp = fp[key].shape[:3]
            _, N_l, Ll = fl[key].shape[:3]
            N_max = max(N_p, N_l)

            p_pad = torch.zeros((B, N_max, Lp) + fp[key].shape[3:], dtype=fp[key].dtype, device=fp[key].device)
            p_pad[:, :N_p, ...] = fp[key]

            l_pad = torch.zeros((B, N_max, Ll) + fl[key].shape[3:], dtype=fl[key].dtype, device=fl[key].device)
            l_pad[:, :N_l, ...] = fl[key]

            out[key] = torch.cat([p_pad, l_pad], dim=2)

    # 5. Templates (B, N, L, ...)
    tmpl_keys = [
        "template_restype", "template_frame_rot", "template_frame_t",
        "template_cb", "template_ca", "template_mask_cb", "template_mask_frame",
        "template_mask", "query_to_template", "visibility_ids"
    ]
    for key in tmpl_keys:
        if key in fp and key in fl:
            B, N_p, Lp = fp[key].shape[:3]
            _, N_l, Ll = fl[key].shape[:3]
            N_max = max(N_p, N_l)

            p_pad = torch.zeros((B, N_max, Lp) + fp[key].shape[3:], dtype=fp[key].dtype, device=fp[key].device)
            p_pad[:, :N_p, ...] = fp[key]

            l_pad = torch.zeros((B, N_max, Ll) + fl[key].shape[3:], dtype=fl[key].dtype, device=fl[key].device)
            l_pad[:, :N_l, ...] = fl[key]

            out[key] = torch.cat([p_pad, l_pad], dim=2)

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
        import copy
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
