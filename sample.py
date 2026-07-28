# python sample.py --name ACMDM_Flow_S_PatchSize22 --text_prompt "A person is running on a treadmill." --motion_length 196 --cfg 3.0 --save_mp4

import os
from os.path import join as pjoin
from pathlib import Path
import argparse
import re
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from models.AE_2D_Causal import AE_models
from models.ACMDM import ACMDM_models
from models.LengthEstimator import LengthEstimator

# Reuse the exact plotting util from MARDM
# Assumes your repo layout matches ../MARDM/utils/motion_process.py
from utils.motion_process import plot_3d_motion, kit_kinematic_chain, t2m_kinematic_chain


def indices_from_batch(shard_index: int, n: int, shard_size: int):
    if shard_index is None or shard_index < 0:
        return set()
    lo = shard_index * shard_size
    hi = min((shard_index + 1) * shard_size - 1, n - 1)
    if lo <= hi:
        return set(range(lo, hi + 1))
    return set()


def apply_index_filter(pending, n_total: int, shard_size: int, shard_index):
    if shard_index is None or shard_size is None:
        return pending
    inc_set = indices_from_batch(shard_index, n_total, shard_size)
    return [i for i in pending if i in inc_set]


def safe_stem(s: str, max_len: int = 120) -> str:
    s = s.replace("/", "_").replace("\\", "_")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("._-")
    if len(s) > max_len:
        s = s[:max_len].rstrip("._-")
    return s or "caption"


def chunked(xs, n: int):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def load_descriptions_dir(desc_dir: Path):
    """
    Reads <desc_dir>/*.txt.
    Returns:
      prompt_list: list[str] of all non-empty lines across files (file order, then line order)
      group_of_prompt: list[str] same length; group = file stem
    """
    prompt_list = []
    group_of_prompt = []

    for fpath in sorted(desc_dir.glob("*.txt")):
        group = fpath.stem
        for line in fpath.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            prompt_list.append(line)
            group_of_prompt.append(group)

    return prompt_list, group_of_prompt


def _pick_length_estimator_ckpt(model_dir: str) -> str:
    """
    Be robust to whatever the zip contains.
    Try common filenames used in similar repos.
    """
    candidates = [
        "finest.tar",
        "latest.tar",
        "net_best_fid.tar",
        "best.tar",
        "model.tar",
    ]
    for c in candidates:
        p = pjoin(model_dir, c)
        if os.path.exists(p):
            return p

    # last resort: any .tar in the directory
    if os.path.isdir(model_dir):
        tars = sorted([pjoin(model_dir, f) for f in os.listdir(model_dir) if f.endswith(".tar")])
        if tars:
            return tars[-1]

    raise FileNotFoundError(
        f"Could not find a length estimator checkpoint under {model_dir}. "
        f"Tried {candidates} and then any *.tar."
    )


def _load_length_estimator(args, device):
    """
    Mirrors MARDM: LengthEstimator(512, 50) and checkpoint key often 'estimator'.
    """
    length_estimator = LengthEstimator(512, 50)

    le_dir = pjoin(args.checkpoints_dir, args.dataset_name, "length_estimator", "model")
    ckpt_path = _pick_length_estimator_ckpt(le_dir)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # MARDM uses ckpt['estimator'].
    # Add fallbacks in case the zip stores differently.
    if isinstance(ckpt, dict):
        if "estimator" in ckpt:
            state = ckpt["estimator"]
        elif "model" in ckpt:
            state = ckpt["model"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            # sometimes the dict itself is a state_dict-like mapping
            # (heuristic: contains some tensor values)
            tensorish = any(hasattr(v, "shape") for v in ckpt.values())
            if tensorish:
                state = ckpt
            else:
                raise KeyError(
                    f"Length estimator checkpoint {ckpt_path} has keys {list(ckpt.keys())}, "
                    "but none of ['estimator','model','state_dict'] matched."
                )
    else:
        raise TypeError(f"Unexpected checkpoint type for length estimator: {type(ckpt)}")

    length_estimator.load_state_dict(state, strict=True)
    length_estimator.eval().to(device)
    print(f"Loaded LengthEstimator from: {ckpt_path}")
    return length_estimator


def main(args):
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    result_dir = pjoin("./generation", args.name)
    os.makedirs(result_dir, exist_ok = True)

    # --- Load 22x3 mean/std used to normalize joint coords during training
    mean_22x3 = np.load(f"utils/22x3_mean_std/{args.dataset_name}/22x3_mean.npy")
    std_22x3  = np.load(f"utils/22x3_mean_std/{args.dataset_name}/22x3_std.npy")

    # --- Load AE
    ae = AE_models[args.ae_model](input_width=3)  # HumanML3D uses xyz per joint
    ae_ckpt = torch.load(
        pjoin(args.checkpoints_dir, args.dataset_name, args.ae_name, "model", "latest.tar"),
        map_location="cpu",
    )
    ae.load_state_dict(ae_ckpt["ae"])
    ae.eval().to(device)

    # --- Load ACMDM (EMA weights)
    acmdm = ACMDM_models[args.model](input_dim=ae.output_emb_width, cond_mode="text")
    acmdm_ckpt_path = pjoin(args.checkpoints_dir, args.dataset_name, args.name, "model", "latest.tar")
    acmdm_ckpt = torch.load(acmdm_ckpt_path, map_location="cpu")
    missing, unexpected = acmdm.load_state_dict(acmdm_ckpt["ema_acmdm"], strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys in ACMDM checkpoint: {unexpected}")
    if any((not k.startswith("clip_model.")) for k in missing):
        raise RuntimeError(f"Missing non-CLIP keys in ACMDM checkpoint: {missing}")
    acmdm.eval().to(device)

    # --- AE post mean/std for latent unnormalization
    after_mean = np.load(pjoin(args.checkpoints_dir, args.dataset_name, args.ae_name, "AE_2D_Causal_Post_Mean.npy"))
    after_std  = np.load(pjoin(args.checkpoints_dir, args.dataset_name, args.ae_name, "AE_2D_Causal_Post_Std.npy"))

    prompt_list = []
    group_of_prompt = []
    length_list = []
    est_length = False

    if args.descriptions:
        desc_dir = Path(args.descriptions_dir)
        prompt_list, group_of_prompt = load_descriptions_dir(desc_dir)
        if len(prompt_list) == 0:
            raise RuntimeError(f"No descriptions found under {desc_dir}/*.txt")
        if args.motion_length <= 0:
            est_length = True
        else:
            length_list = [int(args.motion_length)] * len(prompt_list)
    elif args.text_prompt != "":
        prompt_list = [args.text_prompt]
        group_of_prompt = ["single"]
        if args.motion_length <= 0:
            est_length = True
        else:
            length_list = [int(args.motion_length)]
    elif args.text_path != "":
        with open(args.text_path, 'r', encoding = "utf-8", errors = "replace") as file:
            lines = [ln.strip() for ln in file.readlines()]
        for line in lines:
            if not line:
                continue
            infos = line.split('#')
            prompt_list.append(infos[0].strip())
            group_of_prompt.append("text_path")
            if len(infos) == 1:
                est_length = True
                length_list = []
            else:
                tail = infos[-1].strip()
                if tail.isdigit():
                    if not est_length:
                        length_list.append(int(tail))
                else:
                    est_length = True
                    length_list = []
        if len(prompt_list) == 0:
            raise RuntimeError(f"--text_path {args.text_path} had no prompts.")
    else:
        raise RuntimeError("A text prompt, a file of text prompts, or --descriptions is required.")

    def prompt_done(group: str, idx: int) -> bool:
        s_path = Path(result_dir) / group / str(idx)
        if not s_path.exists():
            return False
        return (len(list(s_path.glob("*.mp4"))) > 0) or (len(list(s_path.glob("*.npy"))) > 0)

    pending_indices = [i for i in range(len(prompt_list)) if not prompt_done(group_of_prompt[i], i)]
    if not pending_indices:
        print(f"All {len(prompt_list)} prompts already have outputs under {result_dir}. Nothing to do.")
        return
    
    completed_count = len(prompt_list) - len(pending_indices)

    pending_indices = apply_index_filter(
        pending=pending_indices,
        n_total=len(prompt_list),
        shard_size=args.shard_size,
        shard_index=args.shard_index,
    )

    if args.shard_index is not None:
        print(
            f"Index filter applied (shard_index={args.shard_index}, shard_size={args.shard_size}). "
            f"Now considering {len(pending_indices)} pending prompts out of {len(prompt_list)} total."
        )

    if not pending_indices:
        if args.shard_index is not None and args.shard_size is not None:
            lo = args.shard_index * args.shard_size
            hi = min((args.shard_index + 1) * args.shard_size - 1, len(prompt_list) - 1)
            print(f"No pending prompts in selected shard {args.shard_index} (global indices {lo}-{hi}). Nothing to do.")
        else:
            print("No pending prompts. Nothing to do.")
        return

    print(f"Found {completed_count} completed; generating {len(pending_indices)} prompts:")
    print(f"First few pending indices: {pending_indices[:10]}")

    kinematic_chain = kit_kinematic_chain if args.dataset_name == "kit" else t2m_kinematic_chain

    length_estimator = None
    if est_length:
        print("No motion length specified for at least one prompt -> using LengthEstimator.")
        length_estimator = _load_length_estimator(args, device = device)

    batch_id_global = 0
    for r in range(args.repeat_times):
        print(f"--> Repeat {r}")

        for batch_indices in chunked(pending_indices, args.batch_size):
            batch_id_global += 1
            print(f"  -> Batch {batch_id_global} (size={len(batch_indices)}): {batch_indices[:10]}")

            batch_prompts = [prompt_list[i] for i in batch_indices]

            if est_length:
                with torch.no_grad():
                    text_emb = acmdm.encode_text(batch_prompts)
                    pred_dis = length_estimator(text_emb)
                    probs = F.softmax(pred_dis, dim = -1)
                    token_lens = Categorical(probs).sample().long()
                    token_lens = torch.clamp(token_lens, min = 1, max = 49)
                batch_latent_lens = token_lens.detach().cpu().tolist()
                batch_lengths = [(int(tl) * 4) for tl in batch_latent_lens]
            else:
                batch_lengths = [int(length_list[i]) for i in batch_indices]
                batch_latent_lens = [max(1, ml // 4) for ml in batch_lengths]

            with torch.no_grad():
                m_lens = torch.tensor(batch_latent_lens, device=device, dtype=torch.long)

                # ACMDM generates latents: (B, C, L, J)
                latents = acmdm.generate(
                    conds=batch_prompts,
                    m_lens=m_lens,
                    cond_scale=float(args.cfg),
                    j=22,
                )

                # Unnormalize latents with after_mean/std: (B,C,L,J) -> (B,L,J,C) -> apply -> back
                lat_np = latents.permute(0, 2, 3, 1).detach().cpu().numpy()  # (B,L,J,C)
                lat_np = lat_np * after_std + after_mean
                lat_t = torch.from_numpy(lat_np).to(device).permute(0, 3, 1, 2).contiguous()  # (B,C,L,J)

                # Decode to normalized xyz (per your AE): expected (B,T,J,3)
                motion_norm = ae.decode(lat_t).detach().cpu().numpy()

                # Unnormalize xyz using 22x3 mean/std from training
                mean = mean_22x3.reshape(1, 1, 1, 3)
                std = std_22x3.reshape(1, 1, 1, 3)
                motion_xyz = motion_norm * std + mean  # (B,T,J,3)

            # ----- Save per sample in MARDM folder structure
            for local_i, (orig_idx, caption, ml) in enumerate(zip(batch_indices, batch_prompts, batch_lengths)):
                group = group_of_prompt[orig_idx]
                print(f"    ----> {group}/{orig_idx}: {caption}  len={ml}")

                s_path = pjoin(result_dir, group, str(orig_idx))
                os.makedirs(s_path, exist_ok=True)

                joints_tj3 = motion_xyz[local_i]  # (T,J,3)
                joints_tj3 = joints_tj3[:ml]
                joints_tj3 = joints_tj3.astype(np.float32, copy=False)

                # basic stats like MARDM prints
                print(
                    "    joint stats:",
                    "min", np.nanmin(joints_tj3),
                    "max", np.nanmax(joints_tj3),
                    "nan?", np.isnan(joints_tj3).any(),
                    "inf?", np.isinf(joints_tj3).any(),
                )

                cap_stem = safe_stem(caption)
                mp4_name = f"{cap_stem}_sample{orig_idx}_repeat{r}_len{ml}.mp4"
                npy_name = f"{cap_stem}_sample{orig_idx}_repeat{r}_len{ml}.npy"
                save_mp4 = pjoin(s_path, mp4_name)
                save_npy = pjoin(s_path, npy_name)

                np.save(save_npy, joints_tj3)

                if not args.no_mp4:
                    plot_3d_motion(
                        save_mp4,
                        kinematic_chain,
                        joints_tj3,
                        title=caption,
                        fps=int(args.fps),
                        radius=float(args.radius),
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Batch/sharding args (match MARDM CLI)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--repeat_times", type=int, default=1)

    parser.add_argument(
        "--shard_index",
        type=int,
        default=None,
        help="Slurm array task id. Owns indices [k*shard_size .. (k+1)*shard_size-1].",
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=None,
        help="How many prompt indices each Slurm array task owns (e.g., 437). If None, no sharding is applied.",
    )

    # Model args
    parser.add_argument("--name", required=True, help="Run folder name under checkpoints/<dataset>/<name>/model/latest.tar")
    parser.add_argument("--model", default="ACMDM-Flow-S-PatchSize22")
    parser.add_argument("--ae_name", default="AE_2D_Causal")
    parser.add_argument("--ae_model", default="AE_Model")
    parser.add_argument("--dataset_name", default="t2m")
    parser.add_argument("--checkpoints_dir", default="./checkpoints")

    # Prompt sources (match MARDM style)
    parser.add_argument("--text_prompt", default="", type=str)
    parser.add_argument("--text_path", default="", type=str)
    parser.add_argument("--descriptions", action="store_true", help="If set, read prompts from descriptions_dir/*.txt; one prompt per line.")
    parser.add_argument("--descriptions_dir", type=str, default="descriptions", help="Directory containing *.txt files of descriptions (one per line).")

    # Length + CFG
    parser.add_argument("--motion_length", type=int, default=0, help="Required for ACMDM unless each line in --text_path provides #<len>.")
    parser.add_argument("--cfg", type=float, default=3.0)

    # Repro / plotting knobs
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--radius", type=float, default=4.0)
    parser.add_argument("--no_mp4", action="store_true", help="If set, do not render MP4s (still saves .npy).")

    args = parser.parse_args()
    main(args)