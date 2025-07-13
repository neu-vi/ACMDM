import os
from os.path import join as pjoin
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from models.AE_2D_Causal import AE_models
from utils.evaluators import Evaluators
from utils.datasets import Text2MotionDataset, collate_fn
from utils.eval_utils import evaluation_ae
import warnings
warnings.filterwarnings('ignore')
import argparse

def main(args):
    #################################################################################
    #                                      Seed                                     #
    #################################################################################
    torch.backends.cudnn.benchmark = False
    os.environ["OMP_NUM_THREADS"] = "1"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    #################################################################################
    #                                    Eval Data                                  #
    #################################################################################
    if args.dataset_name == "t2m":
        data_root = f'{args.dataset_dir}/HumanML3D/'
        dim_pose = 3
    else:
        raise NotImplementedError
    motion_dir = pjoin(data_root, 'new_joints')
    text_dir = pjoin(data_root, 'texts')
    max_motion_length = 196
    # mean = np.load(pjoin(data_root, 'Mean_22x3.npy'))
    # std = np.load(pjoin(data_root, 'Std_22x3.npy'))
    mean = np.load(f'utils/22x3_mean_std/{args.dataset_name}/22x3_mean.npy')
    std = np.load(f'utils/22x3_mean_std/{args.dataset_name}/22x3_std.npy')
    eval_mean = np.load(f'utils/eval_mean_std/{args.dataset_name}/eval_mean.npy')
    eval_std = np.load(f'utils/eval_mean_std/{args.dataset_name}/eval_std.npy')
    split_file = pjoin(data_root, 'test.txt')
    eval_dataset = Text2MotionDataset(mean, std, split_file, args.dataset_name, motion_dir, text_dir,
                                      4, max_motion_length, 20, evaluation=True)
    eval_loader = DataLoader(eval_dataset, batch_size=32, num_workers=args.num_workers, drop_last=True,
                            collate_fn=collate_fn, shuffle=True)
    #################################################################################
    #                                      Models                                   #
    #################################################################################
    model_dir = pjoin(args.checkpoints_dir, args.dataset_name, args.name, 'model')

    ae = AE_models[args.model](input_width=dim_pose)
    model_dir = os.path.join(model_dir, 'latest.tar')
    checkpoint = torch.load(model_dir, map_location='cpu')
    ae.load_state_dict(checkpoint['ae'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_wrapper = Evaluators(args.dataset_name, device=device)
    #################################################################################
    #                                  Evaluation Loop                              #
    #################################################################################
    out_dir = pjoin(args.checkpoints_dir, args.dataset_name, args.name, 'eval')
    os.makedirs(out_dir, exist_ok=True)
    f = open(pjoin(out_dir, 'eval.log'), 'w')

    ae.eval()
    ae.to(device)

    fid = []
    div = []
    top1 = []
    top2 = []
    top3 = []
    matching = []
    mae = []
    repeat_time = 20
    for i in range(repeat_time):
        best_fid, best_div, best_top1, best_top2, best_top3, best_matching, mpjpe = 1000, 0, 0, 0, 0, 100, 100
        best_fid, best_div, best_top1, best_top2, best_top3, best_matching, mpjpe, writer = evaluation_ae(
            model_dir, eval_loader, ae, None, i, device=device, best_fid=best_fid,
            best_div=best_div, best_top1=best_top1, best_top2=best_top2, best_top3=best_top3,
            eval_mean=eval_mean, eval_std=eval_std, best_matching=best_matching, eval_wrapper=eval_wrapper,
            save=False, draw=False)
        fid.append(best_fid)
        div.append(best_div)
        top1.append(best_top1)
        top2.append(best_top2)
        top3.append(best_top3)
        matching.append(best_matching)
        mae.append(mpjpe)

    fid = np.array(fid)
    div = np.array(div)
    top1 = np.array(top1)
    top2 = np.array(top2)
    top3 = np.array(top3)
    matching = np.array(matching)
    mae = np.array(mae)

    print(f'final result')
    print(f'final result', file=f, flush=True)

    msg_final = f"\tFID: {np.mean(fid):.3f}, conf. {np.std(fid) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                f"\tDiversity: {np.mean(div):.3f}, conf. {np.std(div) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                f"\tTOP1: {np.mean(top1):.3f}, conf. {np.std(top1) * 1.96 / np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2) * 1.96 / np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                f"\tMatching: {np.mean(matching):.3f}, conf. {np.std(matching) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                f"\tMAE:{np.mean(mae):.3f}, conf.{np.std(mae) * 1.96 / np.sqrt(repeat_time):.3f}\n\n"

    print(msg_final)
    print(msg_final, file=f, flush=True)
    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='AE_2D_Causal')
    parser.add_argument('--model', type=str, default='AE_Model')
    parser.add_argument('--dataset_dir', type=str, default='./datasets')
    parser.add_argument('--dataset_name', type=str, default='t2m')

    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')

    arg = parser.parse_args()
    main(arg)