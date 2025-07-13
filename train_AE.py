import os
from os.path import join as pjoin
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from models.AE_2D_Causal import AE_models
from utils.evaluators import Evaluators
from utils.datasets import AEDataset, Text2MotionDataset, collate_fn
import time
from collections import OrderedDict, defaultdict
from utils.train_utils import update_lr_warm_up, def_value, save, print_current_loss
from utils.eval_utils import evaluation_ae
import argparse

def main(args):
    #################################################################################
    #                                      Seed                                     #
    #################################################################################
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.autograd.set_detect_anomaly(True)
    # setting this to true significantly increase training and sampling speed
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    #################################################################################
    #                                    Train Data                                 #
    #################################################################################
    if args.dataset_name == "t2m":
        data_root = f'{args.dataset_dir}/HumanML3D/'
        joints_num = 22
        dim_pose = 3
    else:
        raise NotImplementedError
    motion_dir = pjoin(data_root, 'new_joints') # use new_joints for absolute coordinates
    text_dir = pjoin(data_root, 'texts')
    mean = np.load(pjoin(data_root, 'Mean_22x3.npy')) # make sure this is computed
    std = np.load(pjoin(data_root, 'Std_22x3.npy')) # make sure this is computed
    # mean = np.load(f'utils/22x3_mean_std/{args.dataset_name}/22x3_mean.npy')
    # std = np.load(f'utils/22x3_mean_std/{args.dataset_name}/22x3_std.npy')
    train_split_file = pjoin(data_root, 'train.txt')
    val_split_file = pjoin(data_root, 'val.txt')

    train_dataset = AEDataset(mean, std, motion_dir, args.window_size, train_split_file)
    val_dataset = AEDataset(mean, std, motion_dir, args.window_size, val_split_file)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, num_workers=args.num_workers,
                              shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, drop_last=True, num_workers=args.num_workers,
                            shuffle=True, pin_memory=True)
    #################################################################################
    #                                    Eval Data                                  #
    #################################################################################
    eval_mean = np.load(f'utils/eval_mean_std/{args.dataset_name}/eval_mean.npy')
    eval_std = np.load(f'utils/eval_mean_std/{args.dataset_name}/eval_std.npy')
    split_file = pjoin(data_root, 'val.txt')
    max_motion_length = 196
    eval_dataset = Text2MotionDataset(mean, std, split_file, args.dataset_name, pjoin(data_root, 'new_joints'), text_dir,
                                      4, max_motion_length, 20, evaluation=True)
    eval_loader = DataLoader(eval_dataset, batch_size=32, num_workers=args.num_workers, drop_last=True,
                            collate_fn=collate_fn, shuffle=True)
    #################################################################################
    #                                      Models                                   #
    #################################################################################
    model_dir = pjoin(args.checkpoints_dir, args.dataset_name, args.name, 'model')
    os.makedirs(model_dir, exist_ok=True)

    if args.is_ae:
        ae = AE_models['AE_Model'](input_width=dim_pose)
    else:
        ae = AE_models['VAE_Model'](input_width=dim_pose)

    print(ae)
    pc_vae = sum(param.numel() for param in ae.parameters())
    print('Total parameters of all models: {}M'.format(pc_vae / 1000_000))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_wrapper = Evaluators(args.dataset_name, device=device)
    #################################################################################
    #                                    Training Loop                              #
    #################################################################################
    logger = SummaryWriter(model_dir)
    if args.recons_loss == 'l1_smooth':
       criterion = torch.nn.SmoothL1Loss()
    else:
        criterion = torch.nn.MSELoss()

    ae.to(device)
    optimizer = optim.AdamW(ae.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.lr_decay)
    epoch = 0
    it = 0
    if args.is_continue:
        checkpoint = torch.load(pjoin(model_dir, 'latest.tar'), map_location=device)
        ae.load_state_dict(checkpoint['ae'])
        optimizer.load_state_dict(checkpoint[f'opt_ae'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        epoch, it = checkpoint['ep']+1, checkpoint['total_it']
        print("Load model epoch:%d iterations:%d" % (epoch, it))

    start_time = time.time()
    total_iters = args.epoch * len(train_loader)
    print(f'Total Epochs: {args.epoch}, Total Iters: {total_iters}')
    print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))

    current_lr = args.lr
    logs = defaultdict(def_value, OrderedDict())

    best_fid, best_div, best_top1, best_top2, best_top3, best_matching, mpjpe = 1000, 0, 0, 0, 0, 100, 100

    while epoch < args.epoch:
        ae.train()
        for i, batch_data in enumerate(train_loader):
            it += 1
            if it < args.warm_up_iter:
                current_lr = update_lr_warm_up(it, args.warm_up_iter, optimizer, args.lr)

            motions = batch_data.detach().to(device).float()

            # a naive way of writing gradient accumulation
            accum = args.accum
            accum_bs = args.batch_size // accum

            for kk in range(accum):
                if args.is_ae:
                    pred_motion = ae(motions[kk * accum_bs:(kk + 1) * accum_bs])
                    loss = criterion(pred_motion, motions[kk * accum_bs:(kk + 1) * accum_bs])
                    logs['loss'] += loss.item()
                else:
                    pred_motion, loss_dict = ae(motions[kk * accum_bs:(kk + 1) * accum_bs], need_loss=True)
                    loss_kl = loss_dict['kl']* args.kl_penality
                    loss_recon = loss_dict['rec']
                    logs['loss_recon'] += loss_recon.item() / accum
                    logs['loss_kl'] += loss_kl.item() / accum
                    loss = loss_kl + loss_recon
                logs['lr'] += (optimizer.param_groups[0]['lr']) / accum
                loss = loss / accum
                loss.backward()

            if not args.is_ae:
                # seems like vaes are more difficult to train
                torch.nn.utils.clip_grad_norm_(ae.parameters(), max_norm=1.0)

            optimizer.step()
            if it >= args.warm_up_iter:
                scheduler.step()
            optimizer.zero_grad()

            if it % args.log_every == 0:
                mean_loss = OrderedDict()
                for tag, value in logs.items():
                    logger.add_scalar('Train/%s' % tag, value / args.log_every, it)
                    mean_loss[tag] = value / args.log_every
                logs = defaultdict(def_value, OrderedDict())
                print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

        save(pjoin(model_dir, 'latest.tar'), epoch, ae, optimizer, scheduler, it, 'ae')
        #################################################################################
        #                                      Eval Loop                                #
        #################################################################################
        print('Validation time:')
        ae.eval()
        val_loss_rec = []
        val_loss = []
        with torch.no_grad():
            for i, batch_data in enumerate(val_loader):
                motions = batch_data.detach().to(device).float()
                pred_motion = ae(motions)

                loss_rec = criterion(pred_motion, motions)
                loss = loss_rec

                val_loss.append(loss.item())
                val_loss_rec.append(loss_rec.item())

        logger.add_scalar('Val/loss', sum(val_loss) / len(val_loss), epoch)
        logger.add_scalar('Val/loss_rec', sum(val_loss_rec) / len(val_loss_rec), epoch)
        print('Validation Loss: %.5f, Reconstruction: %.5f' %
              (sum(val_loss) / len(val_loss), sum(val_loss_rec) / len(val_loss)))

        best_fid, best_div, best_top1, best_top2, best_top3, best_matching, mpjpe, writer = evaluation_ae(
            model_dir, eval_loader, ae, logger, epoch, device=device, best_fid=best_fid,
            best_div=best_div, best_top1=best_top1, best_top2=best_top2, best_top3=best_top3,
            eval_mean=eval_mean, eval_std=eval_std, best_matching=best_matching, eval_wrapper=eval_wrapper)
        print(f'best fid {best_fid}')
        epoch += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='AE_2D_Causal')
    parser.add_argument('--is_ae', action="store_true")
    parser.add_argument('--dataset_dir', type=str, default='./datasets')
    parser.add_argument('--dataset_name', type=str, default='t2m')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--window_size', type=int, default=64)

    parser.add_argument('--epoch', default=50, type=int)
    parser.add_argument('--warm_up_iter', default=2000, type=int)
    parser.add_argument('--accum', default=2, type=int)
    parser.add_argument('--kl_penality', default=1e-2, type=float)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--milestones', default=[150000, 250000], nargs="+", type=int)
    parser.add_argument('--lr_decay', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--recons_loss', type=str, default='l1_smooth')

    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument('--is_continue', action="store_true")
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')

    parser.add_argument('--log_every', default=1000, type=int)

    arg = parser.parse_args()
    main(arg)