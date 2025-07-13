import os
from os.path import join as pjoin
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from models.AE_2D_Causal import AE_models
from models.ACMDM_Prefix_AR import ACMDM_models
from utils.datasets import Text2MotionDataset
import time
import copy
from collections import OrderedDict, defaultdict
from utils.train_utils import update_lr_warm_up, def_value, save, print_current_loss, update_ema
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

    train_dataset = Text2MotionDataset(mean, std, train_split_file, args.dataset_name, motion_dir, text_dir,
                                          args.unit_length, args.max_motion_length, 20, evaluation=False)
    val_dataset = Text2MotionDataset(mean, std, val_split_file, args.dataset_name, motion_dir, text_dir,
                                          args.unit_length, args.max_motion_length, 20, evaluation=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, num_workers=args.num_workers,
                              shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, drop_last=True, num_workers=args.num_workers,
                            shuffle=True)
    #################################################################################
    #                                      Models                                   #
    #################################################################################
    model_dir = pjoin(args.checkpoints_dir, args.dataset_name, args.name, 'model')
    os.makedirs(model_dir, exist_ok=True)

    ae = AE_models['AE_Model' if args.is_ae else 'VAE_Model'](input_width=dim_pose)
    ckpt = torch.load(pjoin(args.checkpoints_dir, args.dataset_name, args.ae_name, 'model', 'latest.tar'), map_location='cpu')
    model_key = 'ae'
    ae.load_state_dict(ckpt[model_key])

    acmdm = ACMDM_models[args.model](input_dim=ae.output_emb_width, cond_mode='text')
    ema_acmdm = copy.deepcopy(acmdm)
    ema_acmdm.eval()
    for param in ema_acmdm.parameters():
        param.requires_grad_(False)

    all_params = 0
    pc_transformer = sum(param.numel() for param in [p for name, p in acmdm.named_parameters() if not name.startswith('clip_model.')])
    all_params += pc_transformer
    print('Total parameters of all models: {:.2f}M'.format(all_params / 1000_000))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #################################################################################
    #                                    Training Loop                              #
    #################################################################################
    logger = SummaryWriter(model_dir)
    ae.eval()
    ae.to(device)
    acmdm.to(device)
    ema_acmdm.to(device)

    after_mean = torch.from_numpy(np.load(pjoin(args.checkpoints_dir, args.dataset_name, args.ae_name, args.after_mean))) # make sure this is either calculated or obtained from given checkpoint files
    after_std = torch.from_numpy(np.load(pjoin(args.checkpoints_dir, args.dataset_name, args.ae_name, args.after_std))) # make sure this is either calculated or obtained from given checkpoint files
    after_mean = after_mean.to(device)
    after_std = after_std.to(device)

    optimizer = optim.AdamW(acmdm.parameters(), betas=(0.9, 0.99), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.lr_decay)

    epoch = 0
    it = 0
    if args.is_continue:
        checkpoint = torch.load(pjoin(model_dir, 'latest.tar'), map_location=device)
        missing_keys, unexpected_keys = acmdm.load_state_dict(checkpoint['acmdm'], strict=False)
        missing_keys2, unexpected_keys2 = ema_acmdm.load_state_dict(checkpoint['ema_acmdm'], strict=False)
        assert len(unexpected_keys) == 0
        assert len(unexpected_keys2) == 0
        assert all([k.startswith('clip_model.') for k in missing_keys])
        assert all([k.startswith('clip_model.') for k in missing_keys2])
        optimizer.load_state_dict(checkpoint['opt_acmdm'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        epoch, it = checkpoint['ep']+1, checkpoint['total_it']
        print("Load model epoch:%d iterations:%d" % (epoch, it))

    start_time = time.time()
    total_iters = args.epoch * len(train_loader)
    print(f'Total Epochs: {args.epoch}, Total Iters: {total_iters}')
    print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))

    logs = defaultdict(def_value, OrderedDict())
    worst_loss = 100

    while epoch < args.epoch:
        ae.eval()
        acmdm.train()
        optimizer.zero_grad()

        for i, batch_data in enumerate(train_loader):
            it += 1
            if it < args.warm_up_iter:
                update_lr_warm_up(it, args.warm_up_iter, optimizer, args.lr)

            conds, motion, m_lens = batch_data
            motion = motion.detach().float().to(device)
            m_lens = m_lens.detach().long().to(device)

            with torch.no_grad():
                latent = ae.encode(motion).permute(0,2,3,1)
                latent = (latent - after_mean) / after_std
            m_lens = m_lens // 4

            real_in = []
            for j, ml in enumerate(m_lens):
                ll = max(ml - 15, 0)
                m_lens[j] = min(15, ml)
                idx = random.randint(0, ll)
                real_in.append(latent[j:j+1, idx:idx+15, :, :])
            latent = torch.cat(real_in, dim=0)

            conds = conds.to(device).float() if torch.is_tensor(conds) else conds

            # a naive way of writing gradient accumulation
            accum = args.accum
            accum_bs = args.batch_size // accum

            for kk in range(accum):
                loss = acmdm.forward_loss(latent.permute(0, 3, 1, 2)[kk * accum_bs:(kk + 1) * accum_bs],
                                          conds[kk * accum_bs:(kk + 1) * accum_bs],
                                          m_lens[kk * accum_bs:(kk + 1) * accum_bs])
                loss = loss / accum
                loss.backward()
                logs['loss'] += loss.item()
                logs['lr'] += (optimizer.param_groups[0]['lr'])/accum

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            update_ema(acmdm, ema_acmdm, 0.9999)

            if it % args.log_every == 0:
                mean_loss = OrderedDict()
                for tag, value in logs.items():
                    logger.add_scalar('Train/%s' % tag, value / args.log_every, it)
                    mean_loss[tag] = value / args.log_every
                logs = defaultdict(def_value, OrderedDict())
                print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

        save(pjoin(model_dir, 'latest.tar'), epoch, acmdm, optimizer, scheduler, it, 'acmdm', ema=ema_acmdm)
        #################################################################################
        #                                      Eval Loop                                #
        #################################################################################
        print('Validation time:')
        ae.eval()
        acmdm.eval()
        val_loss = []
        with torch.no_grad():
            for i, batch_data in enumerate(val_loader):
                conds, motion, m_lens = batch_data
                motion = motion.detach().float().to(device)
                m_lens = m_lens.detach().long().to(device)

                with torch.no_grad():
                    latent = ae.encode(motion).permute(0, 2, 3, 1)
                    latent = (latent - after_mean) / after_std
                m_lens = m_lens // 4

                real_in = []
                for j, ml in enumerate(m_lens):
                    ll = max(ml - 15, 0)
                    m_lens[j] = min(15, ml)
                    idx = random.randint(0, ll)
                    real_in.append(latent[j:j + 1, idx:idx + 15, :, :])
                latent = torch.cat(real_in, dim=0)

                conds = conds.to(device).float() if torch.is_tensor(conds) else conds

                loss = acmdm.forward_loss(latent.permute(0,3,1,2), conds, m_lens)
                val_loss.append(loss.item())

        print(f"Validation loss:{np.mean(val_loss):.3f}")
        logger.add_scalar('Val/loss', np.mean(val_loss), epoch)
        if np.mean(val_loss) < worst_loss:
            print(f"Improved loss from {worst_loss:.02f} to {np.mean(val_loss)}!!!")
            worst_loss = np.mean(val_loss)
        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='ACMDM_PrefixAR_Flow_S_PatchSize22')
    parser.add_argument('--ae_name', type=str, default="AE_2D_Causal")
    parser.add_argument('--is_ae', action="store_true")
    parser.add_argument('--after_mean', type=str, default='AE_2D_Causal_Post_Mean.npy')
    parser.add_argument('--after_std', type=str, default='AE_2D_Causal_Post_Std.npy')
    parser.add_argument('--model', type=str, default='ACMDM-PrefixAR-Flow-S-PatchSize22')
    parser.add_argument('--dataset_name', type=str, default='t2m')
    parser.add_argument('--dataset_dir', type=str, default='./datasets')
    parser.add_argument("--max_motion_length", type=int, default=196)
    parser.add_argument("--unit_length", type=int, default=4)
    parser.add_argument('--batch_size', default=64, type=int)

    parser.add_argument('--epoch', default=500, type=int)
    parser.add_argument('--warm_up_iter', default=2000, type=int)
    parser.add_argument('--accum', default=1, type=int)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--milestones', default=[50_000], nargs="+", type=int)
    parser.add_argument('--lr_decay', default=0.1, type=float)

    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument('--is_continue', action="store_true")
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')

    parser.add_argument('--log_every', default=50, type=int)

    arg = parser.parse_args()
    main(arg)