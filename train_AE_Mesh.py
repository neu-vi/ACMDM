import os
from os.path import join as pjoin
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from models.AE_Mesh import AE_models
from utils.datasets import AEMeshDataset
import time
from collections import OrderedDict, defaultdict
from utils.train_utils import update_lr_warm_up, def_value, save, print_current_loss
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
    motion_dir = pjoin(data_root, 'meshes')
    mean = np.load(pjoin(data_root, 'Mean_Mesh.npy')) # make sure this is computed
    std = np.load(pjoin(data_root, 'Std_Mesh.npy')) # make sure this is computed
    # mean = np.load(f'utils/mesh_mean_std/{args.dataset_name}/mesh_mean.npy')
    # std = np.load(f'utils/mesh_mean_std/{args.dataset_name}/mesh_std.npy')
    train_split_file = pjoin(data_root, 'train.txt')
    val_split_file = pjoin(data_root, 'val.txt')

    train_dataset = AEMeshDataset(mean, std, motion_dir, args.window_size, train_split_file)
    val_dataset = AEMeshDataset(mean, std, motion_dir, args.window_size, val_split_file)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, num_workers=args.num_workers,
                              shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, drop_last=True, num_workers=args.num_workers,
                            shuffle=True, pin_memory=True)
    #################################################################################
    #                                      Models                                   #
    #################################################################################
    model_dir = pjoin(args.checkpoints_dir, args.dataset_name, args.name, 'model')
    os.makedirs(model_dir, exist_ok=True)

    ae = AE_models[args.model]()  # here

    print(ae)
    pc_vae = sum(param.numel() for param in ae.parameters())
    print('Total parameters of all models: {}M'.format(pc_vae / 1000_000))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #################################################################################
    #                                    Training Loop                              #
    #################################################################################
    logger = SummaryWriter(model_dir)
    if args.recons_loss == 'l1':
        # use L1 loss is necessary for good recon for mesh vertices
        criterion = torch.nn.L1Loss(reduction='none')
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
        epoch, it = checkpoint['ep'] + 1, checkpoint['total_it']
        print("Load model epoch:%d iterations:%d" % (epoch, it))

    start_time = time.time()
    total_iters = args.epoch * len(train_loader)
    print(f'Total Epochs: {args.epoch}, Total Iters: {total_iters}')
    print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))

    current_lr = args.lr
    logs = defaultdict(def_value, OrderedDict())

    accum = 1
    while epoch < args.epoch:
        ae.train()
        for i, batch_data in enumerate(train_loader):
            it += 1
            if it < args.warm_up_iter and it % accum == 0:
                current_lr = update_lr_warm_up(it, args.warm_up_iter, optimizer, args.lr)
            motions = batch_data.detach().to(device).float()

            pred_motion = ae(motions)
            loss = criterion(pred_motion, motions).sum(dim=(1,2,3)).mean()
            loss = loss / accum
            loss_meaned = criterion(pred_motion, motions).mean()/ accum
            loss.backward()
            logs['loss'] += loss.item() * accum
            logs['loss_meaned'] += loss_meaned * accum
            logs['lr'] += (optimizer.param_groups[0]['lr'])
            torch.nn.utils.clip_grad_norm_(ae.parameters(), max_norm=1.0)

            if it % accum == 0:
                optimizer.step()
                if (it % accum) >= args.warm_up_iter:
                    scheduler.step()
                optimizer.zero_grad()

            if it % 100 == 0:
                save(pjoin(model_dir, 'latest.tar'), epoch, ae, optimizer, scheduler, it, 'ae')

            if it % args.log_every == 0:
                mean_loss = OrderedDict()
                for tag, value in logs.items():
                    logger.add_scalar('Train/%s' % tag, value / args.log_every, it)
                    mean_loss[tag] = value / args.log_every
                logs = defaultdict(def_value, OrderedDict())
                print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

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

        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='AE_Mesh')
    parser.add_argument('--model', type=str, default='AE_Model')
    parser.add_argument('--dataset_dir', type=str, default='./datasets')
    parser.add_argument('--dataset_name', type=str, default='t2m')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--window_size', type=int, default=1)

    parser.add_argument('--epoch', default=500, type=int)
    parser.add_argument('--warm_up_iter', default=0, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--milestones', default=[2500000], nargs="+", type=int)
    parser.add_argument('--lr_decay', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--recons_loss', type=str, default='l1')

    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument('--is_continue', action="store_true")
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')

    parser.add_argument('--log_every', default=100, type=int)

    arg = parser.parse_args()
    main(arg)