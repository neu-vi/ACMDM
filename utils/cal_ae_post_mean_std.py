import numpy as np
import os
from os.path import join as pjoin
from tqdm import tqdm
import torch
import argparse
from models.AE_2D_Causal import AE_models

#################################################################################
#                           Calculate Post AE/VAE Mean Std                      #
#################################################################################

def mean_variance(data_dir, save_dir, ae, tp='AE'):
    file_list = os.listdir(data_dir)
    data_list = []
    mean = np.load(f'utils/22x3_mean_std/t2m/22x3_mean.npy')
    std = np.load(f'utils/22x3_mean_std/t2m/22x3_std.npy')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae = ae.to(device)

    for file in tqdm(file_list):
        data = np.load(pjoin(data_dir, file))
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=0)
        if np.isnan(data).any():
            print(file)
            continue
        data = data[:(data.shape[0]//4)*4, :, :]
        if data.shape[0] == 0:
            continue
        data = (data - mean) / std
        data = torch.from_numpy(data).to(device)
        with torch.no_grad():
            data_list.append(ae.encode(data.unsqueeze(0)).squeeze(0).cpu().numpy())

    data = np.concatenate(data_list, axis=1)
    data = data.reshape(4, -1)
    print('Data Range:')
    print(data.min(),data.max())
    Mean = data.mean(axis=1)
    Std = data.std(axis=1)
    print('Mean/Std:')
    print(Mean, Std)

    np.save(pjoin(save_dir, f'{tp}_2D_Causal_Post_Mean.npy'), Mean)
    np.save(pjoin(save_dir, f'{tp}_2D_Causal_Post_Std.npy'), Std)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    data_dir = 'datasets/HumanML3D/new_joints/'

    parser.add_argument('--is_ae', action="store_true")
    parser.add_argument('--ae_name', type=str, default="AE_2D_Causal")
    args = parser.parse_args()

    if args.is_ae:
        ae = AE_models["AE_Model"](input_width=3)
    else:
        ae = AE_models["VAE_Model"](input_width=3)
    ckpt = torch.load(f'checkpoints/t2m/{args.ae_name}/model/latest.tar', map_location='cpu')
    ae.load_state_dict(ckpt['ae'])
    ae = ae.eval()
    save_dir = f'checkpoints/t2m/{args.ae_name}'
    mean_variance(data_dir, save_dir, ae, tp='AE' if args.is_ae else 'VAE')