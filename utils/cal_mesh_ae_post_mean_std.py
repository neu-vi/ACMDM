# very hard coded will clean and refine later
import numpy as np
import os
from os.path import join as pjoin
from tqdm import tqdm
import torch
from models.AE_Mesh import AE_models
from human_body_prior.body_model.body_model import BodyModel

def downsample(data_dir, save_dir):
    ae = AE_models["AE_Model"](test_mode=True)
    ckpt = torch.load('checkpoints/t2m/AE_Mesh/model/latest.tar', map_location='cpu')
    model_key = 'ae'
    ae.load_state_dict(ckpt[model_key])
    ae = ae.eval()
    ae = ae.cuda()

    mean = np.load(f'utils/mesh_mean_std/t2m/mesh_mean.npy')  # or yours
    std = np.load(f'utils/mesh_mean_std/t2m/mesh_std.npy')  # or yours

    bm_path = './body_models/smplh/neutral/model.npz'
    dmpl_path = './body_models/dmpls/neutral/model.npz'
    num_betas = 10
    num_dmpls = 8
    bm = BodyModel(bm_fname=bm_path, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_path).cuda()
    comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    file_list = os.listdir(data_dir)
    file_list.sort()
    for file in tqdm(file_list):
        data = np.load(pjoin(data_dir, file))

        if np.isnan(data).any():
            print(f"Skipping {file} due to NaN")
            continue

        body_parms = {
            'root_orient': torch.from_numpy(data[:, :3]).to(comp_device).float(),
            'pose_body': torch.from_numpy(data[:, 3:66]).to(comp_device).float(),
            'pose_hand': torch.from_numpy(data[:, 66:52*3]).to(comp_device).float(),
            'trans': torch.from_numpy(data[:, 52*3:53*3]).to(comp_device).float(),
            'betas': torch.from_numpy(data[:, 53*3:53*3+10]).to(comp_device).float(),
            'dmpls': torch.from_numpy(data[:, 53*3+10:]).to(comp_device).float()
        }

        with torch.no_grad():
            verts = bm(**body_parms).v
        verts[:, :, 1] -= verts[:, :, 1].min()
        verts = verts.detach().cpu().numpy()
        "Z Normalization"
        vertss = (verts - mean) / std
        T = vertss.shape[0]
        data = torch.from_numpy(vertss).float().to(comp_device)
        if T % 16 != 0:
            pad_len = 16 - (T % 16)
            pad_data = torch.zeros((pad_len, 6890, 3), dtype=data.dtype, device=comp_device)
            data = torch.cat([data, pad_data], dim=0)

        outputs = []
        with torch.no_grad():
            for i in range(0, data.shape[0], 16):
                chunk = data[i:i+16].unsqueeze(0)
                out = ae.encode(chunk).squeeze(0).cpu().numpy()
                outputs.append(out)
        downsampled = np.concatenate(outputs, axis=0)[:T]

        np.save(pjoin(save_dir, f"{file}"), downsampled)

#################################################################################
#                                Calculate Mean Std                             #
#################################################################################
def mean_variance(data_dir, save_dir):
    file_list = os.listdir(data_dir)
    data_list = []

    for file in tqdm(file_list):
        data = np.load(pjoin(data_dir, file))
        if np.isnan(data).any():
            print(file)
            continue
        data_list.append(data.reshape(-1, 12))

    data = np.concatenate(data_list, axis=0)
    print(data.shape)
    Mean = data.mean(axis=0)
    Std = data.std(axis=0)

    np.save(pjoin(save_dir, 'AE_Mesh_Post_Mean.npy'), Mean)
    np.save(pjoin(save_dir, 'AE_Mesh_Post_Std.npy'), Std)
    print(data.min(),data.max())
    Mean2 = data.mean()
    Std2 = data.std()
    print(Mean, Std)
    print(Mean2, Std2)

    return Mean, Std

if __name__ == '__main__':
    data_dir = 'datasets/HumanML3D/meshes/'
    save_dir = 'datasets/HumanML3D/meshes_after_ae/'

    os.makedirs(save_dir, exist_ok=True)
    downsample(data_dir, save_dir)

    data_dir1 = 'datasets/HumanML3D/meshes_after_ae/'
    save_dir1 = 'checkpoints/t2m/AE_Mesh/'
    mean, std = mean_variance(data_dir1, save_dir1)