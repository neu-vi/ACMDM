import numpy as np
import os
from os.path import join as pjoin
from tqdm import tqdm

#################################################################################
#                      Calculate Absolute Coordinate Mean Std                   #
#################################################################################
def mean_variance(data_dir, save_dir):
    file_list = os.listdir(data_dir)
    data_list = []

    for file in tqdm(file_list):
        data = np.load(pjoin(data_dir, file))
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=0)
        if np.isnan(data).any():
            print(file)
            continue
        data_list.append(data.reshape(-1, 3))

    data = np.concatenate(data_list, axis=0)
    print(data.shape)
    Mean = data.mean(axis=0)
    Std = data.std(axis=0)

    np.save(pjoin(save_dir, 'Mean_22x3.npy'), Mean)
    np.save(pjoin(save_dir, 'Std_22x3.npy'), Std)

    return Mean, Std

if __name__ == '__main__':
    data_dir1 = 'datasets/HumanML3D/new_joints/'
    save_dir1 = 'datasets/HumanML3D/'
    mean, std = mean_variance(data_dir1, save_dir1)