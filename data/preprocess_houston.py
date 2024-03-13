# Copid from https://github.com/shuguang-52/2019-RemoteSens-AUSSC/blob/master/geotif2mat.py
from osgeo.gdal_array import DatasetReadAsArray
from osgeo import gdal
import scipy.io as sio


def extract_features(input_file, output_fiule):
    houston = gdal.Open(input_file)  # Change it
    data = DatasetReadAsArray(houston)
    print(data.shape, data.dtype)

    houston = data.transpose()
    print(houston.shape)

    sio.savemat(output_file, {"Houston": houston})


# extract features from the tif file
features_file = "./Houston/original/2013_IEEE_GRSS_DF_Contest_CASI.tif"
output_file = "Houston.mat"
extract_features(features_file, output_file)

# extract ground truth label from the roi file
# following the instruction from https://github.com/shuguang-52/2019-RemoteSens-AUSSC with help of software "ENVI"

gt_TR_file = "./Houston/raw/Houston_TR_gt.tif"
output_file = "Houston_TR_gt.mat"
extract_features(gt_TR_file, output_file)

gt_VA_file = "./Houston/raw/Houston_VA_gt.tif"
output_file = "Houston_VA_gt.mat"
extract_features(features_file, output_file)

# transform the label to one label set and train mask with val mask
# Note: we remove three test nodes, which is also in the training set

import numpy as np
import os
from scipy import io as scio

label_va = scio.loadmat(os.path.join("Houston/analysis", "Houston_VA_gt.mat"))[
    "Houston"
]
label_tr = scio.loadmat(os.path.join("Houston/analysis", "Houston_TR_gt.mat"))[
    "Houston"
]

va_indices = np.nonzero(label_va)
tr_indices = np.nonzero(label_tr)

label = np.zeros_like(label_va)
label[va_indices] = label_va[va_indices]
label[tr_indices] = label_tr[tr_indices]
print("labeled nodes:", np.count_nonzero(label))
scio.savemat(os.path.join("Houston/raw", "Houston_gt.mat"), {"Houston_gt": label})

val_mask = np.zeros_like(label_va)
train_mask = np.zeros_like(label_tr)
val_mask[va_indices] = 1
train_mask[tr_indices] = 1
train_mask[va_indices] = 0
scio.savemat(
    os.path.join("Houston/raw", "Houston_val_mask.mat"), {"Houston_val_mask": val_mask}
)
scio.savemat(
    os.path.join("Houston/raw", "Houston_train_mask.mat"),
    {"Houston_train_mask": train_mask},
)
