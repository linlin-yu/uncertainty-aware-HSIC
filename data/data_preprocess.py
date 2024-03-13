import numpy as np
import os
import time
import scipy.sparse as sp
from scipy import io as scio
from scipy.sparse import coo_matrix
from sklearn.metrics import pairwise
from plotly.subplots import make_subplots
import plotly.graph_objs as go

from data_utils import load_xy, DatasetInfo
from generate_adj_utils import cosine_similar_gpu

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"


def generate_adj(X, sigma=0.1, K=100, adj_type="sparse"):
    assert adj_type in ["sparse", "dense"]
    if adj_type == "sparse":
        # use the faiss package for a sparse adj matrix
        # build the adj based on cosine similarity
        start = time.time()
        D, I = cosine_similar_gpu(X, K)
        end = time.time()
        print("calculate the sparse similarity matrix:", end - start)
        _row = np.array([i for i in range(I.shape[0]) for j in range(I.shape[1])])
        _col = I.reshape(
            -1,
        )
        _data = np.exp(
            -(
                1
                - D.reshape(
                    -1,
                )
                ** 2
            )
            / sigma
        )
        min_w = np.min(_data)
        max_w = np.max(_data)
        w = coo_matrix(
            (_data, (_row, _col)), shape=(X.shape[0], X.shape[0]), dtype=float
        ).tocsr()
        w = 1 / 2 * (w + w.T)
        # scio.savemat(f'{dataset}/raw/adj_{adj_type}.mat', {f'adj_{adj_type}':w})
    elif adj_type == "dense":
        # calculate a dense adj matrix
        start = time.time()
        cos_simiarity = pairwise.cosine_similarity(X, Y=None, dense_output=True)
        end = time.time()
        print("calculate the dense similarity matrix:", end - start)
        w = np.exp(-(1 - cos_simiarity) / sigma)
        min_w = np.min(w)
        max_w = np.max(w)
        # scio.savemat(f'{dataset}/raw/adj_{adj_type}.mat', {f'adj_{adj_type}':w})
    return w, min_w, max_w


def plot_adj(Y, w, min_w, max_w, labeled_indices, datasetinfo, adj_type="sparse"):
    if sp.issparse(w):
        w = w.toarray()
    # plot the weight matrix
    fig = make_subplots(rows=4, cols=4)
    for i in range(0, datasetinfo.num_classes):
        # print(i, int(i/4)+1, (i)%4+1)
        show_index = np.where(Y == i)[0][15]
        orig = np.zeros((datasetinfo.m, datasetinfo.n)).reshape(
            -1,
        )
        orig[labeled_indices] = w[show_index]
        fig.add_trace(
            go.Heatmap(z=orig.reshape(datasetinfo.m, datasetinfo.n), showscale=False),
            row=int(i / 4) + 1,
            col=(i) % 4 + 1,
        )
    fig.update_layout(
        height=1600,
        width=1800,
        title_text=f"weight matrix for one sample in each class min={min_w} max={max_w}, ({adj_type} adjacency matrix)",
    )
    fig.show()
    os.makedirs(f"{datasetinfo.dataset_name}/plots", exist_ok=True)
    fig.write_image(f"{datasetinfo.dataset_name}/plots/adj.pdf")


def get_spatial_labeled_mask(labeled_indices, datasetinfo):
    # consider a mask where the node's left/right/up/down are all labeled
    tmp = np.zeros((datasetinfo.m, datasetinfo.n)).reshape(
        -1,
    )
    tmp[labeled_indices] = 1
    labeled_mask_2D = tmp.reshape(datasetinfo.m, datasetinfo.n)

    tmp_h = np.logical_and(
        labeled_mask_2D[1:-1, :], labeled_mask_2D[:-2, :], labeled_mask_2D[2:, :]
    )
    tmp_w = np.logical_and(
        labeled_mask_2D[:, 1:-1], labeled_mask_2D[:, :-2], labeled_mask_2D[:, 2:]
    )
    spatial_labeled_mask_2D = np.logical_and(tmp_h[:, 1:-1], tmp_w[1:-1, :])
    spatial_labeled_mask_2D = np.pad(
        spatial_labeled_mask_2D, pad_width=1, mode="constant", constant_values=0
    )
    spatial_labeled_mask = spatial_labeled_mask_2D.reshape(
        -1,
    )

    spatial_labeled_mask = spatial_labeled_mask[labeled_indices]
    return spatial_labeled_mask


def get_S_after_permutation(datasetinfo):
    dataset = datasetinfo.dataset_name
    if dataset == "paviaU":
        unmixing_result = scio.loadmat(
            os.path.join("paviaU/unmixing", "unmixing_result.mat")
        )
        S_pred = unmixing_result["S_MBO_fixed"]  # shape:(103,9)
    elif dataset == "KSC":
        unmixing_result = scio.loadmat(
            os.path.join("KSC/unmixing", "unmixing_result.mat")
        )
        S_pred = unmixing_result["S_MBO_fixed"]
    elif dataset == "Houston":
        S_pred = scio.loadmat(os.path.join("Houston/unmixing", "S_init.mat"))["S_init"]

    min_ord = datasetinfo.endmember_map

    # save S seperately for analysis with best permutation
    S_pred = S_pred[:, min_ord]
    return S_pred


def public_split(dataset, labeled_indices):
    assert dataset in ["paviaU", "Houston"]
    if dataset == "Houston":
        label_va = scio.loadmat(
            os.path.join("../raw_data/Houston", "Houston_VA_gt.mat")
        )["Houston"]
        label_tr = scio.loadmat(
            os.path.join("../raw_data/Houston", "Houston_TR_gt.mat")
        )["Houston"]
        va_indices = np.nonzero(label_va)
        tr_indices = np.nonzero(label_tr)
        label = np.zeros_like(label_va)
        label[va_indices] = label_va[va_indices]
        label[tr_indices] = label_tr[tr_indices]
        print("labeled nodes:", np.count_nonzero(label))
        # scio.savemat(os.path.join('Houston/raw', 'Houston_gt.mat'), {'Houston_gt':label})
    elif dataset == "paviaU":
        # public train/test split from https://github.com/danfenghong/IEEE_TGRS_GCN
        label_tr = scio.loadmat(os.path.join("../raw_data/paviaU", "TRpaviaU.mat"))[
            "TRpaviaU"
        ]
        label_va = scio.loadmat(os.path.join("../raw_data/paviaU", "TSpaviaU.mat"))[
            "TSpaviaU"
        ]
        va_indices = np.nonzero(label_va)
        tr_indices = np.nonzero(label_tr)

    val_mask = np.zeros_like(label_va)
    train_mask = np.zeros_like(label_tr)
    val_mask[va_indices] = 1
    train_mask[tr_indices] = 1
    train_mask[va_indices] = 0

    # to 1D mask with only labeled indices
    val_mask_1D = val_mask.reshape(
        -1,
    )[labeled_indices]
    train_mask_1D = train_mask.reshape(
        -1,
    )[labeled_indices]

    np.savez(f"{dataset}/raw/mask", train_mask=train_mask_1D, val_test_mask=val_mask_1D)


def main():

    # dataset in ['KSC', 'Houston', 'paviaU']
    dataset = "Houston"
    # dataset = 'paviaU'
    # dataset = 'Houston'
    datasetinfo = DatasetInfo(dataset)
    os.makedirs(f"{dataset}/raw", exist_ok=True)

    # load data
    X_1d_withlabel, Y_1d_withlabel, labeled_indices = load_xy(dataset)
    Y_1d_withlabel = Y_1d_withlabel - 1

    # generate sparse adj
    sigma = datasetinfo.sigma
    adj_type = "sparse"
    K = 50
    w, min_w, max_w = generate_adj(X_1d_withlabel, sigma=sigma, K=K, adj_type=adj_type)
    plot_adj(
        Y_1d_withlabel,
        w,
        min_w,
        max_w,
        labeled_indices,
        datasetinfo=datasetinfo,
        adj_type=adj_type,
    )

    # get spatial labeled mask
    spatial_labeled_mask = get_spatial_labeled_mask(labeled_indices, datasetinfo)

    # save data and adj as "data.npz"
    np.savez(
        f"{dataset}/raw/data",
        X_1d_withlabel=X_1d_withlabel,
        Y_1d_withlabel=Y_1d_withlabel,
        labeled_indices=labeled_indices,
        spatial_labeled_mask=spatial_labeled_mask,
    )
    sp.save_npz(f"{dataset}/raw/adj", w)

    # get the ordered endmember S and save it
    S_pred = get_S_after_permutation(datasetinfo)
    scio.savemat(os.path.join(f"{dataset}/raw", "S_pred.mat"), {"S_pred": S_pred})

    # save public split for training vs val+test
    if dataset in ["paviaU", "Houston"]:
        public_split(dataset, labeled_indices)


if __name__ == "__main__":
    main()
