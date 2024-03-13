import numpy as np
import os, pickle
import os.path as osp
import torch
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, to_undirected
import scipy.sparse as sp
from scipy import io as scio
from typing import Optional, Callable, List, Union
from .split import get_idx_split
from .data_utils import DatasetInfo

# There are three datasets:
# 1. paviaU
# 2. KSC
# 3. Houston


# Dataset split:
# 1. random
# 1.1 based on number of damples per class
# 1.2 based on ratio per class
# 2. public
class HSI(InMemoryDataset):
    def __init__(
        self,
        root: str,
        name: str,
        split: str = "random",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        seed: int = 0,
    ) -> None:
        torch.manual_seed(seed)
        assert name in ["paviaU", "KSC", "Houston"]
        self.name = name
        self.split = split.lower()
        assert self.split in ["random", "semi-public"]
        self.datasetinfo = DatasetInfo(self.name)
        super().__init__(root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])
        data = self.get(0)

        data.hw = (self.datasetinfo.m, self.datasetinfo.n)

        self.data, self.slices = self.collate([data])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self) -> List[str]:
        return ["data.npz", "adj.npz"]

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def download(self):
        print("No need to download")

    def get_endmemberS(self):
        endmemberS = scio.loadmat(os.path.join(self.raw_dir, "S_pred.mat"))
        endmemberS = endmemberS["S_pred"].T

        endmemberS = endmemberS[self.datasetinfo.endmember_map, :]
        return endmemberS

    def process(self):
        """
        :ALL_Y Classification Labels
        :ALL_X Feature matrix including 103 dimension denoting pixel information with different wave length
        :adj Sparse matrix denoting the weighted adjancency matrix (no-self-connection)
        """

        # load features, labels, adjacency matrix, labeled mask  (single dimension)
        raw_data = np.load(os.path.join(self.raw_dir, "data.npz"))
        features = raw_data["X_1d_withlabel"]
        labels = raw_data["Y_1d_withlabel"]
        labeled_indices = raw_data["labeled_indices"]
        spatial_labeled_mask = raw_data["spatial_labeled_mask"]
        adj = sp.load_npz(os.path.join(self.raw_dir, "adj.npz"))
        adj = adj.tocoo()

        # prepare data object by building tensors from numpy
        features = torch.from_numpy(features)
        y = torch.from_numpy(labels).to(torch.long)
        row = torch.from_numpy(adj.row).to(torch.long)
        col = torch.from_numpy(adj.col).to(torch.long)
        edge_weight = torch.from_numpy(adj.data).to(torch.float)
        edge_index = torch.stack([row, col], dim=0)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_attr=edge_weight)
        edge_index, edge_weight = to_undirected(
            edge_index, edge_attr=edge_weight, num_nodes=features.size(0)
        )
        # build the data object
        data = Data(x=features, edge_index=edge_index, y=y)
        data.edge_weight = edge_weight
        # log necessary information
        data.labeled_indices = torch.from_numpy(labeled_indices)
        data.spatial_labeled_mask = torch.from_numpy(spatial_labeled_mask).to(
            torch.bool
        )
        # one-hot ground-truth abundance map
        data.abundance = F.one_hot(data.y, max(data.y) + 1).to(torch.float)

        # load endmemberS after permutation
        data.endmemberS = torch.from_numpy(self.get_endmemberS()).to(torch.float)

        # load semi-public split for Houston and paviaU dataset
        if self.name in ["paviaU", "Houston"]:
            mask = np.load(os.path.join(self.raw_dir, "mask.npz"))
            train_mask = np.array(mask["train_mask"], dtype=bool)
            val_test_mask = np.array(mask["val_test_mask"], dtype=bool)
            data.train_mask = torch.from_numpy(train_mask).to(torch.bool)
            data.val_test_mask = torch.from_numpy(val_test_mask).to(torch.bool)

        data = data if self.pre_transform is None else self.pre_transform(data)
        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])


class HSIMixing(InMemoryDataset):
    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ) -> None:
        assert name in ["urban"]
        self.name = name

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        data = self.get(0)
        if name == "urban":
            data.hw = (307, 307)
        self.data, self.slices = self.collate([data])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self) -> List[str]:
        if self.name == "urban":
            return ["end4_groundTruth.mat", "Urban_R162.mat", f"{self.name}_adj.npz"]
        elif self.name == "jasper":
            return ["end4.mat", "jasperRidge2_R198.mat", f"{self.name}_adj.npz"]

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def download(self):
        print("No need to download")

    def get_endmemberS(self):
        return self.endmemberS

    def process(self):
        """
        :ALL_Y Classification Labels
        :ALL_X Feature matrix including 103 dimension denoting pixel information with different wave length
        :adj Sparse matrix denoting the weighted adjancency matrix (no-self-connection)
        """
        if self.name == "urban":
            gd = scio.loadmat(os.path.join(self.raw_dir, "end4_groundTruth.mat"))
            gd_A = gd["A"].T
            gd_A = (
                gd_A.astype(float)
                .reshape(307, 307, gd_A.shape[-1], order="F")
                .reshape(-1, gd_A.shape[-1], order="C")
            )
            endmemberS = gd["M"].T
            data = scio.loadmat(os.path.join(self.raw_dir, "Urban_R162.mat"))
            # load and scale the features
            x = data["Y"].T
            x = x.reshape(307, 307, x.shape[-1], order="F").reshape(
                -1, x.shape[-1], order="C"
            )
        elif self.name == "jasper":
            gd = scio.loadmat(os.path.join(self.raw_dir, "end4.mat"))
            gd_A = gd["A"].T
            gd_A = (
                gd_A.astype(float)
                .reshape(100, 100, gd_A.shape[-1], order="F")
                .reshape(-1, gd_A.shape[-1], order="C")
            )
            endmemberS = gd["M"].T
            data = scio.loadmat(os.path.join(self.raw_dir, "jasperRidge2_R198.mat"))
            # load and scale the features
            x = data["Y"].T
            x = x.reshape(100, 100, x.shape[-1], order="F").reshape(
                -1, x.shape[-1], order="C"
            )
        x = x.astype(float)
        y = np.argmax(gd_A, axis=-1)
        maxValue = np.max(x)
        features = x / maxValue
        features = torch.from_numpy(features).to(torch.float)
        # all the ID classes and started from 1
        labels = y.astype(int)
        y = torch.from_numpy(labels).to(torch.long)
        # load sparse adj matrix which is weight matix rather than a binary matrix
        loader = scio.loadmat(os.path.join(self.raw_dir, f"adj_sparse.mat"))
        loader = loader["adj_sparse"]
        adj = loader.tocoo()
        # adj = sp.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
        #               shape=loader['shape']).tocoo()
        row = torch.from_numpy(adj.row).to(torch.long)
        col = torch.from_numpy(adj.col).to(torch.long)
        edge_weight = torch.from_numpy(adj.data).to(torch.float)
        edge_index = torch.stack([row, col], dim=0)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_attr=edge_weight)
        edge_index, edge_weight = to_undirected(
            edge_index, edge_attr=edge_weight, num_nodes=features.size(0)
        )
        data = Data(x=features, edge_index=edge_index, y=y)
        data.edge_weight = edge_weight
        # load endmemberS after permutation
        data.endmemberS = torch.from_numpy(endmemberS).to(torch.float)
        data.abundance = torch.from_numpy(gd_A).to(torch.float)

        data = data if self.pre_transform is None else self.pre_transform(data)
        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])
