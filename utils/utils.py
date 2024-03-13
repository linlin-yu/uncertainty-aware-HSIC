import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import numpy as np
import random
from data.dataset_manager import DatasetManager
from data.ood import get_ood_split
from utils.graph_layers import GCNNet, GATNet
from utils.config import DataConfiguration
from torch_geometric.utils import homophily
import math
from torch import Tensor
import networkx as nx
from networkx.algorithms.shortest_paths.unweighted import (
    single_source_shortest_path_length,
)
import torch_geometric.transforms as T
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data


def get_class_names(dataset):
    pavia_maps = {
        0: "Asphalt",
        1: "Meadows",
        2: "Gravel",
        3: "Trees",
        4: "Metal Sheets",
        5: "Bare Soil",
        6: "Bitumen",
        7: "Bricks",
        8: "Shadows",
    }
    indianpines_names = [
        "Alfalfa",
        "Corn-notill",
        "Corn-mintill",
        "Corn",
        "Grass-Pasture",
        "Grass-trees",
        "Grass-pasture-mowed",
        "Hay-windrowed",
        "Oats",
        "Soybean-notill",
        "Soybean-mintill",
        "Soybean-clean",
        "Wheat",
        "Woods",
        "Buildings-Grass-Trees-Drives",
        "Stone-Steel-Towers",
    ]
    indianpines_map = {
        key: value for key, value in zip(np.arange(16), indianpines_names)
    }
    urban_map = {0: "Asphalt", 1: "Grass", 2: "Tree", 3: "Roof"}
    jasper_map = {0: "tree", 1: "water", 2: "dirt", 3: "road"}
    if dataset == "PaviaU":
        return pavia_maps
    elif dataset == "IndianPines":
        return indianpines_map
    elif dataset == "urban":
        return urban_map
    elif dataset == "jasper":
        return jasper_map


def seed_torch(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    return


def load_data(data: dict):
    data_cfg = DataConfiguration(**data)
    dataset = DatasetManager(
        data_cfg.dataset,
        data_cfg.root,
        split=data_cfg.split,
        train_samples_per_class=data_cfg.train_samples_per_class,
        val_samples_per_class=data_cfg.val_samples_per_class,
        test_samples_per_class=data_cfg.test_samples_per_class,
        train_size=data_cfg.train_size,
        val_size=data_cfg.val_size,
        test_size=data_cfg.test_size,
        seed=data_cfg.split_no,
    )

    print(
        "Data split",
        dataset[0].train_mask.sum(),
        dataset[0].val_mask.sum(),
        dataset[0].test_mask.sum(),
    )
    data = dataset[0]
    num_classes = dataset.num_classes
    print("hompophily", homophily(data.edge_index, data.y, method="edge"))
    # print()
    # print(f'Dataset: {dataset}:')
    # print('======================')
    # print(f'Number of graphs: {len(dataset)}')
    # print(f'Number of features: {dataset.num_features}')
    # print(f'Number of classes: {dataset.num_classes}')
    # # Gather some statistics about the graph.
    # print(f'Number of nodes: {data.num_nodes}')
    # print(f'Number of edges: {data.num_edges}')
    print(f"Average node degree: {data.num_edges / data.num_nodes:.2f}")
    # print(f'Contains isolated nodes: {data.has_isolated_nodes()}')
    # print(f'Contains self-loops: {data.has_self_loops()}')
    # print(f'Is undirected: {data.is_undirected()}')
    # print('======================')
    # print('train ID: {}, val ID: {}, test ID: {}.'.format(data.train_mask.sum(),
    #                                                     data.id_val_mask.sum(),
    #                                                     data.id_test_mask.sum()))
    # print('val OOD: {}, test OOD: {}.'.format(data.ood_val_mask.sum(),
    #                                             data.ood_test_mask.sum()))
    return data, num_classes


def load_ood_data(data: dict):
    data_cfg = DataConfiguration(**data)
    dataset = DatasetManager(
        data_cfg.dataset,
        data_cfg.root,
        split=data_cfg.split,
        train_samples_per_class=data_cfg.train_samples_per_class,
        val_samples_per_class=data_cfg.val_samples_per_class,
        test_samples_per_class=data_cfg.test_samples_per_class,
        train_size=data_cfg.train_size,
        val_size=data_cfg.val_size,
        test_size=data_cfg.test_size,
        seed=data_cfg.split_no,
    )
    print(
        "Before OOD split",
        dataset[0].train_mask.sum(),
        dataset[0].val_mask.sum(),
        dataset[0].test_mask.sum(),
    )
    data, num_classes = get_ood_split(
        dataset[0], ood_left_out_classes=data_cfg.ood_left_out_classes
    )
    print()
    print(f"Dataset: {dataset}:")
    print("======================")
    print(f"Number of graphs: {len(dataset)}")
    print(f"Number of features: {dataset.num_features}")
    print(f"Number of classes: {dataset.num_classes}")
    # Gather some statistics about the graph.
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(f"Average node degree: {data.num_edges / data.num_nodes:.2f}")
    print(f"Contains isolated nodes: {data.has_isolated_nodes()}")
    print(f"Contains self-loops: {data.has_self_loops()}")
    print(f"Is undirected: {data.is_undirected()}")
    print("======================")
    print(
        "train ID: {}, val ID: {}, test ID: {}.".format(
            data.id_train_mask.sum(), data.id_val_mask.sum(), data.id_test_mask.sum()
        )
    )
    print(
        "train ood:{}, val OOD: {}, test OOD: {}.".format(
            data.ood_train_mask.sum(), data.ood_val_mask.sum(), data.ood_test_mask.sum()
        )
    )
    # print('labeled node:{}'.format(data.labeled_indices.shape[0]))
    # print('spatial labeled node:{}'.format(data.spatial_labeled_mask.sum()))
    return data, num_classes


def build_model(data, num_classes, device, model_cfg, train_cfg):
    # init model
    seed_torch(model_cfg.seed)
    in_dim = data.num_features
    out_dim = num_classes
    if model_cfg.model_name == "GCN":
        model = GCNNet(
            in_dim,
            model_cfg.hidden_dim,
            out_dim,
            model_cfg.seed,
            model_cfg.drop_prob,
            bias=True,
        ).to(device)
    elif model_cfg.model_name == "GAT":
        model = GATNet(
            in_dim,
            model_cfg.hidden_dim,
            out_dim,
            model_cfg.seed,
            model_cfg.drop_prob,
            bias=True,
        ).to(device)
    else:
        return 0

    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )

    return model, optimizer


def kernel_distance(x: Tensor, sigma: float = 1.0) -> Tensor:
    sigma_scale = 1.0 / (sigma * math.sqrt(2 * math.pi))
    k_dis = torch.exp(-torch.square(x) / (2 * sigma * sigma))
    # return sigma_scale * k_dis
    return k_dis


def compute_kde(data: Data, num_classes: int, sigma: float = 1.0) -> Tensor:
    transform = T.AddSelfLoops()
    data = transform(data)
    n_nodes = data.y.size(0)

    idx_train = torch.nonzero(data.train_mask, as_tuple=False).squeeze().tolist()
    evidence = torch.zeros((n_nodes, num_classes), device=data.y.device)
    G = to_networkx(data, to_undirected=True)

    for idx_t in idx_train:
        distances = single_source_shortest_path_length(G, source=idx_t, cutoff=10)
        distances = torch.Tensor(
            [distances[n] if n in distances else 1e10 for n in range(n_nodes)]
        ).to(data.y.device)
        evidence[:, data.y[idx_t]] += kernel_distance(distances, sigma=sigma)

    return evidence + 1
