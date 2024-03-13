from typing import Optional, Union
import torch_geometric.transforms as T

# import ogb.nodeproppred as ogbn
from torch_geometric.transforms.to_undirected import to_undirected

# from torch_geometric.transforms import to_undirected
from .split import get_idx_split
from .customize_dataset import HSI, HSIMixing

# partially adapted from https://github.com/stadlmax/Graph-Posterior-Network


class BinarizeFeatures:
    """BinarizeFeatures Transformation for data objects in torch-geometric

    When instantiated transformation object is called, features (data.x) are binarized, i.e. non-zero elements are set to 1.
    """

    def __call__(self, data):
        nz = data.x.bool()
        data.x[nz] = 1.0
        return data

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


class ToUndirected(object):
    """ToUndirected Transformation for data objects in torch-geometric

    When instantiated transfomation object is called, the underlying graph in the data  object is converted to an undirected graph,
    so that :math:`(j,i) \in \mathcal{E}` for every edge :math:`(i,j) \in \mathcal{E}`.
    Depending on the representation of the data object, either data.edge_index or data.adj_t is modified.
    """

    def __call__(self, data):
        if "edge_index" in data:
            data.edge_index = to_undirected(data.edge_index)
            # data.edge_index = to_undirected(data)
        if "adj_t" in data:
            data.adj_t = data.adj_t.to_symmetric()
        return data

    def __repr__(self):
        return f"{self.__class__.__name__}()"


def DatasetManager(
    dataset: str,
    root: str,
    split: str = "public",
    train_samples_per_class: Optional[Union[float, int]] = None,
    val_samples_per_class: Optional[Union[float, int]] = None,
    test_samples_per_class: Optional[Union[float, int]] = None,
    train_size: Optional[Union[float, int]] = None,
    val_size: Optional[Union[float, int]] = None,
    test_size: Optional[Union[float, int]] = None,
    seed: int = 0,
    **_,
):
    """DatasetManager

    Method acting as DatasetManager for loading the desired dataset and split when calling with corresponding specifications.
    If the dataset already exists in the root-directory, it is loaded from disc. Otherwise it is downloaded and stored in the specified root-directory.
    Args:
        dataset (str): Name of the dataset to load. Supported datasets are 'CoauthorCS', 'CoauthorPhysics', 'AmazonComputers', 'AmazonPhotos', 'CoraFull', 'CoraML', 'PubMedFull', 'CiteSeerFull', 'Cora', 'PubMed', 'CiteSeer', 'ogbn-arxiv'.
        root (str): Path of data root-directory for either saving or loading dataset.
        split (str, optional): Desired dataset split ('random', or 'public'). Defaults to 'public'.
        train_samples_per_class (Optional[Union[float, int]], optional): number or fraction of training samples per class. Defaults to None.
        val_samples_per_class (Optional[Union[float, int]], optional): number or fraction of validation samples per class. Defaults to None.
        test_samples_per_class (Optional[Union[float, int]], optional): number or fraction of test samples per class. Defaults to None.
        train_size (Optional[int], optional): size of the training set. Defaults to None.
        val_size (Optional[int], optional): size of the validation set. Defaults to None.
        test_size (Optional[int], optional): size of the test set. Defaults to None.
    Raises:
        ValueError: raised if unsupported dataset passed
    Returns:
        dataset: pytorch-geometric dataset as specified
    """

    supported_datasets = {"paviaU", "KSC", "Houston", "urban", "jasper", "IndianPines"}

    default_transform = T.Compose(
        [
            T.NormalizeFeatures(),
            ToUndirected(),
        ]
    )
    if dataset in ["paviaU", "Houston"]:
        assert split in ["random", "semi-public"]
        data = HSI(
            root,
            name=dataset,
            split=split,
            transform=None,
            pre_transform=None,
            seed=seed,
        )

    elif dataset == "KSC":
        assert split == "random"
        data = HSI(
            root,
            name=dataset,
            split=split,
            transform=None,
            pre_transform=None,
            seed=seed,
        )

    elif dataset in ["urban", "jasper"]:
        assert split == "random"
        data = HSIMixing(root, dataset, transform=None, pre_transform=None)

    # default split
    data = get_idx_split(
        data,
        split=split,
        train_samples_per_class=train_samples_per_class,
        val_samples_per_class=val_samples_per_class,
        test_samples_per_class=test_samples_per_class,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
    )
    return data


# if __name__ == '__main__':
