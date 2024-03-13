import numpy as np
import os
from scipy import io as scio

import plotly.graph_objs as go
from sklearn_extra.cluster import KMedoids


# load the data features and labels in 1D format with only labeled nodes
def load_xy(dataset="paviaU"):
    assert dataset in ["paviaU", "KSC", "Houston"]
    if dataset == "paviaU":
        data = scio.loadmat(os.path.join("../raw_data/paviaU", "paviaU.mat"))
        label = scio.loadmat(os.path.join("../raw_data/paviaU", "paviaU_gt.mat"))
        X = data["paviaU"]
        Y = label["paviaU_gt"]
    elif dataset == "KSC":
        data = scio.loadmat(os.path.join("../raw_data/KSC", "KSC_corrected.mat"))
        label = scio.loadmat(os.path.join("../raw_data/KSC", "KSC_gt.mat"))
        X = data["KSC"]
        Y = label["KSC_gt"]
    elif dataset == "Houston":
        data = scio.loadmat(os.path.join("../raw_data/Houston", "Houston.mat"))
        label = scio.loadmat(os.path.join("../raw_data/Houston", "Houston_gt.mat"))
        X = data["Houston"]
        Y = label["Houston_gt"]

    Y_1d = np.int32(
        Y.reshape(
            -1,
        )
    )
    labeled_indices = np.where(Y_1d > 0)[0]
    Y_1d_withlabel = Y_1d[labeled_indices]
    X_1d = np.float32(X.reshape(-1, X.shape[-1]))
    max_x = np.max(X_1d)
    X_1d = X_1d / max_x
    X_1d_withlabel = X_1d[labeled_indices, :]
    return X_1d_withlabel, Y_1d_withlabel, labeled_indices


def load_unminxg_X_Y(dataset="jasper"):
    # read the features and labels from the mat files
    assert dataset in ["jasper", "urban"]
    if dataset == "jasper":
        gd = scio.loadmat(os.path.join("jasper/raw", "end4.mat"))
        gd_A = gd["A"].T
        gd_A = (
            gd_A.astype(float)
            .reshape(100, 100, gd_A.shape[-1], order="F")
            .reshape(-1, gd_A.shape[-1], order="C")
            .astype(float)
        )
        # endmemberS = gd['M'].T
        data = scio.loadmat(os.path.join("jasper/raw", "jasperRidge2_R198.mat"))
        # load and scale the features
        x = data["Y"].T
        X = x.reshape(100, 100, x.shape[-1], order="F").reshape(
            -1, x.shape[-1], order="C"
        )
    x = x.astype(float)
    y = np.argmax(gd_A, axis=-1)
    maxValue = np.max(x)
    X_2d = x / maxValue
    # all the ID classes and started from 1
    Y_2d = y.astype(int)
    # fig = go.Figure(go.Heatmap(z=Y))
    # fig.update_layout(height=400, width=400, title_text="label map")
    # fig.show()
    return X_2d, Y_2d


class DatasetInfo:
    def __init__(self, dataset) -> None:
        if dataset == "paviaU":
            self.dataset_name = "paviaU"
            self.label_values = [
                "Undefined",
                "Asphalt",
                "Meadows",
                "Gravel",
                "Trees",
                "Painted metal sheets",
                "Bare Soil",
                "Bitumen",
                "Self-Blocking Bricks",
                "Shadows",
            ]
            self.sigma = 0.5
            self.m, self.n = 610, 340
            self.num_classes = 9
            self.ignored_labels = [0]
            self.endmember_map = (7, 3, 2, 6, 1, 0, 5, 8, 4)
        elif dataset == "KSC":
            self.dataset_name = "KSC"
            self.label_values = [
                "Undefined",
                "Scrub",
                "Willow swamp",
                "Cabbage palm hammock",
                "Cabbage palm/oak hammock",
                "Slash pine",
                "Oak/broadleaf hammock",
                "Hardwood swamp",
                "Graminoid marsh",
                "Spartina marsh",
                "Cattail marsh",
                "Salt marsh",
                "Mud flats",
                "Wate",
            ]
            self.sigma = 0.5
            self.m, self.n = 512, 614
            self.num_classes = 13
            self.ignored_labels = [0]
            self.endmember_map = [12, 1, 10, 4, 5, 9, 0, 6, 7, 8, 2, 11, 3]

        elif dataset == "Houston":
            self.dataset_name = "Houston"
            self.label_values = [
                "Unclassified",
                "Healthy grass",
                "Stressed grass",
                "Artificial turf",
                "Evergreen trees",
                "Deciduous trees",
                "Bare earth",
                "Water",
                "Residential buildings",
                "Non-residential buildings",
                "Roads",
                "Sidewalks",
                "Crosswalks",
                "Major thoroughfares",
                "Highways",
                "Railways",
                "Paved parking lots",
                "Unpaved parking lots",
                "Cars",
                "Trains",
                "Stadium seats",
            ]
            self.sigma = 0.1
            self.m, self.n = 1905, 349
            self.num_classes = 15
            self.ignored_labels = [0]
            self.endmember_map = [3, 14, 0, 7, 10, 9, 5, 13, 11, 6, 12, 2, 1, 8, 4]


def generate_S(X_2d, Y_2d, datasetinfo, s_type="medoids"):
    assert s_type in ["random", "medoids"]
    # (dict and the key is the label, values is the material information)
    endmemeberS = dict()
    to_save_endmemberS = dict()
    if datasetinfo.ignored_labels is not None:
        start, end = 1, datasetinfo.num_classes + 1
    else:
        start, end = 0, datasetinfo.num_classes + 1
    if s_type == "medoids":
        # use the median value as the ground-truth of the endmemberS with the KMedoids method
        for i in range(start, end):
            # print(i)
            endmemeberS[i] = X_2d[np.where(Y_2d == i)[0], :]
            kmedoids = KMedoids(n_clusters=1, random_state=0).fit(endmemeberS[i])
            to_save_endmemberS[i] = kmedoids.cluster_centers_.flatten()
    elif s_type == "random":
        # randomly pick one represented element as the endmemberS
        for i in range(start, end):
            indexes = np.where(Y_2d == i)[0]
            index = np.random.choice(indexes, 1)[0]
            to_save_endmemberS[i] = X_2d[index, :]
    # save a matrix for endmemberS
    gd_endmemberS = [to_save_endmemberS[key] for key in to_save_endmemberS.keys()]
    gd_endmemberS = np.stack(gd_endmemberS)
    print("endmemberS shape:", gd_endmemberS.shape)
    # scio.savemat('IndianPines/raw/S_medoids.mat', {'S_medoids':gd_endmemberS.T})
    scio.savemat(
        f"{datasetinfo.dataset_name}/analysis/S_{s_type}.mat",
        {f"S_{s_type}": gd_endmemberS.T},
    )


def plot_endmemberS(datasetinfo, s_type):
    assert s_type in ["random", "medoids", "init", "MBO_fixed", "pred"]
    gd_endmemberS = scio.loadmat(
        os.path.join(f"{datasetinfo.dataset_name}/analysis", f"S_{s_type}.mat")
    )[f"S_{s_type}"]
    gd_endmemberS = gd_endmemberS.T
    print(gd_endmemberS.shape)
    # # plot the meds comparing for all id_classes
    fig = go.Figure()
    for class_id in range(gd_endmemberS.shape[0]):
        fig.add_trace(
            go.Scatter(
                x=[i for i in range(gd_endmemberS.shape[1])],
                y=gd_endmemberS[class_id, :],
                name=datasetinfo.label_values[class_id + 1],
                mode="lines",
            )
        )
    fig.update_layout(title=f"endmembers S_{s_type}", width=2000, height=1000)
    fig.write_image(
        f"{datasetinfo.dataset_name}/analysis/plots/endmemberS_{s_type}.pdf"
    )
    fig.show()
