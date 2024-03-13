import matplotlib.pyplot as plt
from sklearn import metrics
from data.data_utils import DatasetInfo
import numpy as np
import torch


def plot_id_performance(alpha, data, num_classes, dataset, n_bins=10):
    datasetinfo = DatasetInfo(dataset)
    # read data
    alpha_0 = alpha.sum(dim=-1)
    alpha_0_dim = alpha.sum(dim=-1, keepdim=True)
    y_pred_soft = alpha / alpha_0_dim
    y_pred_hard = y_pred_soft.argmax(dim=1)

    id_fig = plt.figure(figsize=(10, 10))

    # plot the prediction map
    inv_class_mapping = {v: k for k, v in data.class_mapping.items()}
    idpred_fig = id_fig.add_subplot(2, 1, 1)
    pred_2D = np.zeros((datasetinfo.m, datasetinfo.n))
    pred_1D = pred_2D.reshape(
        -1,
    )
    pred_1D[data.labeled_indices] = y_pred_hard + 1
    pred_2D = pred_1D.reshape(datasetinfo.m, datasetinfo.n)
    heatmap = idpred_fig.imshow(pred_2D, cmap="cividis", aspect="auto")
    # Customize the colorbar ticks and labels
    ticktext = [f"0: {datasetinfo.label_values[0]}"]
    for i in range(1, num_classes + 1):
        ticktext.append(
            f"{i}: {datasetinfo.label_values[inv_class_mapping[i - 1] + 1]}"
        )
    cbar = plt.colorbar(heatmap, ticks=np.arange(num_classes + 1))
    cbar.set_ticklabels(ticktext)

    # plot the calibration figure
    calibration_fig = id_fig.add_subplot(2, 1, 2)

    y_pred_hard_test = y_pred_hard[data.id_test_mask]
    max_prob_test = torch.max(y_pred_soft, dim=-1)[0][data.id_test_mask]
    ground_truth_test = data.y[data.id_test_mask]
    confidence_all, confidence_acc = np.zeros(n_bins), np.zeros(n_bins)

    for index, value in enumerate(max_prob_test):
        # value -= suboptimal_prob[index]
        interval = int(value / (1 / n_bins) - 0.0001)
        confidence_all[interval] += 1
        if y_pred_hard_test[index] == ground_truth_test[index]:
            confidence_acc[interval] += 1
    for index, value in enumerate(confidence_acc):
        if confidence_all[index] == 0:
            confidence_acc[index] = 0
        else:
            confidence_acc[index] /= confidence_all[index]
    start = np.around(1 / n_bins / 2, 3)
    step = np.around(1 / n_bins, 3)
    start = np.around(1 / n_bins / 2, 3)
    step = np.around(1 / n_bins, 3)

    calibration_fig.bar(
        np.around(np.arange(start, 1.0, step), 3),
        confidence_acc,
        alpha=0.7,
        width=0.03,
        color="dodgerblue",
        label="Outputs",
    )
    calibration_fig.bar(
        np.around(np.arange(start, 1.0, step), 3),
        np.around(np.arange(start, 1.0, step), 3),
        alpha=0.7,
        width=0.03,
        color="lightcoral",
        label="Expected",
    )
    calibration_fig.plot([0, 1], [0, 1], ls="--", c="k")
    calibration_fig.set(xlabel="Confidence", ylabel="Accuracy")
    calibration_fig.title.set_text("calibration for ID")

    id_fig.suptitle(f"ID Prediction Performance")
    return id_fig


def plot_ood_performance(alpha, data, num_classes, model_name, dataset):
    datasetinfo = DatasetInfo(dataset)
    # read data
    alpha_0 = alpha.sum(dim=-1)
    alpha_0_dim = alpha.sum(dim=-1, keepdim=True)
    y_pred_soft = alpha / alpha_0_dim
    y_pred_hard = y_pred_soft.argmax(dim=1)

    # ROC plots
    vacuity_scores = (num_classes / alpha_0).flatten()
    id_mask = data.id_test_mask
    ood_mask = data.ood_test_mask
    mask = ood_mask | id_mask
    corrects = ood_mask
    ns_vacuity_scores = np.zeros_like(vacuity_scores)
    # plot roc and pr
    curve_fig = plt.figure(figsize=(25, 25))
    roc_fig = curve_fig.add_subplot(3, 2, 1)
    fpr, tpr, _ = metrics.roc_curve(corrects[mask], vacuity_scores[mask])
    ns_fpr, ns_tpr, _ = metrics.roc_curve(corrects[mask], ns_vacuity_scores[mask])
    # fpr, tpr, precision, recall = get_auc_metrics(vacuity, data.ood_test_mask, data.id_test_mask)
    roc_fig.plot(ns_fpr, ns_tpr, linestyle="--", label="No Skill")
    roc_fig.plot(fpr, tpr, marker=".", label=model_name)
    roc_fig.title.set_text(f"ROC={metrics.auc(fpr, tpr):.4f}")
    roc_fig.set(xlabel="False Positive Rate", ylabel="True Postive Rate")
    roc_fig.legend()

    # plot pr
    pr_fig = curve_fig.add_subplot(3, 2, 2)
    precision, recall, _ = metrics.precision_recall_curve(
        corrects[mask], vacuity_scores[mask]
    )
    ## no skill is the proportion of postive cases
    no_skill = np.count_nonzero(corrects[mask]) / np.count_nonzero(mask)
    # print(no_skill)
    pr_fig.plot([0, 1], [no_skill, no_skill], linestyle="--", label="No Skill")
    pr_fig.plot(recall, precision, marker=".", label=model_name)
    pr_fig.title.set_text(f"PR Curve (PR={metrics.auc(recall, precision):.4f})")
    pr_fig.set(xlabel="Recall", ylabel="Precision")
    pr_fig.legend()

    # plot the vacuity map
    va_fig = curve_fig.add_subplot(3, 2, 3)
    vacuity_2D = np.zeros((datasetinfo.m, datasetinfo.n))
    vacuity_1D = vacuity_2D.reshape(
        -1,
    )
    vacuity_1D[data.labeled_indices] = vacuity_scores
    vacuity_2D = vacuity_1D.reshape(datasetinfo.m, datasetinfo.n)
    heatmap = va_fig.imshow(vacuity_2D, cmap="viridis", aspect="auto")
    cbar = plt.colorbar(heatmap)
    # cbar.set_ticklabels(['id', 'ood'])
    va_fig.title.set_text("predicted vacuity score")

    # plot ground truth vacuity map
    vagd_fig = curve_fig.add_subplot(3, 2, 4)
    vacuity_scores_gd = np.zeros_like(vacuity_scores)
    vacuity_scores_gd[data.ood_mask] = 1
    vacuity_2D = np.zeros((datasetinfo.m, datasetinfo.n))
    vacuity_1D = vacuity_2D.reshape(
        -1,
    )
    vacuity_1D[data.labeled_indices] = vacuity_scores_gd
    vacuity_2D = vacuity_1D.reshape(datasetinfo.m, datasetinfo.n)
    heatmapgd = vagd_fig.imshow(vacuity_2D, cmap="viridis", aspect="auto")
    cbar = plt.colorbar(heatmapgd)
    # cbar.set_ticklabels(['id', 'ood'])
    vagd_fig.title.set_text("ground-truth vacuity score")

    # plot the vacuity comparision
    vacomp_fig = curve_fig.add_subplot(3, 1, 3)
    vacuity_id = np.array(vacuity_scores[id_mask])
    vacuity_ood = np.array(vacuity_scores[ood_mask])
    vacomp_fig.hist(
        vacuity_id, bins=np.arange(0, 1.001, 0.001), alpha=0.5, label="id_nodes"
    )
    vacomp_fig.hist(
        vacuity_ood, bins=np.arange(0, 1.001, 0.001), alpha=0.5, label="ood_nodes"
    )
    vacomp_fig.title.set_text("Comparision of vacuity score between ID and OOD nodes")
    vacomp_fig.legend()

    curve_fig.suptitle(f"OoD Detection Performance")
    return curve_fig
