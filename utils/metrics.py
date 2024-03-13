import torch
import time
from torch import Tensor
from torch.nn import functional as F
from torch_geometric.data import Data
import numpy as np
from utils.loss import cross_entropy
from torchmetrics.functional import (
    f1_score,
    cohen_kappa,
    confusion_matrix,
    accuracy,
    auroc,
    precision_recall_curve,
)
from torchmetrics.utilities.compute import auc


def Bal(b_i, b_j):
    result = 1 - torch.abs(b_i - b_j) / (b_i + b_j)
    return result


"""
details can be found in https://arxiv.org/pdf/2010.12783.pdf
"""


def get_dissonance(alpha, alpha_0):
    evidence = alpha - 1
    belief = evidence / alpha_0
    dis_un = torch.zeros_like(alpha_0)
    for i in range(belief.shape[1]):
        bi = belief[:, i].reshape(-1, 1)
        term_Bal = torch.zeros_like(alpha_0)
        term_bj = torch.zeros_like(alpha_0)
        for j in range(belief.shape[1]):
            if j != i:
                bj = belief[:, j].reshape(-1, 1)
                term_Bal += bj * Bal(bi, bj)
                term_bj += bj
        dis_ki = bi * term_Bal / term_bj
        dis_un += dis_ki
    return dis_un


def get_metrics(
    metrics: list,
    data: Data,
    logit: Tensor = None,
    alpha: Tensor = None,
    mode: str = "test",
    task: str = "ood",
):
    """get the metrics dictionary from a list of strings naming those
    Args:
        metrics (list): list of metric names
        alpha (Tensor): predicted alpha from the model
        data (Data): get the label information
        mode (str): train/val/test
    Returns:
        dict: result of the evaluation metrics
    """

    assert mode in ["train", "validation", "test"]
    assert task in ["ood", "mis"]
    if metrics is None:
        metrics = {}

    metrics_dict = {}

    y_true = data.y

    if alpha != None:
        P = alpha.shape[-1]
        alpha_0 = alpha.sum(dim=-1, keepdim=True)
        y_pred_soft = alpha / alpha_0
        y_pred_hard = y_pred_soft.argmax(dim=1)
        vacuity_scores = (P / alpha_0).flatten()
        dissonance_scores = get_dissonance(alpha, alpha_0).flatten()
    if logit != None:
        prob = F.softmax(logit, dim=-1)
        P = prob.shape[-1]
        y_pred_soft = prob
        y_pred_hard = y_pred_soft.argmax(dim=1)

    entropy_scores = get_entropy(y_pred_soft).flatten()
    aleatoric_scores = get_aleatoric(y_pred_soft).flatten()

    if task == "ood":
        if mode == "train":
            id_mask = data.id_train_mask
            ood_mask = None
            mask = id_mask
        elif mode == "validation":
            id_mask = data.id_val_mask
            ood_mask = data.ood_val_mask
            mask = ood_mask | id_mask
        elif mode == "test":
            id_mask = data.id_test_mask
            ood_mask = data.ood_test_mask
            mask = ood_mask | id_mask
        corrects = ood_mask
    elif task == "mis":
        if mode == "train":
            id_mask = data.train_mask
            ood_mask = None
        if mode == "validation":
            id_mask = data.val_mask
            ood_mask = None
        if mode == "test":
            id_mask = data.test_mask
            ood_mask = None
        mask = id_mask
        corrects = y_true != y_pred_hard

    for metric in metrics:
        metric = metric.lower()
        # id metric
        if metric == "id_oa":
            metrics_dict[metric] = get_id_oa(y_pred_hard[id_mask], y_true[id_mask], P)
        if metric == "id_f1":
            metrics_dict[metric] = get_id_f1(y_pred_hard[id_mask], y_true[id_mask], P)
        if metric == "id_ce":
            metrics_dict[metric] = get_id_ce(y_pred_soft[id_mask], y_true[id_mask])
        if metric == "id_kappa":
            metrics_dict[metric] = get_id_kappa(
                y_pred_hard[id_mask], y_true[id_mask], P
            )
        if metric == "id_aa":
            metrics_dict[metric] = get_id_aa(y_pred_hard[id_mask], y_true[id_mask], P)
        if metric == "id_acc_classwise":
            results = get_id_acc_classwise(y_pred_hard[id_mask], y_true[id_mask], P)
            for i in range(P):
                metrics_dict[f"{metric}_{i}"] = results[i]
        # uncertainty metric
        if metric == "vacuity_roc":
            metrics_dict[metric] = get_roc(vacuity_scores[mask], corrects[mask])
        if metric == "vacuity_pr":
            metrics_dict[metric] = get_pr(vacuity_scores[mask], corrects[mask])
            # metrics_dict[metric] = get_ood_pr(score_type = 'APR', scores=vacuity_scores[mask], corrects=corrects[mask])
        if metric == "dissonance_roc":
            metrics_dict[metric] = get_roc(dissonance_scores[mask], corrects[mask])
        if metric == "dissonance_pr":
            metrics_dict[metric] = get_pr(dissonance_scores[mask], corrects[mask])
            # metrics_dict[metric] = get_ood_auc(score_type = 'APR', scores=dissonance_scores[mask], corrects=corrects[mask])
        if metric == "entropy_roc":
            metrics_dict[metric] = get_roc(entropy_scores[mask], corrects[mask])
        if metric == "entropy_pr":
            metrics_dict[metric] = get_pr(entropy_scores[mask], corrects[mask])
            # metrics_dict[metric] = get_ood_auc(score_type = 'APR', scores=entropy_scores[mask], corrects=corrects[mask])
        if metric == "aleatoric_roc":
            metrics_dict[metric] = get_roc(aleatoric_scores[mask], corrects[mask])
        if metric == "aleatoric_pr":
            metrics_dict[metric] = get_pr(aleatoric_scores[mask], corrects[mask])
            # metrics_dict[metric] = get_ood_auc(score_type = 'APR', scores=aleatoric_scores[mask], corrects=corrects[mask])

    return metrics_dict


def get_id_oa(y_pred_hard, y_true, P, **kwargs):
    return accuracy(
        y_pred_hard, y_true, num_classes=P, task="multiclass", average="micro"
    )


def get_id_f1(y_pred_hard, y_true, P, **kwargs):
    return f1_score(
        y_pred_hard, y_true, num_classes=P, task="multiclass", average="macro"
    )


def get_id_ce(y_pred_soft, y_true, **kwargs):
    return cross_entropy(y_pred_soft, y_true)


def get_id_kappa(y_pred_hard, y_true, P, **kwargs):
    return cohen_kappa(y_pred_hard, y_true, num_classes=P, task="multiclass")


def get_id_aa(y_pred_hard, y_true, P, **kwargs):
    # cm = confusion_matrix(y_pred_hard, y_true, num_classes=P)
    # AA = cm.diagonal()/cm.sum(axis=1)
    # aa_mean = torch.mean(AA)
    return accuracy(
        y_pred_hard, y_true, num_classes=P, task="multiclass", average="macro"
    )


def get_id_acc_classwise(y_pred_hard, y_true, P, **kwargs):
    # cm = confusion_matrix(y_pred_hard, y_true, num_classes=P)
    # AA = cm.diagonal()/cm.sum(axis=1)
    return accuracy(
        y_pred_hard, y_true, num_classes=P, task="multiclass", average="none"
    )


def get_entropy(prob):
    entropy = -prob * (torch.log(prob))
    entropy_un = torch.sum(entropy, dim=1, keepdims=True)
    return entropy_un


def get_aleatoric(prob):
    aleatoric_neg = torch.max(prob, dim=-1, keepdims=True)
    return -aleatoric_neg.values


def get_roc(scores, corrects):
    return auroc(scores, corrects, task="binary")


def get_pr(scores, corrects):
    prec, rec, _ = precision_recall_curve(scores, corrects, task="binary")
    return auc(rec, prec, reorder=True)


# class AUCPR(torchmetrics.Metric):
#     """
#     Computes the area under the precision recall curve.
#     """

#     values: List[torch.Tensor]
#     targets: List[torch.Tensor]

#     def __init__(self, compute_on_step: bool = True, dist_sync_fn: Any = None):
#         super().__init__(compute_on_step=compute_on_step, dist_sync_fn=dist_sync_fn)

#         self.add_state("values", [], dist_reduce_fx="cat")
#         self.add_state("targets", [], dist_reduce_fx="cat")

#     def update(self, values: torch.Tensor, targets: torch.Tensor) -> None:
#         self.values.append(values)
#         self.targets.append(targets)

#     def compute(self) -> torch.Tensor:
#         precision, recall, _ = M.precision_recall_curve(
#             torch.cat(self.values), torch.cat(self.targets), pos_label=1
#         )
#         return M.auc(cast(torch.Tensor, recall), cast(torch.Tensor, precision), reorder=True)


# def get_ood_auc(score_type: str, corrects: np.array, scores: np.array) -> Tensor:
#     """calculates the area-under-the-curve score (either PR or ROC)
#     Args:
#         score_type (str): desired score type (either APR or AUROC)
#         corrects (np.array): binary array indicating correct predictions
#         scores (np.array): array of prediction scores
#     Raises:
#         AssertionError: raised if score other than APR or AUROC passed
#     Returns:
#         Tensor: area-under-the-curve scores
#     """
#     # avoid INF or NAN values
#     scores = scores.cpu()
#     corrects = corrects.cpu()
#     scores = np.nan_to_num(scores)

#     if score_type == 'AUROC':
#         fpr, tpr, _ = metrics.roc_curve(corrects, scores)
#         return torch.as_tensor(metrics.auc(fpr, tpr))

#     if score_type == 'APR':
#         prec, rec, _ = metrics.precision_recall_curve(corrects, scores)
#         return torch.as_tensor(metrics.auc(rec, prec))

#     raise AssertionError
