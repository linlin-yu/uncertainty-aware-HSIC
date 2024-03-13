import os
import torch
import torch.nn.functional as F
import torch.distributions as D
from torch import Tensor
import math
import networkx as nx
from networkx.algorithms.shortest_paths.unweighted import (
    single_source_shortest_path_length,
)
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
import torch_geometric.transforms as T


class ModelLoss(torch.nn.Module):
    def __init__(self, model_cfg, data, mode="ood", dataset="Cora"):
        super(ModelLoss, self).__init__()
        """
        Args:
            reduction (str, optional): reduction method ('sum' | 'mean' | 'none')
        """
        self.emse_loss_weight = model_cfg.emse_loss_weight
        self.uce_loss_weight = model_cfg.uce_loss_weight
        self.uce_log_loss_weight = model_cfg.uce_log_loss_weight
        self.ce_loss_weight = model_cfg.ce_loss_weight
        self.entropy_reg_weight = model_cfg.entropy_reg_weight
        self.reconstruction_reg_weight = model_cfg.reconstruction_reg_weight
        self.tv_alpha_reg_weight = model_cfg.tv_alpha_reg_weight
        self.tv_vacuity_reg_weight = model_cfg.tv_vacuity_reg_weight
        self.probability_teacher_weight = model_cfg.probability_teacher_weight
        self.alpha_teacher_weight = model_cfg.alpha_teacher_weight
        self.reduction = model_cfg.reduction
        """
            endmemberS (torch.Tensor): endmember matrix for id material
            X (torch.Tensor): node features
        """
        data = data
        self.X = data.x
        if hasattr(data, "endmemberS"):
            self.endmemberS = data.endmemberS
            self.h, self.w = data.hw
            self.spatial_labeled_mask = data.spatial_labeled_mask
            self.labeled_indices = data.labeled_indices
        self.mode = mode
        if self.probability_teacher_weight != None:
            # teacher_file = os.path.join('teacher', f'probability_teacher_{dataset}.pt')
            # probability_prior = torch.load(teacher_file).to(data.x.get_device())
            # if mode == 'ood':
            #     self.probability_prior = data.probability_prior
            # else:
            #     self.probability_prior = probability_prior
            self.probability_prior = data.probability_prior
        if self.alpha_teacher_weight != None:
            self.alpha_prior = data.alpha_prior

    def loss_reduce(
        self,
        loss: torch.Tensor,
    ) -> torch.Tensor:
        """utility function to reduce raw losses
        Args:
            loss (torch.Tensor): raw loss which should be reduced

        Returns:
            torch.Tensor: reduced loss
        """

        if self.reduction == "sum":
            return loss.sum()

        if self.reduction == "mean":
            return loss.mean()

        if self.reduction == "none":
            return loss

        raise ValueError(f"{self.reduction} is not a valid value for reduction")

    def uce_loss(
        self,
        alpha: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """utility function computing uncertainty cross entropy /
        bayesian risk of cross entropy
        Args:
            alpha (torch.Tensor): parameters of Dirichlet distribution
            y (torch.Tensor): ground-truth class labels (not one-hot encoded)
        Returns:
            torch.Tensor: loss
        """

        if alpha.dim() == 1:
            alpha = alpha.view(1, -1)

        a_sum = alpha.sum(-1)
        a_true = alpha.gather(-1, y.view(-1, 1)).squeeze(-1)
        uce = a_sum.digamma() - a_true.digamma()
        return self.loss_reduce(uce)

    def uce_log_loss(
        self,
        alpha: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """utility function computing uncertainty cross entropy /
        bayesian risk of cross entropy
        Args:
            alpha (torch.Tensor): parameters of Dirichlet distribution
            y (torch.Tensor): ground-truth class labels (not one-hot encoded)
        Returns:
            torch.Tensor: loss
        """
        if alpha.dim() == 1:
            alpha = alpha.view(1, -1)

        a_sum = alpha.sum(-1)
        a_true = alpha.gather(-1, y.view(-1, 1)).squeeze(-1)
        uce_log = torch.log(torch.log(1 + 1e-5 + a_sum)) - torch.log(
            torch.log(1 + 1e-5 + a_true)
        )
        return self.loss_reduce(uce_log)

    def ce_loss(self, alpha: torch.Tensor, y: torch.Tensor):

        alpha_0 = alpha.sum(dim=-1, keepdim=True)
        y_hat = alpha / alpha_0
        log_soft = torch.log(y_hat)
        return F.nll_loss(log_soft, y, reduction=self.reduction)

    def entropy_reg_loss(
        self,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        """calculates entopy regularizer
        Args:
            alpha (torch.Tensor): dirichlet-alpha scores
            beta_reg (float): regularization factor
            approximate (bool, optional): flag specifying if the entropy is approximated or not. Defaults to False.
        Returns:
            torch.Tensor: REG
        """
        reg = D.Dirichlet(alpha).entropy()

        return -self.loss_reduce(reg)

    def emse_loss(
        self,
        alpha: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """bayesian-risk-loss of sum-of-squares
        Args:
            alpha (torch.Tensor): dirichlet-alpha scores
            y (torch.Tensor): ground-truth labels
        Returns:
            torch.Tensor: loss
        """

        alpha_0 = alpha.sum(dim=-1, keepdim=True)
        y_pred = alpha / alpha_0
        num_classes = alpha.size(-1)
        y_one_hot = F.one_hot(y, num_classes)
        loss_err = (y_one_hot - y_pred) ** 2
        loss_var = y_pred * (1 - y_pred) / (alpha_0 + 1.0)
        loss = (loss_err + loss_var).sum(-1)
        return self.loss_reduce(loss)

    def reconstruction_reg_ood_loss(
        self,
        alpha: torch.tensor,
        endmemberS_00: torch.tensor,
    ) -> torch.Tensor:
        """bayesian-risk-loss of sum-of-squares
        Args:
            alpha (torch.Tensor): dirichlet-alpha scores
            endmemberS_00 (torch.Tensor): endmember marix for ood material
        Returns:
            torch.Tensor: loss
        """

        alpha_0 = alpha.sum(dim=-1, keepdim=True)
        num_classes = alpha.size(-1)
        scale = alpha_0 + num_classes
        scaled_probability = alpha / scale
        scaled_vacuity = num_classes / scale

        # print(self.X.shape, scaled_probability.shape, self.endmemberS.shape, scaled_vacuity.shape, endmemberS_00.shape)
        loss = torch.linalg.norm(
            (
                self.X
                - torch.mm(scaled_probability, self.endmemberS)
                - torch.mm(scaled_vacuity, endmemberS_00)
            ),
            ord=2,
            dim=1,
        )
        return self.loss_reduce(loss)

    def reconstruction_reg_mis_loss(
        self,
        alpha: torch.tensor,
    ) -> torch.Tensor:
        """bayesian-risk-loss of sum-of-squares
        Args:
            alpha (torch.Tensor): dirichlet-alpha scores
        Returns:
            torch.Tensor: loss
        """

        alpha_0 = alpha.sum(dim=-1, keepdim=True)
        num_classes = alpha.size(-1)
        scale = alpha_0 + num_classes
        scaled_probability = alpha / scale

        loss = torch.linalg.norm(
            (self.X - torch.mm(scaled_probability, self.endmemberS)), ord=2, dim=1
        )
        return self.loss_reduce(loss)

    def total_variation_alpha_loss(
        self, alpha: torch.tensor, epison=1e-8
    ) -> torch.Tensor:
        """
        Total variation on ID class probabilities
        Args:
            alpha (torch.Tensor): dirichlet-alpha scores
        Returns:
            torch.Tensor: loss
        """
        # import pdb
        # pdb.set_trace()
        num_classes = alpha.size(-1)
        alpha_pad = torch.zeros(self.h * self.w, num_classes).to(alpha.device)
        alpha_pad[self.labeled_indices, :] = alpha
        alpha_2D = alpha_pad.reshape(self.h, self.w, num_classes)

        diff_h = (torch.diff(alpha_2D[1:-1, :, :], dim=1)).pow(2)
        sqrt_h = (diff_h[:, 1:, :] + diff_h[:, :-1, :] + epison).sqrt()
        sqrt_h = F.pad(
            sqrt_h, pad=(0, 0, 1, 1, 1, 1), mode="constant", value=0
        ).reshape(-1, num_classes)[self.labeled_indices, :]
        tv_h = sqrt_h[self.spatial_labeled_mask].sum()

        diff_w = (torch.diff(alpha_2D[:, 1:-1, :], dim=0)).pow(2)
        sqrt_w = (diff_w[1:, :, :] + diff_w[:-1, :, :] + epison).sqrt()
        sqrt_w = F.pad(
            sqrt_w, pad=(0, 0, 1, 1, 1, 1), mode="constant", value=0
        ).reshape(-1, num_classes)[self.labeled_indices, :]
        tv_w = sqrt_w[self.spatial_labeled_mask].sum()

        loss = tv_h + tv_w

        # diff_h = (torch.diff(alpha_2D, dim=1)).pow(2)[1:-1, 1:,:]
        # pad_h = F.pad(diff_h, pad=(0,0,1,1,1,1), mode='constant', value=0).reshape(-1, )[self.labeled_indices]
        # diff_w = (torch.diff(alpha_2D, dim=0)).pow(2)[1:,1:-1,:]
        # pad_w = F.pad(diff_w, pad=(0,0,1,1,1,1), mode='constant', value=0).reshape(-1,)[self.labeled_indices]
        # tmp = pad_h + pad_w + 1e-8
        # loss = tmp.sqrt()
        return self.loss_reduce(loss)

    def total_variation_vacuity_loss(
        self, alpha: torch.tensor, epison=1e-8
    ) -> torch.Tensor:
        """
        Total variation on vacuity
        Args:
            alpha (torch.Tensor): dirichlet-alpha scores
        Returns:
            torch.Tensor: loss
        """
        # import pdb
        # pdb.set_trace()
        alpha_0 = alpha.sum(dim=-1, keepdim=True)
        num_classes = alpha.size(-1)
        vacuity = num_classes / alpha_0

        vacuity_pad = torch.zeros(self.h * self.w, 1).to(vacuity.device)
        vacuity_pad[self.labeled_indices] = vacuity
        vacuity_2D = vacuity_pad.reshape(self.h, self.w, 1)

        diff_h = (torch.diff(vacuity_2D[1:-1, :], dim=1)).pow(2)
        sqrt_h = (diff_h[:, 1:] + diff_h[:, :-1] + epison).sqrt()
        sqrt_h = F.pad(
            sqrt_h, pad=(0, 0, 1, 1, 1, 1), mode="constant", value=0
        ).reshape(-1,)[self.labeled_indices]
        tv_h = sqrt_h[self.spatial_labeled_mask].sum()

        diff_w = (torch.diff(vacuity_2D[:, 1:-1], dim=0)).pow(2)
        sqrt_w = (diff_w[1:, :] + diff_w[:-1, :] + epison).sqrt()
        sqrt_w = F.pad(
            sqrt_w, pad=(0, 0, 1, 1, 1, 1), mode="constant", value=0
        ).reshape(-1,)[self.labeled_indices]
        tv_w = sqrt_w[self.spatial_labeled_mask].sum()

        loss = tv_h + tv_w

        # diff_h = (torch.diff(vacuity_2D, dim=1)).pow(2)[1:-1, 1:,:]
        # pad_h = F.pad(diff_h, pad=(0,0,1,1,1,1), mode='constant', value=0).reshape(-1, )[self.labeled_indices]
        # diff_w = (torch.diff(vacuity_2D, dim=0)).pow(2)[1:,1:-1,:]
        # pad_w = F.pad(diff_w, pad=(0,0,1,1,1,1), mode='constant', value=0).reshape(-1,)[self.labeled_indices]
        # tmp = pad_h + pad_w + 1e-8
        # loss = tmp.sqrt()
        # import pdb
        # pdb.set_trace()
        return self.loss_reduce(loss)

    def probability_teacher(self, alpha: torch.Tensor) -> torch.Tensor:

        alpha_0 = alpha.sum(dim=-1, keepdim=True)
        y_pred = alpha / alpha_0
        loss = y_pred * (torch.log(torch.div(y_pred, self.probability_prior)))
        return self.loss_reduce(loss)

    def alpha_teacher(self, alpha: torch.Tensor) -> torch.Tensor:
        dirichlet = D.Dirichlet(alpha)
        alpha_prior = self.alpha_prior.to(alpha.device)
        dirichlet_prior = D.Dirichlet(alpha_prior)
        KL_prior = D.kl.kl_divergence(dirichlet, dirichlet_prior)
        return self.loss_reduce(KL_prior)

    def forward(
        self,
        alpha: torch.tensor,
        y: torch.Tensor,
        endmemberS_00: torch.tensor = None,
        mask: torch.tensor = None,
    ) -> torch.Tensor:

        loss = dict()
        # print('199', alpha)
        # print(torch.var(alpha))
        if self.emse_loss_weight != None:
            loss["emse_loss"] = self.emse_loss_weight * self.emse_loss(
                alpha[mask], y[mask]
            )
        if self.uce_loss_weight != None:
            loss["uce_loss"] = self.uce_loss_weight * self.uce_loss(
                alpha[mask], y[mask]
            )
        if self.uce_log_loss_weight != None:
            loss["uce_log_loss"] = self.uce_log_loss_weight * self.uce_log_loss(
                alpha[mask], y[mask]
            )
        if self.ce_loss_weight != None:
            loss["ce_loss"] = self.ce_loss_weight * self.ce_loss(alpha[mask], y[mask])
        if self.tv_alpha_reg_weight != None:
            loss["tv_alpha_reg"] = (
                self.tv_alpha_reg_weight * self.total_variation_alpha_loss(alpha)
            )
            # print('208', loss['tv_alpha_reg'])
        if self.entropy_reg_weight != None:
            loss["entropy_reg"] = self.entropy_reg_weight * self.entropy_reg_loss(
                alpha[mask]
            )

        if self.reconstruction_reg_weight != None:
            if self.mode == "ood":
                loss["reconstruction_reg"] = (
                    self.reconstruction_reg_weight
                    * self.reconstruction_reg_ood_loss(alpha, endmemberS_00)
                )
            elif self.mode == "mis":
                loss["reconstruction_reg"] = (
                    self.reconstruction_reg_weight
                    * self.reconstruction_reg_mis_loss(alpha)
                )

        if self.tv_vacuity_reg_weight != None:
            loss["tv_vacuity_reg"] = (
                self.tv_vacuity_reg_weight * self.total_variation_vacuity_loss(alpha)
            )

        if self.probability_teacher_weight != None:
            loss["probability_teacher_reg"] = (
                self.probability_teacher_weight * self.probability_teacher(alpha)
            )

        if self.alpha_teacher_weight != None:
            loss["alpha_teacher_reg"] = self.alpha_teacher_weight * self.alpha_teacher(
                alpha
            )

        return loss


def cross_entropy(
    alpha: torch.Tensor, y: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """wrapper for cross-entropy loss
    Args:
        alpha (torch.Tensor): dirichlet-alpha scores
        y (torch.Tensor): ground-truth labels
        reduction (str, optional): loss reduction. Defaults to 'mean'.
    Returns:
        torch.Tensor: CE
    """
    alpha_0 = alpha.sum(dim=-1, keepdim=True)
    y_hat = alpha / alpha_0
    log_soft = torch.log(y_hat)
    return F.nll_loss(log_soft, y, reduction=reduction).cpu().detach()


class UCELoss(torch.nn.Module):
    def __init__(self, reduction: str = "mean"):
        super(UCELoss, self).__init__()
        self.reduction = reduction

    def loss_reduce(
        self,
        loss: torch.Tensor,
    ) -> torch.Tensor:
        """utility function to reduce raw losses
        Args:
            loss (torch.Tensor): raw loss which should be reduced

        Returns:
            torch.Tensor: reduced loss
        """

        if self.reduction == "sum":
            return loss.sum()

        if self.reduction == "mean":
            return loss.mean()

        if self.reduction == "none":
            return loss

        raise ValueError(f"{self.reduction} is not a valid value for reduction")

    def forward(self, alpha: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """utility function computing uncertainty cross entropy /
        bayesian risk of cross entropy
        Args:
            alpha (torch.Tensor): parameters of Dirichlet distribution
            y (torch.Tensor): ground-truth class labels (not one-hot encoded)
        Returns:
            torch.Tensor: loss
        """

        if alpha.dim() == 1:
            alpha = alpha.view(1, -1)

        a_sum = alpha.sum(-1)
        a_true = alpha.gather(-1, y.view(-1, 1)).squeeze(-1)
        uce = a_sum.digamma() - a_true.digamma()
        return self.loss_reduce(uce)


class ProjectedCELoss(torch.nn.Module):
    def __init__(self, reduction: str = "mean"):
        super(ProjectedCELoss, self).__init__()
        self.reduction = reduction

    def forward(self, alpha: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """wrapper for cross-entropy loss
        Args:
            alpha (torch.Tensor): dirichlet-alpha scores
            y (torch.Tensor): ground-truth labels
            reduction (str, optional): loss reduction. Defaults to 'mean'.
        Returns:
            torch.Tensor: CE
        """
        alpha_0 = alpha.sum(dim=-1, keepdim=True)
        y_hat = alpha / alpha_0
        log_soft = torch.log(y_hat)
        return F.nll_loss(log_soft, y, reduction=self.reduction).cpu().detach()
