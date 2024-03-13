import torch
from torch_geometric.nn import GCNConv, GATv2Conv, APPNP
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch.nn import Linear, Parameter
from .flow_layers_multiple import Density, Evidence
from typing import Dict, Tuple, List
from torch_geometric.data import Data


class GPNNet(torch.nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        latent_dim,
        out_dim,
        radial_layers,
        drop_prob=0,
        iteration_step=10,
        teleport=0.1,
        pretrain_mode="flow",
    ) -> None:
        super(GPNNet, self).__init__()
        self.num_classes = out_dim
        self.pretrain_mode = pretrain_mode
        self.latent_dim = latent_dim

        self.input_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(p=drop_prob)
        )
        self.latent_encoder = nn.Linear(hidden_dim, latent_dim)

        self.flow = Density(
            dim_latent=latent_dim,
            num_mixture_elements=self.num_classes,
            radial_layers=radial_layers,
        )

        self.alpha_evidence_scale = "latent-new-plus-classes"
        self.evidence = Evidence(scale=self.alpha_evidence_scale)

        self.propagation = APPNP(
            K=iteration_step,
            alpha=teleport,
            dropout=0,
            cached=False,
            add_self_loops=False,
            normalize="sym",
        )
        assert self.pretrain_mode in ("encoder", "flow", None)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        h = self.input_encoder(data.x)
        z = self.latent_encoder(h)

        # compute feature evidence (with Normalizing Flows)
        # log p(z, c) = log p(z | c) p(c)
        p_c = self.get_class_probalities(data)
        log_q_ft_per_class = self.flow(z) + p_c.view(1, -1).log()

        if "-plus-classes" in self.alpha_evidence_scale:
            further_scale = self.num_classes
        else:
            further_scale = 1.0

        beta_ft = self.evidence(
            log_q_ft_per_class, dim=self.latent_dim, further_scale=further_scale
        ).exp()

        alpha_features = 1.0 + beta_ft

        beta = self.propagation(beta_ft, edge_index)
        alpha = 1.0 + beta

        return alpha

    def get_class_probalities(self, data: Data) -> torch.Tensor:
        l_c = torch.zeros(self.num_classes, device=data.x.device)
        y_train = data.y[data.train_mask]

        # calculate class_counts L(c)
        for c in range(self.num_classes):
            class_count = (y_train == c).int().sum()
            l_c[c] = class_count

        L = l_c.sum()
        p_c = l_c / L

        return p_c

    def get_optimizer(
        self, lr: float, weight_decay: float
    ) -> Tuple[optim.Adam, optim.Adam]:
        flow_lr = lr
        flow_weight_decay = 0

        flow_params = list(self.flow.named_parameters())
        flow_param_names = [f"flow.{p[0]}" for p in flow_params]
        flow_param_weights = [p[1] for p in flow_params]

        all_params = list(self.named_parameters())
        params = [p[1] for p in all_params if p[0] not in flow_param_names]

        # all params except for flow
        flow_optimizer = optim.Adam(
            flow_param_weights, lr=flow_lr, weight_decay=flow_weight_decay
        )
        model_optimizer = optim.Adam(
            [
                {
                    "params": flow_param_weights,
                    "lr": flow_lr,
                    "weight_decay": flow_weight_decay,
                },
                {"params": params},
            ],
            lr=lr,
            weight_decay=weight_decay,
        )

        return model_optimizer, flow_optimizer

    # def get_warmup_optimizer(self, lr: float, weight_decay: float) -> optim.Adam:
    #     model_optimizer, flow_optimizer = self.get_optimizer(lr, weight_decay)

    #     if self.pretrain_mode == 'encoder':
    #         warmup_optimizer = model_optimizer
    #     else:
    #         warmup_optimizer = flow_optimizer

    #     return warmup_optimizer

    # def get_finetune_optimizer(self, lr: float, weight_decay: float) -> optim.Adam:
    #     # similar to warmup
    #     return self.get_warmup_optimizer(lr, weight_decay)

    # def finetune_loss(self,):
    #     pass

    # def warmup_loss(self,):
    #     pass
