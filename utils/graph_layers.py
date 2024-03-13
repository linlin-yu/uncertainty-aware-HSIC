import torch
from torch_geometric.nn import GCNConv, GATv2Conv, APPNP
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch.nn import Linear, Parameter

# class GPNNet(torch.nn.Module):
#     def __init__(self, in_dim, hidden_dim, latent_dim, out_dim, seed, drop_prob=0, bias=True, iteration_step=10, teleport=0.1) -> None:
#         super(GPNNet, self).__init__()
#         torch.manual_seed(seed)

#         self.input_encoder = nn.Sequential(
#             nn.Linear(self.in_dim, self.hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(p=self.dropout_prob))
#         self.latent_encoder = nn.Linear(self.hidden_dim, self.params.latent_dim)


#         self.BN1 = torch.nn.BatchNorm1d(in_dim)
#         self.conv1 = GCNConv(in_dim, hidden_dim, bias=bias, add_self_loops=True)
#         self.BN2 = torch.nn.BatchNorm1d(hidden_dim)
#         self.conv2 = GCNConv(hidden_dim, out_dim, bias=bias, add_self_loops=True)
#         self.prop = APPNP(K=iteration_step, alpha=teleport, dropout=0, cached=False, add_self_loops=False, normalize='sym')


#     def forward(self, data):
#         x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
#         # # print('20', x)
#         # x = self.BN1(x)
#         # # print('22', x)
#         x = self.conv1(x, edge_index, edge_weight)
#         # x = self.BN2(x)
#         x = x.relu()
#         x = F.dropout(x, p=self.drop_prob, training=self.training)
#         x = self.conv2(x, edge_index, edge_weight)
#         x = F.relu(x)
#         x = torch.exp(x)
#         x =  self.prop(x, edge_index)
#         # x =  self.prop(x, edge_index, edge_weight)
#         return x
class GCNNet(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, seed, drop_prob=0, bias=True):
        super(GCNNet, self).__init__()
        torch.manual_seed(seed)
        self.drop_prob = drop_prob
        self.BN1 = torch.nn.BatchNorm1d(in_dim)
        self.conv1 = GCNConv(in_dim, hidden_dim, bias=bias, add_self_loops=True)
        self.BN2 = torch.nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim, bias=bias, add_self_loops=True)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        # x = self.BN1(x)
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        x = F.dropout(x, p=self.drop_prob, training=self.training)
        # x = self.BN2(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x) + 1

        return x


class GCNNetExp(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, seed, drop_prob=0, bias=True):
        super(GCNNetExp, self).__init__()
        torch.manual_seed(seed)
        self.drop_prob = drop_prob
        self.BN1 = torch.nn.BatchNorm1d(in_dim)
        self.conv1 = GCNConv(in_dim, hidden_dim, bias=bias, add_self_loops=True)
        self.BN2 = torch.nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim, bias=bias, add_self_loops=True)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        # x = self.BN1(x)
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        x = F.dropout(x, p=self.drop_prob, training=self.training)
        # x = self.BN2(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = x.clamp(min=-20.0, max=20.0)
        # x = F.relu(x)
        x = torch.exp(x)
        x = x + 1

        return x


class GCNNetReExp(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, seed, drop_prob=0, bias=True):
        super(GCNNetReExp, self).__init__()
        torch.manual_seed(seed)
        self.drop_prob = drop_prob
        self.BN1 = torch.nn.BatchNorm1d(in_dim)
        self.conv1 = GCNConv(in_dim, hidden_dim, bias=bias, add_self_loops=True)
        self.BN2 = torch.nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim, bias=bias, add_self_loops=True)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        # x = self.BN1(x)
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        x = F.dropout(x, p=self.drop_prob, training=self.training)
        # x = self.BN2(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = torch.exp(x)

        return x


class GCNNetExpPropagation(torch.nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        seed,
        drop_prob=0,
        bias=True,
        iteration_step=10,
        teleport=0.1,
    ):
        super(GCNNetExpPropagation, self).__init__()
        torch.manual_seed(seed)
        self.drop_prob = drop_prob
        self.BN1 = torch.nn.BatchNorm1d(in_dim)
        self.conv1 = GCNConv(in_dim, hidden_dim, bias=bias, add_self_loops=True)
        self.BN2 = torch.nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim, bias=bias, add_self_loops=True)
        self.prop = APPNP(
            K=iteration_step,
            alpha=teleport,
            dropout=0,
            cached=False,
            add_self_loops=False,
            normalize="sym",
        )

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        # # print('20', x)
        # x = self.BN1(x)
        # # print('22', x)
        x = self.conv1(x, edge_index, edge_weight)
        # x = self.BN2(x)
        x = x.relu()
        x = F.dropout(x, p=self.drop_prob, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = torch.exp(x)
        x = self.prop(x, edge_index)
        # x =  self.prop(x, edge_index, edge_weight)
        return x


class GCNNetPropagation(torch.nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        seed,
        drop_prob=0,
        bias=True,
        iteration_step=10,
        teleport=0.1,
    ):
        super(GCNNetPropagation, self).__init__()
        torch.manual_seed(seed)
        self.drop_prob = drop_prob
        self.BN1 = torch.nn.BatchNorm1d(in_dim)
        self.conv1 = GCNConv(in_dim, hidden_dim, bias=bias, add_self_loops=True)
        self.BN2 = torch.nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim, bias=bias, add_self_loops=True)
        self.prop = APPNP(
            K=iteration_step,
            alpha=teleport,
            dropout=0,
            cached=False,
            add_self_loops=False,
            normalize="sym",
        )

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        # # print('20', x)
        # x = self.BN1(x)
        # # print('22', x)
        x = self.conv1(x, edge_index, edge_weight)
        # x = self.BN2(x)
        x = x.relu()
        x = F.dropout(x, p=self.drop_prob, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x) + 1
        x = self.prop(x, edge_index)
        # x =  self.prop(x, edge_index, edge_weight)
        return x


class GCNNetClassification(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, seed, drop_prob=0, bias=True):
        super(GCNNetClassification, self).__init__()
        torch.manual_seed(seed)
        self.drop_prob = drop_prob
        self.BN1 = torch.nn.BatchNorm1d(in_dim)
        self.conv1 = GCNConv(in_dim, hidden_dim, bias=bias, add_self_loops=True)
        self.BN2 = torch.nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim, bias=bias, add_self_loops=True)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        # x = self.BN1(x)
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        x = F.dropout(x, p=self.drop_prob, training=self.training)
        # x = self.BN2(x)
        x = self.conv2(x, edge_index, edge_weight)

        return x


class GATNet(torch.nn.Module):
    def __init__(
        self, in_dim, hidden_dim, out_dim, seed, drop_prob=0, bias=True, heads=8
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.drop_prob = drop_prob
        self.BN1 = torch.nn.BatchNorm1d(in_dim)
        self.conv1 = GATv2Conv(
            in_dim, hidden_dim, heads=heads, bias=bias, add_self_loops=True
        )
        self.BN2 = torch.nn.BatchNorm1d(hidden_dim)
        self.conv2 = GATv2Conv(
            hidden_dim, out_dim, heads=heads, bias=bias, add_self_loops=True
        )

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = self.BN1(x)
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        x = F.dropout(x, p=self.drop_prob, training=self.training)
        x = self.BN2(x)
        x = self.conv2(x, edge_index, edge_weight)
        return x


class GCNNetUnmixing(torch.nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_nodes,
        num_features,
        drop_prob=0,
        bias=True,
    ):
        super(GCNNetUnmixing, self).__init__()
        torch.manual_seed(1234567)
        self.drop_prob = drop_prob
        self.conv1 = GCNConv(in_dim, hidden_dim, bias=bias, add_self_loops=True)
        self.conv2 = GCNConv(hidden_dim, out_dim, bias=bias, add_self_loops=True)
        self.abundanceA = Parameter(torch.rand(num_nodes, out_dim), requires_grad=True)
        self.endmemberS_00 = Parameter(torch.rand(1, num_features), requires_grad=True)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        x = F.dropout(x, p=self.drop_prob, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        abundanceA = F.relu(self.abundanceA)
        endmemberS_00 = F.relu(self.endmemberS_00)

        return x, abundanceA, endmemberS_00


class GCNNetTrueA(torch.nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_nodes,
        num_features,
        drop_prob=0,
        bias=True,
    ):
        super(GCNNetTrueA, self).__init__()
        torch.manual_seed(1234567)
        self.drop_prob = drop_prob
        self.conv1 = GCNConv(in_dim, hidden_dim, bias=bias, add_self_loops=True)
        self.conv2 = GCNConv(hidden_dim, out_dim, bias=bias, add_self_loops=True)
        self.endmemberS_00 = Parameter(torch.rand(1, num_features), requires_grad=True)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        x = F.dropout(x, p=self.drop_prob, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        endmemberS_00 = F.relu(self.endmemberS_00)

        return x, endmemberS_00


class GCNNetTrueAS(torch.nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_nodes,
        num_features,
        drop_prob=0,
        bias=True,
    ):
        super(GCNNetTrueAS, self).__init__()
        torch.manual_seed(1234567)
        self.drop_prob = drop_prob
        self.conv1 = GCNConv(in_dim, hidden_dim, bias=bias, add_self_loops=True)
        self.conv2 = GCNConv(hidden_dim, out_dim, bias=bias, add_self_loops=True)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        x = F.dropout(x, p=self.drop_prob, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)

        return x


class GCNNetTrueS(torch.nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_nodes,
        num_features,
        drop_prob=0,
        bias=True,
    ):
        super(GCNNetTrueS, self).__init__()
        torch.manual_seed(1234567)
        self.drop_prob = drop_prob
        self.conv1 = GCNConv(in_dim, hidden_dim, bias=bias, add_self_loops=True)
        self.conv2 = GCNConv(hidden_dim, out_dim, bias=bias, add_self_loops=True)
        self.c = Parameter(torch.ones(1, 1), requires_grad=False)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        x = F.dropout(x, p=self.drop_prob, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        x = x.relu() + 1
        # c = self.c.relu()

        return x, self.c
