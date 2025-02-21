import torch
import torch.nn as nn
import torch_scatter
from torch_geometric.nn import GATConv, GraphNorm
import torch.nn.functional as F
import numpy as np


class DHMNN(nn.Module):

    def __init__(self, args):
        super(DHMNN, self).__init__()

        dg, dl, h, heads = args.dg, args.dl, args.h, args.heads

        self.ling = nn.Linear(dg, h)
        self.linl = nn.Linear(dl, h)

        self.local_att = GATConv(h, h//heads, heads=heads)
        self.gobal_att = nn.MultiheadAttention(embed_dim=h, num_heads=heads)

        self.normg = nn.LayerNorm(h)
        self.norml = GraphNorm(h)

        self.ling_S = nn.Linear(4 * h, 1)
        self.linl_S = nn.Linear(4 * h, 1)

        self.ling_D = nn.Linear(h, h, bias=False)
        self.linl_D = nn.Linear(h, h, bias=False)

        self.mlp_P = MLP(dg, args.Classifier_hidden, 1, num_layers=args.Classifier_num_layers, dropout=0.2, Normalization='ln')

        self.act = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.cos = nn.CosineSimilarity(dim=1)
        self.q = args.q

        self.reset_parameters()

    def reset_parameters(self):
        self.ling.reset_parameters()
        self.linl.reset_parameters()

        self.local_att.reset_parameters()
        self.gobal_att._reset_parameters()

        self.normg.reset_parameters()
        self.norml.reset_parameters()

        self.ling_S.reset_parameters()
        self.linl_S.reset_parameters()

        self.ling_D.reset_parameters()
        self.linl_D.reset_parameters()

        self.mlp_P.reset_parameters()



    def forward(self, data):

        def score(T_Xve, H_Xve, linS, linD):
            Mean_T_Xe = torch_scatter.scatter(T_Xve, T_edge, dim=0, reduce='mean')
            Mean_H_Xe = torch_scatter.scatter(H_Xve, H_edge, dim=0, reduce='mean')

            Diff_T = torch_scatter.scatter((T_Xve - Mean_T_Xe[T_edge, :]) ** 2, T_edge, dim=0, reduce='mean')
            Diff_H = torch_scatter.scatter((H_Xve - Mean_H_Xe[H_edge, :]) ** 2, H_edge, dim=0, reduce='mean')
            Diff_TH = torch_scatter.scatter((T_Xve - Mean_H_Xe[T_edge, :]) ** 2, T_edge, dim=0, reduce='mean')
            Diff_HT = torch_scatter.scatter((H_Xve - Mean_T_Xe[H_edge, :]) ** 2, H_edge, dim=0, reduce='mean')

            Se = self.sigmoid(linS(torch.cat((Diff_T, Diff_H, Diff_TH, Diff_HT), dim=1)))
            De = (self.cos(linD(Mean_T_Xe), Mean_H_Xe).unsqueeze(1) + 1)/2
            return Se, De


        Xl, Xg, Xe = data.Xl, data.Xg, data.Xe
        C_vertex, C_edge = data.C_vertex, data.C_edge
        T_vertex, H_vertex = data.T_vertex, data.H_vertex
        T_edge, H_edge = data.T_edge, data.H_edge
        e_index = data.e_index

        # structural feature g_v
        Xg = self.act(self.ling(Xg))
        Xg1, _ = self.gobal_att(Xg, Xg, Xg)
        Xg = self.act(self.normg(Xg + Xg1))
        Xg = Xg[C_vertex, :]
        Seg, Deg = score(Xg[T_vertex, :], Xg[H_vertex, :], self.ling_S, self.ling_D)  # similarity and directionality score

        # connectivity feature l_v
        Xl = self.act(self.linl(Xl))
        Xl = self.norml(Xl, C_edge)
        Xl = self.act(Xl + self.local_att(Xl, e_index))
        Sel, Del = score(Xl[T_vertex, :], Xl[H_vertex, :], self.linl_S, self.linl_D)  # similarity and directionality score

        # information fusion
        Se = self.q * Seg + (1-self.q) * Sel  # similarity score
        De = self.q * Deg + (1-self.q) * Del  # directional score

        Pe = self.sigmoid(self.mlp_P(Xe))    # topological score

        return {'Pe': Pe, 'Se': Se, 'De': De}




class MLP(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5, Normalization='bn', InputNorm=False):
        super(MLP, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.lins = nn.ModuleList()
        self.normalizations = nn.ModuleList()
        self.InputNorm = InputNorm

        assert Normalization in ['bn', 'ln', 'None']
        if Normalization == 'bn':
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                if InputNorm:
                    self.normalizations.append(nn.BatchNorm1d(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                if InputNorm:
                    self.normalizations.append(nn.BatchNorm1d(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.BatchNorm1d(hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.BatchNorm1d(hidden_channels))
                self.lins.append(nn.Linear(hidden_channels, out_channels))
        elif Normalization == 'ln':
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                if InputNorm:
                    self.normalizations.append(nn.LayerNorm(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                if InputNorm:
                    self.normalizations.append(nn.LayerNorm(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.LayerNorm(hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.LayerNorm(hidden_channels))
                self.lins.append(nn.Linear(hidden_channels, out_channels))
        else:
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.Identity())
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for normalization in self.normalizations:
            if not (normalization.__class__.__name__ == 'Identity'):
                normalization.reset_parameters()

    def forward(self, x):
        x = self.normalizations[0](x)
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.normalizations[i+1](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x

    def flops(self, x):
        num_samples = np.prod(x.shape[:-1])
        flops = num_samples * self.in_channels # first normalization
        flops += num_samples * self.in_channels * self.hidden_channels # first linear layer
        flops += num_samples * self.hidden_channels # first relu layer

        # flops for each layer
        per_layer = num_samples * self.hidden_channels * self.hidden_channels
        per_layer += num_samples * self.hidden_channels # relu + normalization
        flops += per_layer * (len(self.lins) - 2)

        flops += num_samples * self.out_channels * self.hidden_channels # last linear layer

        return flops
class PlainMLP(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5, Normalization='bn', InputNorm=False):
        super(PlainMLP, self).__init__()
        self.lins = nn.ModuleList()

        self.lins.append(nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x