import torch
from torch_geometric.nn import GINEConv, global_max_pool
from torch import nn
import torch.nn.functional as F


class GraphEBM(nn.Module):
    def __init__(self, inp_dim, out_dim, mem):
        super(GraphEBM, self).__init__()
        h = 128

        self.edge_map = nn.Linear(1, h // 2)
        self.edge_map_opt = nn.Linear(1, h // 2)

        self.conv1 = GINEConv(nn.Sequential(nn.Linear(1, 128)), edge_dim=h)
        self.conv2 = GINEConv(nn.Sequential(nn.Linear(128, 128)), edge_dim=h)
        self.conv3 = GINEConv(nn.Sequential(nn.Linear(128, 128)), edge_dim=h)

        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, inp, opt_edge):

        edge_embed = self.edge_map(inp.edge_attr)
        opt_edge_embed = self.edge_map_opt(opt_edge)

        edge_embed = torch.cat([edge_embed, opt_edge_embed], dim=-1)

        h = self.conv1(inp.x, inp.edge_index, edge_attr=edge_embed)
        h = self.conv2(h, inp.edge_index, edge_attr=edge_embed)
        h = self.conv3(h, inp.edge_index, edge_attr=edge_embed)

        mean_feat = global_max_pool(h, inp.batch)
        energy = self.fc2(F.relu(self.fc1(mean_feat)))

        return energy


class GraphIterative(nn.Module):
    def __init__(self, inp_dim, out_dim, mem):
        super(GraphIterative, self).__init__()
        h = 128

        self.edge_map = nn.Linear(1, h // 2)
        self.edge_map_opt = nn.Linear(1, h // 2)

        self.conv1 = GINEConv(nn.Sequential(nn.Linear(1, 128)), edge_dim=h)
        self.conv2 = GINEConv(nn.Sequential(nn.Linear(128, 128)), edge_dim=h)
        self.conv3 = GINEConv(nn.Sequential(nn.Linear(128, 128)), edge_dim=h)

        self.decode = nn.Linear(256, 1)

    def forward(self, inp, opt_edge):

        edge_embed = self.edge_map(inp.edge_attr)
        opt_edge_embed = self.edge_map_opt(opt_edge)

        edge_embed = torch.cat([edge_embed, opt_edge_embed], dim=-1)

        h = self.conv1(inp.x, inp.edge_index, edge_attr=edge_embed)
        h = self.conv2(h, inp.edge_index, edge_attr=edge_embed)
        h = self.conv3(h, inp.edge_index, edge_attr=edge_embed)

        edge_index = inp.edge_index
        hidden = h

        h1 = hidden[edge_index[0]]
        h2 = hidden[edge_index[1]]

        h = torch.cat([h1, h2], dim=-1)
        output = self.decode(h)

        return output


class GraphRecurrent(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(GraphRecurrent, self).__init__()
        h = 128

        self.edge_map = nn.Linear(1, h)

        self.conv1 = GINEConv(nn.Sequential(nn.Linear(1, 128)), edge_dim=h)

        self.lstm = nn.LSTM(h, h)
        # self.conv2 = GINEConv(nn.Sequential(nn.Linear(128, 128)), edge_dim=h)
        self.conv3 = GINEConv(nn.Sequential(nn.Linear(128, 128)), edge_dim=h)

        self.decode = nn.Linear(256, 1)

    def forward(self, inp, state=None):

        edge_embed = self.edge_map(inp.edge_attr)

        h = self.conv1(inp.x, inp.edge_index, edge_attr=edge_embed)

        h, state = self.lstm(h[None], state)
        h = self.conv3(h[0], inp.edge_index, edge_attr=edge_embed)

        edge_index = inp.edge_index
        hidden = h

        h1 = hidden[edge_index[0]]
        h2 = hidden[edge_index[1]]

        h = torch.cat([h1, h2], dim=-1)
        output = self.decode(h)

        return output, state


class GraphPonder(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(GraphPonder, self).__init__()
        h = 128

        self.edge_map = nn.Linear(1, h)

        self.conv1 = GINEConv(nn.Sequential(nn.Linear(1, 128)), edge_dim=h)
        self.conv2 = GINEConv(nn.Sequential(nn.Linear(128, 128)), edge_dim=h)
        self.conv3 = GINEConv(nn.Sequential(nn.Linear(128, 128)), edge_dim=h)

        self.decode = nn.Linear(256, 1)

    def forward(self, inp, iters=1):

        edge_embed = self.edge_map(inp.edge_attr)

        h = self.conv1(inp.x, inp.edge_index, edge_attr=edge_embed)
        outputs = []

        for i in range(iters):
            h = F.relu(self.conv2(h, inp.edge_index, edge_attr=edge_embed))

            output = self.conv3(h, inp.edge_index, edge_attr=edge_embed)

            edge_index = inp.edge_index
            hidden = output

            h1 = hidden[edge_index[0]]
            h2 = hidden[edge_index[1]]

            output = torch.cat([h1, h2], dim=-1)
            output = self.decode(output)

            outputs.append(output)

        return outputs



class GraphFC(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(GraphFC, self).__init__()
        h = 128

        self.edge_map = nn.Linear(1, h)
        self.conv1 = GINEConv(nn.Sequential(nn.Linear(1, 128)), edge_dim=h)
        self.conv2 = GINEConv(nn.Sequential(nn.Linear(128, 128)), edge_dim=h)
        self.conv3 = GINEConv(nn.Sequential(nn.Linear(128, 128)), edge_dim=h)

        self.decode = nn.Linear(256, 1)


    def forward(self, inp):

        edge_embed = self.edge_map(inp.edge_attr)

        h = self.conv1(inp.x, inp.edge_index, edge_attr=edge_embed)
        h = self.conv2(h, inp.edge_index, edge_attr=edge_embed)
        h = self.conv3(h, inp.edge_index, edge_attr=edge_embed)

        edge_index = inp.edge_index
        hidden = h

        h1 = hidden[edge_index[0]]
        h2 = hidden[edge_index[1]]

        h = torch.cat([h1, h2], dim=-1)
        output = self.decode(h)

        return output
