from torch.nn import ModuleList
import torch.nn.functional as F
import torch.nn as nn
import torch
from easydict import EasyDict
import numpy as np
from torch.nn import TransformerEncoderLayer


def swish(x):
    return x * torch.sigmoid(x)


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        torch.nn.init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = torch.autograd.Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


class EBM(nn.Module):
    def __init__(self, inp_dim, out_dim, mem):
        super(EBM, self).__init__()
        h = 512

        if mem:
            self.fc1 = nn.Linear(inp_dim + out_dim + h, h)
        else:
            self.fc1 = nn.Linear(inp_dim + out_dim, h)

        self.mem = mem
        self.fc2 = nn.Linear(h, h)
        self.fc3 = nn.Linear(h, h)
        self.fc4 = nn.Linear(h, 1)

        self.inp_dim = inp_dim
        self.out_dim = out_dim

    def forward(self, x):
        h = swish(self.fc1(x))
        h = swish(self.fc2(h))
        h = swish(self.fc3(h))

        output = self.fc4(h)

        return output


class EBMTwin(nn.Module):
    def __init__(self, inp_dim, out_dim, mem):
        super(EBMTwin, self).__init__()
        h = 512
        self.inp_dim = inp_dim
        self.out_dim = out_dim

        self.first_fc1 = nn.Linear(inp_dim, h)
        self.second_fc1 = nn.Linear(out_dim, h)

        self.first_fc2 = nn.Linear(h, h)
        self.second_fc2 = nn.Linear(h, h)

        self.first_fc3 = nn.Linear(h, h)
        self.second_fc3 = nn.Linear(h, h)

        self.prod_h1 = nn.Linear(2 * h, h)
        self.prod_h2 = nn.Linear(2 * h, h)

        self.scale_parameter = torch.nn.Parameter(torch.ones(1))

    def forward(self, x):
        x_out = x[..., :self.out_dim]
        x = x[..., self.out_dim:]

        first_h1 = self.first_fc1(x)
        second_h1 = self.second_fc1(x_out)

        first_h2 = self.first_fc2(swish(first_h1))
        second_h2 = self.second_fc2(swish(second_h1))

        first_h3 = self.first_fc3(swish(first_h2))
        second_h3 = self.second_fc3(swish(second_h2))

        h = torch.cat([first_h2, second_h2], dim=-1)

        energy_3 = torch.abs(first_h3 - second_h3).mean(dim=-1)

        energy = energy_3

        return energy


class FC(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(FC, self).__init__()
        h = 512

        self.fc1 = nn.Linear(inp_dim, h)

        self.fc2 = nn.Linear(h, h)
        self.fc3 = nn.Linear(h, h)
        self.fc4 = nn.Linear(h, out_dim)

        self.inp_dim = inp_dim
        self.out_dim = out_dim

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        output = self.fc4(h)

        return output


class RecurrentFC(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(RecurrentFC, self).__init__()
        h = 196

        self.inp_map = nn.Linear(inp_dim, h)
        self.inp_map2 = nn.Linear(inp_dim, h)
        self.fc1 = nn.Linear(inp_dim, h)
        self.lstm = nn.LSTM(h, h)
        self.fc3 = nn.Linear(h, h)
        self.fc4 = nn.Linear(h, out_dim)
        self.scale_parameter = torch.nn.Parameter(torch.ones(1))

        self.inp_dim = inp_dim
        self.out_dim = out_dim

    def forward(self, x, state=None):

        h = F.relu(self.fc1(x))
        h, state = self.lstm(h[None], state)
        output = self.fc4(h[0])

        return output, state


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['random_feature']:
            freq_bands = 2.0**torch.linspace(0., max_freq, steps=N_freqs)
        elif self.kwargs['log_sampling']:
            freq_bands = 2.0**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(
                    lambda x,
                    p_fn=p_fn,
                    freq=freq: p_fn(
                        x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class IterativeFC(nn.Module):
    def __init__(self, inp_dim, out_dim, *args):
        super(IterativeFC, self).__init__()
        h = 512

        input_dims = 1
        multires = 10

        self.fc1 = nn.Linear(inp_dim + out_dim, h)
        self.fc2 = nn.Linear(h, h)
        self.fc3 = nn.Linear(h, h)
        self.fc4 = nn.Linear(h, out_dim)
        self.scale_parameter = torch.nn.Parameter(torch.ones(1))
        self.fc_map = nn.Linear(out_dim, inp_dim)

        self.out_dim = out_dim
        self.inp_dim = inp_dim

    def forward(self, x):
        x_out = x[..., :self.out_dim]
        x = x[..., self.out_dim:]

        h = self.fc_map(x_out)

        prod = x * h
        x = torch.cat([x, x_out], dim=-1)

        h = h_first = swish(self.fc1(x))
        h = swish(self.fc2(h))
        h = swish(self.fc3(h))
        output = self.fc4(h)

        return output


class PonderFC(nn.Module):
    def __init__(self, inp_dim, out_dim, num_step):
        super(PonderFC, self).__init__()
        h = 512

        input_dims = 1
        multires = 10

        self.fc1 = nn.Linear(inp_dim + out_dim, h)
        self.fc2 = nn.Linear(h, h)
        self.fc3 = nn.Linear(h, h)
        self.fc4 = nn.Linear(h, h)
        self.decode_logit = nn.Linear(h, 1)

        self.fc4_decode = nn.Linear(h, out_dim)
        self.scale_parameter = torch.nn.Parameter(torch.ones(1))
        self.fc_map = nn.Linear(out_dim, inp_dim)

        self.out_dim = out_dim
        self.inp_dim = inp_dim
        self.ponder_prob = 1 - 1 / num_step

    def forward(self, x, eval=False, iters=1):
        logits = []
        outputs = []

        x_out = x[..., :self.out_dim]
        x = x[..., self.out_dim:]

        joint = torch.cat([x_out, x], dim=-1)

        for i in range(iters):
            h = F.relu(self.fc1(joint))
            h = F.relu(self.fc2(h))
            logit = self.decode_logit(h)
            output = self.fc4_decode(h)

            joint = torch.cat([output, x], dim=-1)

            logits.append(logit)
            outputs.append(output)

        logits = torch.stack(logits, dim=1)

        return outputs, logits


class IterativeFCAttention(nn.Module):
    def __init__(self, inp_dim, out_dim, rank):
        super(IterativeFCAttention, self).__init__()
        h = 512

        input_dims = 1
        multires = 10
        self.rank = rank

        self.inp_fc1 = nn.Linear(inp_dim, h)
        self.inp_fc2 = nn.Linear(h, h)

        self.out_fc1 = nn.Linear(out_dim, h)
        self.out_fc2 = nn.Linear(h, h)

        self.fc1 = nn.Linear(3 * h, h)
        self.fc2 = nn.Linear(h, h)
        self.fc3 = nn.Linear(h, out_dim)

        self.out_dim = out_dim
        self.inp_dim = inp_dim
        self.scale_parameter = torch.nn.Parameter(torch.ones(1))

    def forward(self, x):
        # Expects a input x which consists of input computation as well as
        # partially optimized result
        x_out = x[..., :self.out_dim]
        x = x[..., self.out_dim:]

        inp_h = F.relu(self.inp_fc2(F.relu(self.inp_fc1(x))))
        out_h = F.relu(self.out_fc2(F.relu(self.out_fc1(x_out))))

        prod_h = out_h * inp_h
        h = torch.cat([inp_h, out_h, prod_h], dim=-1)

        out = self.fc3(F.relu(self.fc2(F.relu(self.fc1(h)))))

        return out


class IterativeTransformer(nn.Module):
    def __init__(self, inp_dim, out_dim, rank):
        super(IterativeTransformer, self).__init__()
        h = 512
        input_dims = 1
        multires = 10
        self.rank = rank

        embed_kwargs = {
            'include_input': True,
            'random_feature': True,
            'input_dims': input_dims,
            'max_freq_log2': multires - 1,
            'num_freqs': multires,
            'log_sampling': True,
            'periodic_fns': [torch.sin, torch.cos],
        }

        self.inp_embed = Embedder(**embed_kwargs)
        self.out_embed = Embedder(**embed_kwargs)
        self.inp_linear = nn.Linear(1, 63)
        self.out_linear = nn.Linear(1, 63)

        coord = np.mgrid[:inp_dim] / inp_dim
        coord = torch.Tensor(coord)[None, :, None].cuda()

        self.coord = coord

        out_coord = np.mgrid[:out_dim] / out_dim
        out_coord = torch.Tensor(out_coord)[None, :, None].cuda()

        self.out_coord = out_coord

        self.encode1 = TransformerEncoderLayer(84, 4, dim_feedforward=256)
        self.encode2 = TransformerEncoderLayer(84, 4, dim_feedforward=256)

        self.fc1 = nn.Linear(84, 128)
        self.fc2 = nn.Linear(128, 1)

        self.out_dim = out_dim
        self.inp_dim = inp_dim

    def forward(self, x):
        x_out = x[..., :self.out_dim].view(-1, self.out_dim, 1)
        x = x[..., self.out_dim:].view(-1, self.inp_dim, 1)

        x_linear = self.inp_linear(x)
        x_out_linear = self.out_linear(x_out)
        x_pos_embed = self.inp_embed.embed(self.coord)
        x_out_pos_embed = self.out_embed.embed(self.out_coord)

        x_pos_embed = x_pos_embed.expand(x_linear.size(0), -1, -1)
        x_out_pos_embed = x_out_pos_embed.expand(x_linear.size(0), -1, -1)
        pos_embed = torch.cat([x_linear, x_pos_embed], dim=-1)
        out_embed = torch.cat([x_out_linear, x_out_pos_embed], dim=-1)

        s = pos_embed.size()
        pos_embed = pos_embed.view(s[0], -1, s[-1])
        out_embed = out_embed.view(s[0], -1, s[-1])

        embed = torch.cat([pos_embed, out_embed], dim=1)
        embed = embed.permute(1, 0, 2).contiguous()

        embed = self.encode1(embed)
        embed = self.encode2(embed)

        embed = self.fc2(F.relu(self.fc1(embed.permute(1, 0, 2))))[:, :, 0]
        output = embed[:, :self.out_dim]

        return output


class IterativeAttention(nn.Module):
    def __init__(self, inp_dim, out_dim, rank):
        super(IterativeAttention, self).__init__()
        h = 64

        input_dims = 1
        multires = 10
        self.rank = rank
        self.out_dim = out_dim

        embed_kwargs = {
            'include_input': True,
            'random_feature': True,
            'input_dims': input_dims,
            'max_freq_log2': multires - 1,
            'num_freqs': multires,
            'log_sampling': True,
            'periodic_fns': [torch.sin, torch.cos],
        }

        self.inp_embed = Embedder(**embed_kwargs)
        self.out_embed = Embedder(**embed_kwargs)
        self.inp_linear = nn.Linear(1, 42)
        self.out_linear = nn.Linear(1, 42)

        coord = np.mgrid[:rank, :rank] / rank
        coord = torch.Tensor(coord).permute(1, 2, 0)[None, :, :, :].cuda()

        self.coord = coord

        out_coord = np.mgrid[:out_dim] / rank
        out_coord = torch.Tensor(out_coord)[None, :, None].cuda()

        self.out_coord = out_coord

        self.inp_fc1 = nn.Linear(84, h)
        self.inp_fc2 = nn.Linear(h, h)
        self.inp_fc3 = nn.Linear(h, h)

        self.out_fc1 = nn.Linear(63, h)
        self.out_fc2 = nn.Linear(h, h)
        self.out_fc3 = nn.Linear(h, h)

        self.at_fc1 = nn.Linear(63, h)
        self.at_fc2 = nn.Linear(h, h)

        self.fc1 = nn.Linear(3 * h, h)
        self.fc2 = nn.Linear(h, h)
        self.fc3 = nn.Linear(h, 1)
        self.scale_parameter = torch.nn.Parameter(torch.ones(1))

        self.out_dim = out_dim
        self.inp_dim = inp_dim

    def forward(self, x, idx):
        # Expects a input x which consists of input computation as well as
        # partially optimized result
        output = x[..., :self.out_dim]
        x_out = x[..., :self.out_dim].view(-1, self.out_dim, 1)
        x = x[..., self.out_dim:].view(-1, self.rank, self.rank, 1)

        x_linear = self.inp_linear(x)
        x_out_linear = self.out_linear(x_out)
        x_pos_embed = self.inp_embed.embed(self.coord)
        x_out_pos_embed = self.out_embed.embed(self.out_coord)

        x_pos_embed = x_pos_embed.expand(x_linear.size(0), -1, -1, -1)
        x_out_pos_embed = x_out_pos_embed.expand(x_linear.size(0), -1, -1)

        pos_embed = torch.cat([x_linear, x_pos_embed], dim=-1)
        out_embed = torch.cat([x_out_linear, x_out_pos_embed], dim=-1)

        s = pos_embed.size()
        pos_embed = pos_embed.view(s[0], -1, s[-1])
        s = out_embed.size()
        out_embed = out_embed.view(s[0], -1, s[-1])

        idx = idx[:, :, None].expand(-1, -1, out_embed.size(-1))
        at_embed = torch.gather(pos_embed, 1, idx)

        at_embed = self.at_fc2(F.relu(self.at_fc1(at_embed)))
        out_embed = self.out_fc2(F.relu(self.out_fc1(out_embed)))
        pos_embed = self.inp_fc2(F.relu(self.inp_fc1(pos_embed)))

        out_val = self.out_fc3(F.relu(out_embed))
        pos_val = self.inp_fc3(F.relu(pos_embed))

        at_out_wt = torch.einsum('bij,bkj->bik', at_embed, out_embed)
        at_out_wt = torch.softmax(at_out_wt, dim=-1)
        out_embed = (at_out_wt[:, :, :, None] *
                     out_val[:, None, :, :]).sum(dim=2)

        at_pos_wt = torch.einsum('bij,bkj->bik', at_embed, pos_embed)
        at_pos_wt = torch.softmax(at_pos_wt, dim=-1)
        pos_embed = (at_pos_wt[:, :, :, None] *
                     pos_val[:, None, :, :]).sum(dim=2)

        embed = torch.cat([at_embed, pos_embed, out_embed], dim=-1)
        out = self.fc3(F.relu(self.fc2(F.relu(self.fc1(embed)))))[:, :, 0]

        output.requires_grad_()
        idx = idx[:, :, 0]
        output = torch.scatter(output, 1, idx, out)

        return output


class AttentionEBM(nn.Module):
    def __init__(self, inp_dim, out_dim, rank):
        super(AttentionEBM, self).__init__()
        h = 64

        input_dims = 1
        multires = 10
        self.rank = rank
        self.out_dim = out_dim

        embed_kwargs = {
            'include_input': True,
            'random_feature': True,
            'input_dims': input_dims,
            'max_freq_log2': multires - 1,
            'num_freqs': multires,
            'log_sampling': True,
            'periodic_fns': [torch.sin, torch.cos],
        }

        self.inp_embed = Embedder(**embed_kwargs)
        self.out_embed = Embedder(**embed_kwargs)
        self.inp_linear = nn.Linear(1, 42)
        self.out_linear = nn.Linear(1, 42)

        coord = np.mgrid[:rank, :rank] / rank
        coord = torch.Tensor(coord).permute(1, 2, 0)[None, :, :, :].cuda()

        self.coord = coord

        out_coord = np.mgrid[:out_dim] / rank
        out_coord = torch.Tensor(out_coord)[None, :, None].cuda()

        self.out_coord = out_coord

        self.inp_fc1 = nn.Linear(84, h)
        self.inp_fc2 = nn.Linear(h, h)
        self.inp_fc3 = nn.Linear(h, h)

        self.out_fc1 = nn.Linear(63, h)
        self.out_fc2 = nn.Linear(h, h)
        self.out_fc3 = nn.Linear(h, h)

        self.at_fc1 = nn.Linear(63, h)
        self.at_fc2 = nn.Linear(h, h)

        self.fc1 = nn.Linear(3 * h, h)
        self.fc2 = nn.Linear(h, h)
        self.fc3 = nn.Linear(h, 1)
        self.scale_parameter = torch.nn.Parameter(torch.ones(1))

        self.out_dim = out_dim
        self.inp_dim = inp_dim

    def forward(self, x, idx):
        # Expects a input x which consists of input computation as well as
        # partially optimized result

        act_fn = swish

        output = x[..., :self.out_dim]
        x_out = x[..., :self.out_dim].view(-1, self.out_dim, 1)
        x = x[..., self.out_dim:].view(-1, self.rank, self.rank, 1)

        x_linear = self.inp_linear(x)
        x_out_linear = self.out_linear(x_out)
        x_pos_embed = self.inp_embed.embed(self.coord)
        x_out_pos_embed = self.out_embed.embed(self.out_coord)

        x_pos_embed = x_pos_embed.expand(x_linear.size(0), -1, -1, -1)
        x_out_pos_embed = x_out_pos_embed.expand(x_linear.size(0), -1, -1)

        pos_embed = torch.cat([x_linear, x_pos_embed], dim=-1)
        out_embed = torch.cat([x_out_linear, x_out_pos_embed], dim=-1)

        s = pos_embed.size()
        pos_embed = pos_embed.view(s[0], -1, s[-1])
        s = out_embed.size()
        out_embed = out_embed.view(s[0], -1, s[-1])

        idx = idx[:, :, None].expand(-1, -1, out_embed.size(-1))
        at_embed = torch.gather(pos_embed, 1, idx)

        at_embed = self.at_fc2(act_fn(self.at_fc1(at_embed)))
        out_embed = self.out_fc2(act_fn(self.out_fc1(out_embed)))
        pos_embed = self.inp_fc2(act_fn(self.inp_fc1(pos_embed)))

        out_val = self.out_fc3(act_fn(out_embed))
        pos_val = self.inp_fc3(act_fn(pos_embed))

        at_out_wt = torch.einsum('bij,bkj->bik', at_embed, out_embed)
        at_out_wt = torch.softmax(at_out_wt, dim=-1)
        out_embed = (at_out_wt[:, :, :, None] *
                     out_val[:, None, :, :]).sum(dim=2)

        at_pos_wt = torch.einsum('bij,bkj->bik', at_embed, pos_embed)
        at_pos_wt = torch.softmax(at_pos_wt, dim=-1)
        pos_embed = (at_pos_wt[:, :, :, None] *
                     pos_val[:, None, :, :]).sum(dim=2)

        embed = torch.cat([at_embed, pos_embed, out_embed], dim=-1)
        out = self.fc3(act_fn(self.fc2(act_fn(self.fc1(embed)))))[:, :, 0]

        return out


if __name__ == "__main__":
    # Unit test the model
    args = EasyDict()
    args.filter_dim = 64
    args.latent_dim = 64
    args.im_size = 256

    model = LatentEBM(args).cuda()
    x = torch.zeros(1, 3, 256, 256).cuda()
    latent = torch.zeros(1, 64).cuda()
    model(x, latent)
