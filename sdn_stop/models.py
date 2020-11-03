#!/usr/bin/env python
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import aux_funcs as af
import model_funcs as mf

from architectures.CNNs.ResNet import ResNet

import pdb


class PolicyNet(nn.Module):
    """docstring for PolicyNet"""
    def __init__(self, data_type, num_classes=200):
        super(PolicyNet, self).__init__()
        params = {}
        params['num_blocks'] = [5,5,5]
        params['num_classes'] = num_classes
        params['augment_training'] = False
        if data_type == 'tiny_imagenet':
            params['input_size'] = 64
        else:
            params['input_size'] = 32
        params['block_type'] = 'basic'

        self.resnet = ResNet(params)

        self.end_layer = nn.Sequential(
            nn.Linear(num_classes, num_classes),
            nn.ReLU(),
            nn.Linear(num_classes, 1),
            nn.Sigmoid()
            )

    def forward(self, x0, yi):
        # x0_embed = self.resnet(x0)
        # merged = torch.cat((x0_embed, yi), 1)
        return self.end_layer(yi)

class SeqNet(nn.Module):
    def __init__(self, A, args, train_post, device):
        super(SeqNet, self).__init__()
        self.A = A
        self.m, self.n = A.shape
        self.args = args
        hidden_dims = args.policy_hidden_dims + '-' + str(1) # output size is one (score)
        self.mlp = MLP(input_dim=2*self.m + self.n + args.post_dim,
                       hidden_dims=hidden_dims,
                       nonlinearity=args.nonlinearity,
                       act_last=None)
        self.post_dim = args.post_dim
        self.train_post = train_post
        self.device = device

    def forward(self, y, xhs):
        pi_t_score = []
        position_enc = position_encoding(len(self.train_post), self.post_dim).to(self.device)
        batch_size = y.shape[0]
        for i, t in self.train_post.items():
            x_hat = xhs[t]
            ax = torch.matmul(x_hat, self.A.t())
            pe = torch.stack([position_enc[i] for _ in range(batch_size)], dim=0)
            f = torch.cat([y, y-ax, x_hat, pe], dim=-1)
            pi_t_score.append(self.mlp(f))

        scores = torch.cat(pi_t_score, dim=-1)
        # p = torch.sigmoid(self.mlp(f).view(-1))
        return scores


class MulticlassNet(nn.Module):
    def __init__(self, args, A, nz_post):
        super(MulticlassNet, self).__init__()
        self.A = A
        self.m, self.n = A.shape
        self.args = args

        self.nz_post = nz_post
        self.num_class = len(nz_post)
        multiclass_dims = args.policy_multiclass_dims + '-' + str(self.num_class)
        hidden_dims = tuple(map(int, args.policy_hidden_dims.split("-")))
        input_dim = hidden_dims[-1] * self.num_class

        self.mlp = MLP(input_dim=2*self.m + self.n,
                       hidden_dims=args.policy_hidden_dims,
                       nonlinearity=args.nonlinearity,
                       act_last=args.nonlinearity)

        self.mlp2 = MLP(input_dim=input_dim,
                       hidden_dims=multiclass_dims,
                       nonlinearity=args.nonlinearity,
                       act_last=None)

    def forward(self, y, xhs):
        class_feature = []
        for t in self.nz_post.values():
            x_hat = xhs[t]
            ax = torch.matmul(x_hat, self.A.t())
            inputs = torch.cat([y, y-ax, x_hat], dim=-1)
            f = self.mlp(inputs)
            class_feature.append(f)
        features = torch.cat(class_feature, dim=-1)
        scores = self.mlp2(features)
        # p = F.softmax(score, dim=-1)
        return scores


class MulticlassNetImage(nn.Module):
    """docstring for MulticlassNetImage"""
    def __init__(self, args, x, internal_fm, nz_post, category=200, pd=20, net_size=1, 
        device='cuda', target_channel=128, share=False):
        super(MulticlassNetImage, self).__init__()
        self.args = args
        self.internal_fm = internal_fm
        self.nz_post = nz_post
        self.num_class = len(nz_post)
        self.target_channel = target_channel
        self.device = device
        self.pe = af.position_encoding(self.num_class, pd).to(self.device)
        self.input_shape_list = self._get_input_shapes()
        self.used_shape_list = self._get_used_shapes()
        self.smallest_fm_shape = self._get_smallest_shape()
        self.fm_module_list = nn.ModuleList()
        self.flatten_size = target_channel*self.smallest_fm_shape*self.smallest_fm_shape


        self.x_module = nn.Sequential(
                nn.Conv2d(in_channels=3, 
                    out_channels=target_channel, 
                    kernel_size=3, padding=1), 
                nn.BatchNorm2d(target_channel),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(self.smallest_fm_shape),
                af.Flatten(),
                nn.Linear(self.flatten_size, self.flatten_size)
                )

        for shape in self.used_shape_list:
            if len(shape)==4:
                fm_module = nn.Sequential(
                    nn.Conv2d(in_channels=shape[1], 
                        out_channels=target_channel, 
                        kernel_size=3, padding=1), 
                    nn.BatchNorm2d(target_channel),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d(self.smallest_fm_shape), 
                    af.Flatten(),
                    nn.Linear(self.flatten_size, self.flatten_size)
                    )
            if len(shape)==2:
                fm_module = nn.Sequential(
                    nn.Linear(shape[1], self.flatten_size),
                    nn.ReLU()
                    )
            self.fm_module_list.append(fm_module)           
            

        self.total_channels = target_channel*(len(self.used_shape_list)+1)

        self.final_module = nn.Sequential(
            nn.Conv2d(in_channels=self.total_channels, 
                out_channels=target_channel, 
                kernel_size=3, padding=1), 
            nn.BatchNorm2d(target_channel),
            nn.ReLU(),
            nn.Dropout(0.8),
            af.Flatten(),
            nn.Linear(self.flatten_size, self.num_class)
            )

        self.xhs_module_fc = nn.Sequential(
            nn.Linear(category*self.num_class, category),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(category, self.num_class)
            )


        self.encoder_layer = nn.TransformerEncoderLayer(category, 4)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, 2)

        self.xhs_module_conv = nn.Sequential(
            nn.Conv1d(in_channels=category, 
                out_channels=50, 
                kernel_size=1), 
            nn.ReLU(),
            nn.Dropout(0.7),
            af.Flatten(),
            nn.Linear(self.num_class*50, self.num_class)
            )

        self.shared_module = nn.Sequential(
            nn.Linear(category+pd+3+self.flatten_size, 400),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Dropout(0.8)
            )

        self.shared_module_fm = nn.Sequential(
            nn.Linear(self.flatten_size*2+3, 400),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Dropout(0.8)
            )

        self.last_layer = nn.Sequential(
            nn.Linear(200*self.num_class, self.num_class)
            )

    def _get_input_shapes(self):
        input_shape_list = []
        for fm in self.internal_fm:
            input_shape_list.append(fm.shape)
        return input_shape_list

    def _get_used_shapes(self):
        used_shape_list = []
        for i in range(self.num_class):
        # for t in self.nz_post.values():
            t = self.nz_post[i]
            used_shape_list.append(self.internal_fm[t].shape)
        return used_shape_list

    def _get_smallest_shape(self):
        len_list = list(map(lambda x: x[-1], self.used_shape_list))
        return min(len_list)


    def forward(self, x, fm, xhs):
        '''
        xhs: a list of 200dims, t*200 channels
        '''
        # x_map = self.x_module(x)
        # fm_map = list()
        # for i in range(self.num_class):
        #     fm_map.append(self.fm_module_list[i](fm[self.nz_post[i]]))
        # maps = torch.cat([x_map]+fm_map, dim=1)
        # return self.final_module(maps)


        batch_size = x.size(0)
        # pdb.set_trace()

        processed_x = self.x_module(x)
        # processed_x = processed_x.view(batch_size, -1) # batch*flatten_size

        xhs_used = list()
        for i in range(self.num_class):
            # pe = torch.stack([self.pe[i] for _ in range(batch_size)], dim=0)

            fm_processed = self.fm_module_list[i](fm[self.nz_post[i]])
            mme = af.mean_max_entropy(xhs[self.nz_post[i]])
            # go through a shared parameter module
            xhs_used.append(
                self.shared_module_fm(torch.cat([fm_processed, mme, processed_x], dim=-1)))


        # go through the final classification layer

        return self.last_layer(torch.cat(xhs_used, dim=-1))


        # xhs_used = xhs_used.permute(1, 0, -1) # t*batch*200
        # xhs_trans = self.transformer_encoder(xhs_used)
        # xhs_trans = xhs_trans.permute(1, -1, 0) # batch*200*t
        # return self.xhs_module_conv(xhs_trans)


class MNIconfidence(nn.Module):
    """docstring for MNIconfidence"""
    def __init__(self, args, x, internal_fm, nz_post, net_size=1, 
        category=200, pd=20, device='cuda', target_channel=128, share=False):
        super(MNIconfidence, self).__init__()
        self.share = share
        # print('Share parameter: ', self.share)
        self.net_size = net_size
        self.args = args
        self.internal_fm = internal_fm
        self.nz_post = nz_post
        self.num_class = len(nz_post)
        self.target_channel = target_channel
        self.device = device
        self.pe = af.position_encoding(self.num_class, pd).to(self.device)
        self.input_shape_list = self._get_input_shapes()
        self.used_shape_list = self._get_used_shapes()
        self.smallest_fm_shape = self._get_smallest_shape()
        self.flatten_size = target_channel*self.smallest_fm_shape*self.smallest_fm_shape
        
        self.x_module = nn.Sequential(
                nn.Conv2d(in_channels=3, 
                    out_channels=target_channel, 
                    kernel_size=3, padding=1), 
                nn.BatchNorm2d(target_channel),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(self.smallest_fm_shape),
                af.Flatten(),
                nn.Linear(self.flatten_size, self.flatten_size)
                )

        self.unshared_module_list = nn.ModuleList()


        for i in range(self.num_class):
            confidence_module = nn.Sequential(
                nn.Linear(category+pd+3, 200*net_size),
                nn.ReLU(),
                nn.Dropout(0.7),
                nn.Linear(200*net_size, 100*net_size),
                nn.ReLU(),
                nn.Dropout(0.7)
                )
            self.unshared_module_list.append(confidence_module)      
        
        self.shared_module = nn.Sequential(
            nn.Linear(category+pd+3, 200*net_size),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(200*net_size, 100*net_size),
            nn.ReLU(),
            nn.Dropout(0.8)
            )
        self.last_layer = nn.Sequential(
            nn.Linear(100*self.num_class*net_size, self.num_class)
            )


    def _get_input_shapes(self):
        input_shape_list = []
        for fm in self.internal_fm:
            input_shape_list.append(fm.shape)
        return input_shape_list

    def _get_used_shapes(self):
        used_shape_list = []
        for i in range(self.num_class):
        # for t in self.nz_post.values():
            t = self.nz_post[i]
            used_shape_list.append(self.internal_fm[t].shape)
        return used_shape_list

    def _get_smallest_shape(self):
        len_list = list(map(lambda x: x[-1], self.used_shape_list))
        return min(len_list)

    def forward(self, x, fm, xhs):
        batch_size = x.size(0)

        processed_x = self.x_module(x)
        # processed_x = processed_x.view(batch_size, -1) # batch*flatten_size

        xhs_used = list()
        for i in range(self.num_class):
            pe = torch.stack([self.pe[i] for _ in range(batch_size)], dim=0)
            mme = af.mean_max_entropy(xhs[self.nz_post[i]])
            # go through a shared parameter module
            if self.share:
                xhs_used.append(
                    self.shared_module(torch.cat([xhs[self.nz_post[i]], mme, pe], dim=-1)))
            else:
                xhs_used.append(
                    self.unshared_module_list[i](torch.cat([xhs[self.nz_post[i]], mme, pe], dim=-1)))                
        # go through the final classification layer

        return self.last_layer(torch.cat(xhs_used, dim=-1))

        
class Imiconfidence(nn.Module):
    """docstring for MNIconfidence"""
    def __init__(self, args, x, internal_fm, nz_post, net_size=1, 
        category=200, pd=20, device='cuda', target_channel=128, share=False):
        super(Imiconfidence, self).__init__()
        self.share = share
        # print('Share parameter: ', self.share)
        self.net_size = net_size
        self.args = args
        self.internal_fm = internal_fm
        self.nz_post = nz_post
        self.num_class = len(nz_post)
        self.target_channel = target_channel
        self.device = device
        self.pe = af.position_encoding(self.num_class, pd).to(self.device)
        self.input_shape_list = self._get_input_shapes()
        self.used_shape_list = self._get_used_shapes()
        self.smallest_fm_shape = self._get_smallest_shape()
        self.flatten_size = target_channel*self.smallest_fm_shape*self.smallest_fm_shape
        
        self.unshared_module_list = nn.ModuleList()

        for i in range(self.num_class):
            confidence_module = nn.Sequential(
                nn.Linear(category+3, 1),
                # nn.Dropout(0.7)
                )
            self.unshared_module_list.append(confidence_module)      
        
        self.shared_module = nn.Sequential(
            nn.Linear(category+3, 1),
            # nn.Dropout(0.7)
            )


    def _get_input_shapes(self):
        input_shape_list = []
        for fm in self.internal_fm:
            input_shape_list.append(fm.shape)
        return input_shape_list

    def _get_used_shapes(self):
        used_shape_list = []
        for i in range(self.num_class):
        # for t in self.nz_post.values():
            t = self.nz_post[i]
            used_shape_list.append(self.internal_fm[t].shape)
        return used_shape_list

    def _get_smallest_shape(self):
        len_list = list(map(lambda x: x[-1], self.used_shape_list))
        return min(len_list)

    def forward(self, x, fm, xhs):
        batch_size = x.size(0)

        xhs_used = list()
        for i in range(self.num_class):
            pe = torch.stack([self.pe[i] for _ in range(batch_size)], dim=0)
            
            mme = af.mean_max_entropy(xhs[self.nz_post[i]])
            # go through a shared parameter module
            if self.share:
                xhs_used.append(
                    self.shared_module(torch.cat([xhs[self.nz_post[i]], mme], dim=-1)))
            else:
                xhs_used.append(
                    self.unshared_module_list[i](torch.cat([xhs[self.nz_post[i]], mme], dim=-1)))                
        # go through the final classification layer
        return torch.cat(xhs_used, dim=-1)


# model = MulticlassNetImage(cmd_args,x,  internal_fm, nz_post)
# model.to(device)
# tmp = model(x, internal_fm)









