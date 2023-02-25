import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def swish(x):
    return x * torch.sigmoid(x)


def get_gaussian_kernel(kernel_size=3, sigma=2, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2*variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, bias=False, padding=kernel_size//2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    
    return gaussian_filter


class ResBlock(nn.Module):
    def __init__(self, filters=64):
        super(ResBlock, self).__init__()

        self.filters = filters

        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        x_orig = x

        x = self.conv1(x)
        x = swish(x)

        x = self.conv2(x)
        x = swish(x)

        x_out = x_orig + x

        x_out = swish(x_out)

        return x_out

class EBM(nn.Module):
    def __init__(self, channels, num_of_layers=17, iter=10):
        super(EBM, self).__init__()
        self.num_of_layers = num_of_layers
        kernel_size = 3
        padding = 1
        features = 32
        self.layers = nn.ModuleList()

        self.conv_avg = get_gaussian_kernel(kernel_size=7, channels=1)
        self.layers.append(nn.Conv2d(in_channels=2*channels, out_channels=features,
            kernel_size=kernel_size, padding=padding, bias=True))
        # self.layers.append(nn.ReLU(inplace=True))

        for _ in range((num_of_layers-2) // 2):
            self.layers.append(ResBlock(filters=features))

        self.layers.append(nn.Conv2d(in_channels=features,
            out_channels=1, kernel_size=1, padding=0, bias=True))

    def compute_energy(self, x):
        for i, l in enumerate(self.layers):
            x = l(x)

            if i == 0:
                x = swish(x)

        energy = x.mean(dim=-1).mean(dim=-1)
        return energy

    def forward(self, x):

        w = x.size(-1)

        if w >= 64:
            train = False
            # Best setting for noise level of 75

            # 2000 was the one used in the paper
            num_steps = 200
            # step_lr = 2000.0
            step_lr = 1000.0

            # num_steps = 80
            # step_lr = 3000.0
        else:
            train = True
            num_steps = 3
            step_lr = 100.0


        ims = []
        with torch.enable_grad():
            x = torch.clamp(x, 0, 1)
            opt = self.conv_avg(x.clone())
            opt = x.clone()
            opt.requires_grad_()
            ims.append(x.clone())

            for i in range(num_steps):
                inp = torch.cat([opt, x], dim=1)
                energy = self.compute_energy(inp)

                if i > -1:
                    if train:
                        opt_grad, = torch.autograd.grad([energy.sum()], [opt], create_graph=True)
                    else:
                        opt_grad, = torch.autograd.grad([energy.sum()], [opt], create_graph=False)
                else:
                    opt_grad, = torch.autograd.grad([energy.sum()], [opt], create_graph=False)

                opt = opt - opt_grad * step_lr
                opt = torch.clamp(opt, 0, 1)

                # if not train and i < 3:
                #     opt = opt.detach()
                #     opt.requires_grad_()

                if i % 2 == 0:
                    ims.append(opt.detach())

        return opt, ims


class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN, self).__init__()
        self.num_of_layers = num_of_layers
        kernel_size = 3
        padding = 1
        features = 64
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(in_channels=channels, out_channels=features, 
            kernel_size=kernel_size, padding=padding, bias=False))
        self.layers.append(nn.ReLU(inplace=True))

        for _ in range(num_of_layers-2):
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_channels=features, 
                out_channels=features, kernel_size=kernel_size, padding=padding, bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True)
                ))
        self.layers.append(nn.Conv2d(in_channels=features, 
            out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
    def forward(self, x):
        output = list()
        # To save internal layers
        saved_layers = list(range(0,self.num_of_layers-1,3))[1:]
        for i, l in enumerate(self.layers):
            x = l(x)
            if i in saved_layers:
                output.append(x)
        output.append(x)
        return output

class DnCNN_DS(DnCNN):
    def __init__(self, channels, num_of_layers=20):
        super(DnCNN_DS, self).__init__(channels, num_of_layers)
        self.features = 64
        self.kernel_size = 3
        self.padding = 1
        if num_of_layers<27:
            self.saved_layers = list(range(1,self.num_of_layers-1))
        else:
            self.saved_layers = list(range(1,self.num_of_layers-1, 2))
        print(self.saved_layers)
        self.num_internal_layers = len(self.saved_layers)
        self.internal_transform = nn.ModuleList()

        for _ in range(self.num_internal_layers):
            self.internal_transform.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=self.features, out_channels=self.features, 
                        kernel_size=self.kernel_size, padding=self.padding, bias=False),
                    nn.BatchNorm2d(self.features),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=self.features, out_channels=channels, 
                        kernel_size=self.kernel_size, padding=self.padding, 
                        bias=False))
                )

    def forward(self, x):
        output = list()
        # To save internal layers
        saved = 0
        for i, l in enumerate(self.layers):
            x = l(x)
            if i in self.saved_layers:
                output.append(self.internal_transform[saved](x))
                saved = saved+1
        output.append(x)
        return output

class EnhanceUnit(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d( 2, 32, 5, 1, 2),
            nn.PReLU(init=0.1)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.PReLU(init=0.1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.PReLU(init=0.1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1, dilation = 1),
            nn.PReLU(init=0.1)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 2, dilation = 2),
            nn.PReLU(init=0.1)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 4, dilation = 4),
            nn.PReLU(init=0.1)
        )
        self.conv6 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReflectionPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, stride = 1),
            nn.PReLU(init=0.1)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.PReLU(init=0.1)
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(32,  1, 5, 1, 2),
        )
        
    def forward(self, long_x, prev_x):
        x = torch.cat([prev_x, long_x], dim=1)
        x = self.conv0(x)
        x = self.conv1(x)
        res = x
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = torch.cat([x, res], dim=1)
        x = self.conv7(x)
        x = self.conv8(x) + prev_x
        return x


class Recurrent_DS(nn.Module):
    def __init__(self, num_out=5):
        super(Recurrent_DS, self).__init__()
        self.num_out = num_out
        self.unit = EnhanceUnit()

    def forward(self, x):
        output = list()
        org_x = x
        cur_x = x
        for i in range(self.num_out):
            cur_x = self.unit(org_x, cur_x)
            output.append(cur_x)
        return output

# model = Recurrent_DS()
# data = torch.rand(128,1,64,64)
# out = model(data)


class MulticlassNet(nn.Module):
    def __init__(self, args, nz_post, channels, layers=1):
        super(MulticlassNet, self).__init__()
        self.args = args
        self.nz_post = nz_post
        self.num_class = len(nz_post)
        self.channels = channels
        self.features = 64
        self.kernel_size = 3
        self.padding = 1

        self.shared_module = nn.ModuleList()

        self.shared_module.append(nn.Sequential(
            nn.Conv2d(in_channels=channels*3, out_channels=self.features, 
                kernel_size=self.kernel_size, padding=self.padding, bias=False),
            nn.BatchNorm2d(self.features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
            ))

        for _ in range(layers): 
            self.shared_module.append(nn.Sequential(
                nn.Conv2d(in_channels=self.features, out_channels=self.features, 
                    kernel_size=self.kernel_size, padding=self.padding, bias=False),
                nn.BatchNorm2d(self.features),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2), 
                ))

        self.shared_module.append(nn.Sequential(
            nn.Conv2d(in_channels=self.features, out_channels=self.features, 
                kernel_size=self.kernel_size, padding=self.padding, bias=False),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Linear(self.features, 1)
            ))

        self.shared_module_seq = nn.Sequential(*self.shared_module)
        
        # self.last_layer = nn.Sequential(
        #     nn.Linear(self.num_class, self.num_class)
        #     )
    def forward(self, imgn, predns):
        '''
        imgn: batch*1*w*h
        predns: a list of batch*1*w*h
        '''
        scores = list()
        for i in range(self.num_class):
            predn = predns[self.nz_post[i]]
            scores.append(self.shared_module_seq(
                torch.cat([imgn, predn, imgn-predn], dim=1)))

        return torch.cat(scores, dim=-1)






