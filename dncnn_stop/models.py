import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


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






