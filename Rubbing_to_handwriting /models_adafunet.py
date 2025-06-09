import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch
from non_local import NLBlockND 
from torch import einsum
from einops import rearrange
import torch.nn.functional as F



class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        # self.activation = activation

        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)

        out = self.gamma*out + x
        return out,attention
class resBlock(nn.Module):
    def __init__(self, in_channel):
        super(resBlock, self).__init__()

        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channel, in_channel, 3),
            nn.InstanceNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channel, in_channel, 3),
            nn.InstanceNorm2d(in_channel),
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)
class resBlock1(nn.Module):
    def __init__(self, in_channel, out_channel,down=2):
        super(resBlock1, self).__init__()

        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channel, out_channel, 3, stride=down),
            nn.InstanceNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_channel, out_channel, 3),
            nn.InstanceNorm2d(out_channel),
        ]

        self.conv_block = nn.Sequential(*conv_block)
        
        self.conv11 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=down)
        
        # self.11conv = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=2)
    def forward(self, x):
        return self.conv11(x) + self.conv_block(x)

class Decoder(nn.Module):
  def __init__(self, in_channels, middle_channels, out_channels):
    super(Decoder, self).__init__()
    self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    self.conv_relu = nn.Sequential(
        nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
        )
  def forward(self, x1, x2):
    x1 = self.up(x1)
    x2 = F.interpolate(x2, size=x1.shape[2:], mode='bilinear', align_corners=False)
    x1 = torch.cat((x1, x2), dim=1)
    x1 = self.conv_relu(x1)
    return x1

class PreActBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )

        # SE layers
        self.fc1 = nn.Linear(planes, planes//16)
        self.fc2 = nn.Linear(planes//16, planes)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))

        # Squeeze
        w = out.mean((2,3))
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        # Excitation
        out = out * w.unsqueeze(2).unsqueeze(3).expand_as(out)
        out += shortcut
        return out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        a = torch.cat([avg_out, max_out], dim=1)
        a = self.conv1(a)
        a = self.sigmoid(a)
        return a*x

class UNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        
        # self.base_model = torchvision.models.resnet18(True)
        # self.base_layers = list(self.base_model.children())
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            # self.base_layers[1],
            # self.base_layers[2]
            resBlock(64),
            resBlock(64)
            )
        # self.layer2 = nn.Sequential(*self.base_layers[3:5])
        self.layer2 = resBlock1(64,64)
        self.layer3 = resBlock1(64,128)
        self.layer4 = resBlock1(128,256)
        # self.layer4 = self.base_layers[6]
        self.layer5 = resBlock1(256,512)
        self.layer6 = PreActBlock(512,512)
        self.layer7 = SpatialAttention()
        # self.layer11 = Self_Attn(64)
        #self.layer6 = resBlock(512,512)
        self.decode4 = Decoder(512, 256+256, 256)
        self.decode3 = Decoder(256, 256+128, 256)
        self.decode2 = Decoder(256, 128+64, 128)
        self.decode1 = Decoder(128, 64+64, 64)
        self.decode0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
            )
        self.conv_last = nn.Conv2d(64, n_class, 1)
        self.In = nn.InstanceNorm2d(1)
    
  
    def forward(self, input):
        # a1 = self.layer7(input)
        e1 = self.layer1(input)# 64,128,128
        # e1,at = self.layer11(e1)
        e1 = self.layer7(e1)
        e2 = self.layer2(e1) # 64,64,64
       
        e3 = self.layer3(e2) # 128,32,32
        e4 = self.layer4(e3) # 256,16,16
        f = self.layer5(e4)  # 512,8,8
        for _ in range (5):
            # f = self.layer7(f)
            f = self.layer6(f)
        d4 = self.decode4(f, e4) # 256,16,16
        d3 = self.decode3(d4, e3) # 256,32,32
        d2 = self.decode2(d3, e2) # 128,64,64
        d1 = self.decode1(d2, e1) # 64,128,128
        d0 = self.decode0(d1) # 64,256,256
        out = self.conv_last(d0) # 1,256,256
        #out = self.In(out)
        return torch.clamp(out, 0, 1)
        

class UNet1(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        
#####################减少一半通道#############
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
#             resBlock(32),
#             resBlock(32)
#         )
#         self.layer2 = resBlock1(32, 32)
#         self.non_local2 = NLBlockND(in_channels=32, mode='dot', dimension=2, bn_layer=False)

#         self.layer3 = resBlock1(32, 64)
#         self.non_local3 = NLBlockND(in_channels=64, mode='dot', dimension=2, bn_layer=False)

#         self.layer4 = resBlock1(64, 128)
#         self.non_local4 = NLBlockND(in_channels=128, mode='dot', dimension=2, bn_layer=False)

#         self.layer5 = resBlock1(128, 256)
#         self.layer6 = PreActBlock(256, 256)
#         self.layer7 = SpatialAttention()

#         self.decode4 = Decoder(256, 128+128, 128)
#         self.decode3 = Decoder(128, 128+64, 128)
#         self.decode2 = Decoder(128, 64+32, 64)
#         self.decode1 = Decoder(64, 32+32, 32)
#         self.decode0 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#             nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
#             nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False)
#         )
#         self.conv_last = nn.Conv2d(32, n_class, 1)
#         self.In = nn.InstanceNorm2d(1)

###########################减少三分之一通道####
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(1, 42, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
#             resBlock(42),
#             resBlock(42)
#         )
#         self.layer2 = resBlock1(42, 42)
#         self.non_local2 = NLBlockND(in_channels=42, mode='dot', dimension=2, bn_layer=False)

#         self.layer3 = resBlock1(42, 84)
#         self.non_local3 = NLBlockND(in_channels=84, mode='dot', dimension=2, bn_layer=False)

#         self.layer4 = resBlock1(84, 168)
#         self.non_local4 = NLBlockND(in_channels=168, mode='dot', dimension=2, bn_layer=False)

#         self.layer5 = resBlock1(168, 336)
#         self.layer6 = PreActBlock(336, 336)
#         self.layer7 = SpatialAttention()

#         self.decode4 = Decoder(336, 168+168, 168)
#         self.decode3 = Decoder(168, 168+84, 168)
#         self.decode2 = Decoder(168, 84+42, 84)
#         self.decode1 = Decoder(84, 42+42, 42)
#         self.decode0 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#             nn.Conv2d(42, 21, kernel_size=3, padding=1, bias=False),
#             nn.Conv2d(21, 42, kernel_size=3, padding=1, bias=False)
#         )
#         self.conv_last = nn.Conv2d(42, n_class, 1)
#         self.In = nn.InstanceNorm2d(1)


        
##########原先64的通道数####################        
        self.base_model = torchvision.models.resnet18(False)
        self.base_layers = list(self.base_model.children())
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            # self.base_layers[1],
            # self.base_layers[2]
            resBlock(64),
            resBlock(64)
            )
        # self.layer2 = nn.Sequential(*self.base_layers[3:5])
        self.layer2 = resBlock1(64,64)
        self.non_local2 = NLBlockND(in_channels=64, mode='dot', dimension=2, bn_layer=False)

        self.layer3 = resBlock1(64,128)
        self.non_local3 = NLBlockND(in_channels=128, mode='dot', dimension=2, bn_layer=False)

        self.layer4 = resBlock1(128,256)
        self.non_local4 = NLBlockND(in_channels=256, mode='dot', dimension=2, bn_layer=False)

        # self.layer4 = self.base_layers[6]
        self.layer5 = resBlock1(256,512)
        self.layer6 = PreActBlock(512,512)
        self.layer7 = SpatialAttention()
        # self.layer11 = Self_Attn(64)
        #self.layer6 = resBlock(512,512)
        self.decode4 = Decoder(512, 256+256, 256)
        self.decode3 = Decoder(256, 256+128, 256)
        self.decode2 = Decoder(256, 128+64, 128)
        self.decode1 = Decoder(128, 64+64, 64)
        self.decode0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
            )
        self.conv_last = nn.Conv2d(64, n_class, 1)
        self.In = nn.InstanceNorm2d(1)

        
    def forward(self, input):
        # a1 = self.layer7(input)
        e1 = self.layer1(input)# 64,128,128
        # e1,at = self.layer11(e1)
        e1 = self.layer7(e1)
        e2 = self.layer2(e1) # 64,64,64
        e2 = self.non_local2(e2)
        e3 = self.layer3(e2) # 128,32,32
        e3 = self.non_local3(e3)
        e4 = self.layer4(e3) # 256,16,16
        e4 = self.non_local4(e4)
        f = self.layer5(e4)  # 512,8,8
        for _ in range (5):
            # f = self.layer7(f)
            f = self.layer6(f)
        d4 = self.decode4(f, e4) # 256,16,16
        d3 = self.decode3(d4, e3) # 256,32,32
        d2 = self.decode2(d3, e2) # 128,64,64
        d1 = self.decode1(d2, e1) # 64,128,128
        d0 = self.decode0(d1) # 64,256,256
        out = self.conv_last(d0) # 1,256,256
        #################新改的#####
        # f = self.decode4(f, e4) # 256,16,16
        # del e4
        # f = self.decode3(f, e3) # 256,32,32
        # del e3
        # f = self.decode2(f, e2) # 128,64,64
        # del e2
        # f = self.decode1(f, e1) # 64,128,128
        # del e1
        # f = self.decode0(f) # 64,256,256
        # f = self.conv_last(f) # 1,256,256
        # return torch.clamp(f, 0, 1)

        #out = self.In(out)#没用过
        return torch.clamp(out, 0, 1)
        

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)
        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

class UNet2(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        
        # self.base_model = torchvision.models.resnet18(True)
        # self.base_layers = list(self.base_model.children())
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            # self.base_layers[1],
            # self.base_layers[2]
            resBlock(64),
            resBlock(64)
            )
        # self.layer2 = nn.Sequential(*self.base_layers[3:5])
        self.layer2 = resBlock1(64,64)
        self.non_local2 = NLBlockND(in_channels=64, mode='dot', dimension=2, bn_layer=False)

        self.layer3 = resBlock1(64,128)
        self.non_local3 = NLBlockND(in_channels=128, mode='dot', dimension=2, bn_layer=False)

        self.layer4 = resBlock1(128,256)
        self.non_local4 = NLBlockND(in_channels=256, mode='dot', dimension=2, bn_layer=False)

        # self.layer4 = self.base_layers[6]
        self.layer5 = resBlock1(256,512)
        self.layer6 = PreActBlock(512,512)
        self.layer7 = SpatialAttention()
        self.atten = Attention(512)
        # self.layer11 = Self_Attn(64)
        #self.layer6 = resBlock(512,512)
        self.decode4 = Decoder(512, 256+256, 256)
        self.decode3 = Decoder(256, 256+128, 256)
        self.decode2 = Decoder(256, 128+64, 128)
        self.decode1 = Decoder(128, 64+64, 64)
        self.decode0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
            )
        self.conv_last = nn.Conv2d(64, n_class, 1)
        self.In = nn.InstanceNorm2d(1)
    
  
    def forward(self, input):
        # a1 = self.layer7(input)
        e1 = self.layer1(input)# 64,128,128
        # e1,at = self.layer11(e1)
        e1 = self.layer7(e1)
        e2 = self.layer2(e1) # 64,64,64
        # e2 = self.non_local2(e2)
        e3 = self.layer3(e2) # 128,32,32
        # e3 = self.non_local3(e3)
        e4 = self.layer4(e3) # 256,16,16
        # e4 = self.non_local4(e4)
        f = self.layer5(e4)  # 512,8,8

        # for _ in range (5):
        #     # f = self.layer7(f)
        #     f = self.layer6(f)
        f = self.atten(f)
        d4 = self.decode4(f, e4) # 256,16,16
        d3 = self.decode3(d4, e3) # 256,32,32
        d2 = self.decode2(d3, e2) # 128,64,64
        d1 = self.decode1(d2, e1) # 64,128,128
        d0 = self.decode0(d1) # 64,256,256
        out = self.conv_last(d0) # 1,256,256
        #out = self.In(out)
        return torch.clamp(out, 0, 1)
     

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        net = [
            nn.ReflectionPad2d(3),
            #channnel1->3=rgb
            nn.Conv2d(1, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        ]

        #downsample
        in_channel = 64
        out_channel = in_channel * 2
        for _ in range(2):
            net += [
                nn.Conv2d(in_channel, out_channel, 3,
                          stride=2, padding=1),
                nn.InstanceNorm2d(out_channel),
                nn.ReLU(inplace=True),
            ]
            in_channel = out_channel
            out_channel = in_channel * 2
        for _ in range(9):
            net += [resBlock(in_channel)]

        ##upsampleing
        out_channel = in_channel //2
        for _ in range(2):
            net += [nn.ConvTranspose2d(in_channel,
                                       out_channel,
                                       3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.InstanceNorm2d(out_channel),
                    nn.ReLU(inplace=True)
                    ]
            in_channel = out_channel
            out_channel = in_channel // 2

        net += [
            nn.ReflectionPad2d(3),
            #channnel1->3=rgb
            nn.Conv2d(in_channel, 1, 7),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*net)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
         #channnel1->3=rgb
        model = [nn.Conv2d(1, 32, 3, stride=1, padding=1),
                nn.InstanceNorm2d(32),
                nn.LeakyReLU(0.2, inplace=True),
                nn.AvgPool2d(2,2)]

        model += [SpatialAttention()]
        model += [nn.Conv2d(32, 64, 3, stride=1, padding=1),
                 nn.InstanceNorm2d(64),
                 nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
                nn.InstanceNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.AvgPool2d(2,2)]

        model += [nn.Conv2d(128, 256, 3, stride=1, padding=1),
                nn.InstanceNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.AvgPool2d(2,2)]

        model += [nn.Conv2d(256, 1, 3, stride=1, padding=1),
                nn.InstanceNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.AvgPool2d(2,2)]
        
        self.line = nn.Linear(16 * 16,1)
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        x = self.line(x.reshape(x.size(0),x.size(2) * x.size(3)))
        #print(x.size())
        return x



if __name__=='__main__':
    G = UNet2(1)
    # G = Generator()
    D = Discriminator()
   
    input_tensor = torch.ones((1, 1, 512, 512),dtype=torch.float)
    out= G(input_tensor)
    print(out.size())

    out = D(input_tensor)
    print(out.size())


