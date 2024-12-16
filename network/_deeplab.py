import einops
import torch
from torch import nn
from torch.nn import functional as F

from .utils import _SimpleSegmentationModel
import torch.fft as fft

from scipy.spatial import ConvexHull



__all__ = ["DeepLabV3"]


class DeepLabV3(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass

class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus, self).__init__()
        self.project = nn.Sequential( 
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(in_channels, aspp_dilate)

        self.head = nn.ModuleList(
            [
                 nn.Sequential(
                    nn.Conv2d(304, 256, 3, padding=1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, c, 1)
                ) for c in num_classes]
        )
        
        self._init_weight()

    def forward(self, feature):
        low_level_feature = self.project( feature['low_level'] )
        output_feature = self.aspp(feature['out'])
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)
        
        output_feature = torch.cat( [ low_level_feature, output_feature ], dim=1 )
        
        heads = [h(output_feature) for h in self.head]
        heads = torch.cat(heads, dim=1)
        
        return heads
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #nn.init.kaiming_normal_(m.weight)
                nn.init.normal_(m.weight, mean=0, std=0.001)
                #nn.init.normal_(m.weight, mean=5, std=0.01)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def _head_initialize(self):
        for m in self.head:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)


class FeatureProjection(nn.Module):
    def __init__(self,in_channels):
        super(FeatureProjection, self).__init__()
        self.translation = nn.Parameter(torch.zeros(1, 2, 1, 1).cuda())  
        self.scale = nn.Parameter(torch.ones(1) * 0.5).cuda()
        # self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        # self.relu = nn.ReLU(inplace=True)
    def forward(self, feature):
        feature = feature.float() 
        device = feature.device  

        b, num_classes, h, w = feature.size()

        projected_feature = torch.zeros((b, num_classes, h, w), dtype=torch.float32, device=device)

       
        for i in range(num_classes):
           
            class_feature = feature[:, i, :, :] 
            class_feature = F.interpolate(class_feature.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=False)
            class_feature = class_feature.squeeze(1)

            coords = torch.zeros((b, 3, h, w), dtype=torch.float32, device=device)
            coords[:, 0, :, :] = torch.arange(w, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  
            coords[:, 1, :, :] = torch.arange(h, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(2) 
            coords[:, 2, :, :] = class_feature 

            nonzero_indices = torch.nonzero(coords[:, 2, :, :])

            if nonzero_indices.numel() > 4: 
                nonzero_coords = coords[nonzero_indices[:, 0], :, nonzero_indices[:, 1], nonzero_indices[:, 2]]

                try:
                    convex_hull = ConvexHull(nonzero_coords.cpu().detach().numpy())
                    vertices = nonzero_coords[convex_hull.vertices]
                    diameter = torch.norm(vertices, dim=1).max()
                    self.translation.data = torch.zeros(1, 2, 1, 1, device=device)
                    self.translation.data[:, :, 0, 0] = diameter

                    translation = self.translation
                    scale = self.scale
                    coords[:, :2, :, :] += translation
                    coords *= scale
                    projected_feature[:, i, :, :] = coords[:, 2, :, :]

                except Exception as e:
                    projected_feature = feature
                    return projected_feature
            else:
                diameter = 1.0  

                self.translation.data = torch.zeros(1, 2, 1, 1, device=device)
                self.translation.data[:, :, 0, 0] = diameter

                translation = self.translation
                scale = self.scale
                coords[:, :2, :, :] += translation
                coords *= scale
                projected_feature[:, i, :, :] = coords[:, 2, :, :]

        return projected_feature


class FeatureOperation(nn.Module):
    def __init__(self, hidden_dim):
        super(FeatureOperation, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, feature, feature1):
        batch_size, channels, height, width = feature.size()
        feature1 = feature1.to(feature.dtype)
        attended_values = feature * 0.8 + feature1 * 0.2
        attended_values = attended_values.view(batch_size, channels, height, width)
        return attended_values

# v3 classifier head
class DeepLabHead(nn.Module):
    def __init__(self, in_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHead, self).__init__()

        self.aspp = ASPP(in_channels, aspp_dilate)
        
        self.head = nn.ModuleList(
            [
                 nn.Sequential(
                    nn.Conv2d(256, 256, 3, padding=1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                ) for c in num_classes]
        )


        self.head2 = nn.ModuleList(
            [nn.Sequential(
                    nn.Conv2d(256, c, 1, bias=True) # True
                ) for c in num_classes]
        )
        self.proposal_head = nn.ModuleList(
            [nn.Sequential(
                    nn.Conv2d(256, 256, 3, padding=1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, 3, padding=1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, 3, padding=1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, c, 1, bias=True) # True
                ) for c in num_classes]
        )

        self._init_weight()

    def forward(self, feature, proposal_flag, proposal=None):
        """

        :param feature:
        :param proposal_flag:
        :param proposal: b 100 h' w' one hot
        :return:
        """
        
        feature = self.aspp(feature['out']) 
        #Feature Reconstruction: Compression and Sparsity
        projection = FeatureProjection(feature.shape[1])
        feature1 = projection(feature)
        #Integrating Original Features and Reshaped Features
        hidden_dim = 256
        self_attention = FeatureOperation(hidden_dim)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self_attention = self_attention.to(device)
        feature = self_attention(feature, feature1)
        logits = [h(feature) for h in self.head]

        heads = []
        for i,h in enumerate(self.head2):
            heads.append(h(logits[i]))

        heads = torch.cat(heads, dim=1)


        proposal = (F.interpolate(
            input=proposal.float(), size=(feature.shape[2], feature.shape[3]),
            mode='nearest')).float() # b 100 h' w'
        PPs = []
        '''transed gap'''
        _, proposal_max = proposal.max(dim=1)
        for i in range(feature.shape[0]):
            feature_sample = feature[i] # c h w
            proposal_sample = proposal_max[i] # h w
            feature_sample = einops.rearrange(feature_sample, 'c h w ->(h w) c ')
            proposal_sample = einops.rearrange(proposal_sample,' h w -> ( h w )')
            PP = []
            for i in range(proposal.shape[1]):
                mask = (proposal_sample == i)
                prototype_sample = feature_sample[mask]
                if not prototype_sample.shape[0]==0:
                    prototype_sample=prototype_sample.mean(dim=0)
                else:
                    prototype_sample=torch.zeros(feature_sample.shape[1]).to(feature_sample) +1e-9

                PP.append(prototype_sample)

            PP = torch.stack(PP,dim=0) # 100 c
            PPs.append(PP)

        PPs = torch.stack(PPs, dim=0)  # b 100 c


        B_,N_,C_ = PPs.shape
        PPs = PPs.view(B_*N_,-1)
        PPs = PPs.unsqueeze(-1).unsqueeze(-1)

        cl = [ph(PPs) for ph in self.proposal_head]

        cl = torch.cat(cl, dim=1)
        cl = cl.view(B_,N_,cl.shape[1])

        return cl, heads, feature





    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def _head_initialize(self):
        for m in self.head:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

class AtrousSeparableConvolution(nn.Module):
    """ Atrous Separable Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                            stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # Separable Conv
            nn.Conv2d( in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=in_channels ),
            # PointWise Conv
            nn.Conv2d( in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )
        
        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)



def convert_to_separable_conv(module):
    new_module = module
    if isinstance(module, nn.Conv2d) and module.kernel_size[0]>1:
        new_module = AtrousSeparableConvolution(module.in_channels,
                                      module.out_channels, 
                                      module.kernel_size,
                                      module.stride,
                                      module.padding,
                                      module.dilation,
                                      module.bias)
    for name, child in module.named_children():
        new_module.add_module(name, convert_to_separable_conv(child))
    return new_module