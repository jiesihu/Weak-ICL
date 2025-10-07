from dataclasses import dataclass, field
from pydantic import validate_arguments
from .nn.vmap import Vmap, vmap
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import *


from .nn.layers import ResidualUnit
from .nn.layers import ConvBlock_target_encoder, ConvBlock_context_c2t,\
        ConvBlock_context_t2c, PairwiseConvAvgModelBlock_c2t, PairwiseConvAvgModelOutput
from .nn.unet_backbone import TargetDecoder, TargetEncoder, ContextUNet

class Neuroverse3D(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 stages,
                 dim,
                 inner_channels,
                 conv_layers_per_stage,
                ):
        super().__init__()
        self.context_unet = ContextUNet(in_channels = in_channels,
                    out_channels = out_channels, 
                    stages = stages,
                    dim = dim,
                    inner_channels = inner_channels,
                    unet_block_cls_t2c = ConvBlock_context_t2c,
                    unet_block_cls_c2t = ConvBlock_context_c2t,
                    kwargs = {'conv_layers_per_stage':conv_layers_per_stage})
        
        self.target_encoder = TargetEncoder(in_channels = in_channels, 
                    out_channels = out_channels, 
                    stages = stages, 
                    dim = dim,
                    unet_block_cls_t2c = ConvBlock_target_encoder,
                    inner_channels = inner_channels,
                    kwargs = {'conv_layers_per_stage':conv_layers_per_stage})
        
        self.target_decoder = TargetDecoder(in_channels = in_channels, 
                                    out_channels = out_channels, 
                                    stages = stages, 
                                    dim = dim,
                                    inner_channels = inner_channels,
                                    unet_block_cls_c2t = PairwiseConvAvgModelBlock_c2t,
                                    output_block_cls = PairwiseConvAvgModelOutput,
                                    kwargs = {'conv_layers_per_stage':conv_layers_per_stage})
    
        self.dim = dim
        self.stages = stages
    def fuse_feature(self, Features, features, Weight, weight = 1):
        # Fuse the context features into mean context features.
        ori_context_weight = Weight/(Weight+weight)
        new_context_weight = weight/(Weight+weight)
        for ind in range(len(Features)):
            Features[ind] = Features[ind]*ori_context_weight+features[ind]*new_context_weight
        Weight+=weight
        return Features, Weight
    
    def forward(self, target_in, context_in, context_out, l=3):
        '''
        Args:
            context_in (torch.Tensor): Context input, shape BxLxCinxHxWxD. Could be store in the cpu memory.
            context_out (torch.Tensor): Context output, shape BxLxCoutxHxWxD. Could be store in the cpu memory.
            target_in (torch.Tensor): Target input, shape BxCinxHxWxD
            l (int): The mini-context size
            
        Returns:
            torch.Tensor: Target output, shape BxCoutxHxWxD
            
        '''
        # 处理 target 编码器。Target processing.
        shortcuts, target = self.target_encoder(target_in)
        target_features = [i[1] for i in shortcuts] # list of [BxCxHxWxD]
        
        # 处理 context。Context processing.
        # initialize vars
        Weight = 0
        context_features_mean = [0 for _ in range(self.stages)]

        # Process the 1 to (n-1) mini-batch
        with torch.no_grad():
            for index in range(math.ceil(context_in.shape[1]/l)-1):
#                 print('---Use the sequential processing---'*3)
                context_features = self.context_unet(context_in[:,index*l:index*l+l,:].to(next(self.parameters()).device), # To save GPU memory 
                                                         context_out[:,index*l:index*l+l,:].to(next(self.parameters()).device), 
                                                         target_features) # list of [BxLxCxHxWxD]

                # update the context_features_mean
                tmp_context_features_mean = [i.mean(dim=1, keepdim=True) for i in context_features] # list of [Bx1xCxHxWxD]
                context_features_mean, Weight = self.fuse_feature(context_features_mean, 
                                                    tmp_context_features_mean, 
                                                    Weight, 
                                                    weight = context_features[0].shape[1])
        # Detach
        context_features_mean = [c.detach() if type(c) is not int else c for c in context_features_mean]
        
        # Process the last mini-batch
        index = math.ceil(context_in.shape[1]/l)-1
        context_features_last = self.context_unet(context_in[:,index*l:,:].to(next(self.parameters()).device), # To save GPU memory 
                                                     context_out[:,index*l:,:].to(next(self.parameters()).device), 
                                                     target_features) # list of [BxLxCxHxWxD]
#         print('----last one----'*3)
        # update the context_features_mean
        tmp_context_features_last_mean = [i.mean(dim=1, keepdim=True) for i in context_features_last] # list of [Bx1xCxHxWxD]
        
        context_features_mean, Weight = self.fuse_feature(context_features_mean, 
                                            tmp_context_features_last_mean, 
                                            Weight, 
                                            weight = context_features_last[0].shape[1])
        
        # 处理 target 解码器，并与 context 特征交互。 Target decoder processing with context features.
        output = self.target_decoder(target, context_features_mean, shortcuts) # BxCoutxHxWxD
        
        
        return output


if __name__ == '__main__':
    device = 'cuda'
    unet2d = PairwiseConvAvgModel(dim=2, stages=4, in_channels=2,
                                  out_channels=3, inner_channels=32, conv_layers_per_stage=2).to(device)
    context_in = torch.rand(7, 8, 2, 64, 64).to(device)
    context_out = torch.rand(7, 8, 3, 64, 64).to(device)
    target_in = torch.rand(7, 2, 64, 64).to(device)
    target_out = unet2d(context_in, context_out, target_in)
    print(unet2d)
    print('2d ok')

    unet3d = PairwiseConvAvgModel(dim=3,
                                  stages=4,
                                  in_channels=2,
                                  out_channels=3,
                                  inner_channels=32,
                                  conv_layers_per_stage=2).to(device)
    context_in = torch.rand(7, 8, 2, 64, 64, 32).to(device)
    context_out = torch.rand(7, 8, 3, 64, 64, 32).to(device)
    target_in = torch.rand(7, 2, 64, 64, 32).to(device)
    target_out = unet3d(context_in, context_out, target_in)
    print(unet3d)
    print('3d ok')
