#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Shi Qiu (based on Yue Wang's codes)
@Contact: shi.qiu@anu.edu.au
@File: model.py
@Time: 2021/04/23
"""


import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from util import PointNetSetAbstraction , PointNetFeaturePropagation


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  
    return idx


def geometric_point_descriptor(x, k=3, idx=None):
   
    batch_size = x.size(0)
    num_points = x.size(2)
    org_x = x
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k) 
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx_base = idx_base.type(torch.cuda.LongTensor)
    idx = idx.type(torch.cuda.LongTensor)
    idx = idx + idx_base 
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous() 
    neighbors = x.view(batch_size * num_points, -1)[idx, :]
    neighbors = neighbors.view(batch_size, num_points, k, num_dims)

    neighbors = neighbors.permute(0, 3, 1, 2)  # B,C,N,k
    neighbor_1st = torch.index_select(neighbors, dim=-1, index=torch.cuda.LongTensor([1])) # B,C,N,1
    neighbor_1st = torch.squeeze(neighbor_1st, -1)  # B,3,N
    neighbor_2nd = torch.index_select(neighbors, dim=-1, index=torch.cuda.LongTensor([2])) # B,C,N,1
    neighbor_2nd = torch.squeeze(neighbor_2nd, -1)  # B,3,N

    edge1 = neighbor_1st-org_x
    edge2 = neighbor_2nd-org_x
    normals = torch.cross(edge1, edge2, dim=1) # B,3,N
    dist1 = torch.norm(edge1, dim=1, keepdim=True) # B,1,N
    dist2 = torch.norm(edge2, dim=1, keepdim=True) 

    new_pts = torch.cat((org_x, normals, dist1, dist2, edge1, edge2), 1) 

    return new_pts

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx_base = idx_base.type(torch.cuda.LongTensor)
    idx = idx.type(torch.cuda.LongTensor)
    idx = idx + idx_base
    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2)
  
    return feature


class CAA_Module(nn.Module):
    """ Channel-wise Affinity Attention module"""
    def __init__(self, in_dim):
        super(CAA_Module, self).__init__()

        self.bn1 = nn.BatchNorm1d(1024//8)
        self.bn2 = nn.BatchNorm1d(1024//8)
        self.bn3 = nn.BatchNorm1d(in_dim)

        self.query_conv = nn.Sequential(nn.Conv1d(in_channels=1024, out_channels=1024//8, kernel_size=1, bias=False),
                                        self.bn1,
                                        nn.ReLU())
        self.key_conv = nn.Sequential(nn.Conv1d(in_channels=1024, out_channels=1024//8, kernel_size=1, bias=False),
                                        self.bn2,
                                        nn.ReLU())
        self.value_conv = nn.Sequential(nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1, bias=False),
                                        self.bn3,
                                        nn.ReLU())

        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X N )
            returns :
                out : output feature maps( B X C X N )
        """

        # Compact Channel-wise Comparator block
        x_hat = x.permute(0, 2, 1)#[8,1024,64]
        proj_query = self.query_conv(x_hat) #[8,128,64]
        proj_key = self.key_conv(x_hat).permute(0, 2, 1) #[8,64,128]
        similarity_mat = torch.bmm(proj_key, proj_query) #[8,64,64]

        # Channel Affinity Estimator block
        affinity_mat = torch.max(similarity_mat, -1, keepdim=True)[0].expand_as(similarity_mat)-similarity_mat
        affinity_mat = self.softmax(affinity_mat)#[8,64,64]
        
        proj_value = self.value_conv(x)#[8,64,1024]
        out = torch.bmm(affinity_mat, proj_value)
        # residual connection with a learnable weight
        out = self.alpha*out + x 
        return out
    

class ABEM_Module(nn.Module):
    """ Attentional Back-projection Edge Features Module (ABEM)"""
    def __init__(self, in_dim, out_dim, k):
        super(ABEM_Module, self).__init__()

        self.k = k
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.conv1 = nn.Sequential(nn.Conv2d(in_dim*2, out_dim, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.bn2 = nn.BatchNorm2d(in_dim)
        #LFC KERNEL_SIZE[1,K]
        self.conv2 = nn.Sequential(nn.Conv2d(out_dim, in_dim, kernel_size=[1,self.k], bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2)) 

        self.bn3 = nn.BatchNorm2d(out_dim)
        self.conv3 = nn.Sequential(nn.Conv2d(in_dim*2, out_dim, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.caa1 = CAA_Module(out_dim)

        self.bn4 = nn.BatchNorm2d(out_dim)
        self.conv4 = nn.Sequential(nn.Conv2d(in_dim*2, out_dim, kernel_size=[1,self.k], bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.caa2 = CAA_Module(out_dim)

    def forward(self,x):
        # Prominent Feature Encoding
        x1 = x # input：[[b,14,1024]
        input_edge = get_graph_feature(x, k=self.k) #[8,input_dim*2,1024,K] (xi,xi-xj)[24,28,1024,20]
        x = self.conv1(input_edge)#[8,out_dim,1024,K]
        x2 = x # EdgeConv for input [B,out_dim,N,K]

        x = self.conv2(x) # LFC [B,in_dim,][B,in_dim,1024,1]
        x = torch.squeeze(x, -1)
        x3 = x # Back-projection signal

        delta = x3 - x1 # Error signal [24,14,1024]

        x = get_graph_feature(delta, k=self.k)  # EdgeConv for Error signal [b,28,1024,20]
        x = self.conv3(x)#[24,64,1024,20]
        x4 = x
        x = x2 + x4 # Attentional feedback
        x = x.max(dim=-1, keepdim=False)[0]#[24,64,1024]
        x = self.caa1(x) # B,out_dim,N

        # Fine-grained Feature Encoding
        x_local = self.conv4(input_edge)#[b,64,N]
        x_local = torch.squeeze(x_local, -1) 
        x_local = self.caa2(x_local) # B,out_dim,N

        return x, x_local


class PointNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        gloablfeat = x

        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x,gloablfeat


class DGCNN(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = args.k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)#[24,1024,1024]
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)#[24,2048]      
        globalfeat = x
        
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)#[24,]
        return x , globalfeat


class GBNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(GBNet, self).__init__()
        self.args = args
        self.k = args.k #20
        
        self.abem1 = ABEM_Module(14, 64, self.k) #input:14 output:64
        self.abem2 = ABEM_Module(64, 64, self.k)
        self.abem3 = ABEM_Module(64, 128, self.k)
        self.abem4 = ABEM_Module(128, 256, self.k)

        self.bn = nn.BatchNorm1d(args.emb_dims) #args.emb_dims:1024
        self.conv = nn.Sequential(nn.Conv1d(1024, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.caa = CAA_Module(1024) #input:(B,C,N) output:(B,C,N)

        self.linear1 = nn.Linear(args.emb_dims * 2, 512, bias=False)
        self.bn_linear1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn_linear2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)
        
        #self.projectionhead = nn.Linear(2048,2048)
       # self.bn_projectionhead = nn.BatchNorm1d(1024)
        
 
        #self.conv1 = torch.nn.Conv1d(2560, 1024, 1)
        #self.conv2 = torch.nn.Conv1d(1024, 512, 1)
        #self.conv3 = torch.nn.Conv1d(512, 256, 1)
        #self.conv4 = torch.nn.Conv1d(256, 128, 1)
        #self.conv5 = torch.nn.Conv1d(128, 2, 1)
        #self.conv6 = torch.nn.Conv1d(2, 1, 1)
        #self.bn1 = nn.BatchNorm1d(1024)
        #self.bn2 = nn.BatchNorm1d(512)
        #self.bn3 = nn.BatchNorm1d(256)
        #self.bn4 = nn.BatchNorm1d(128)
        #self.bn5 = nn.BatchNorm1d(2)

    def forward(self, x):
        # x: B,3,N
        batch_size = x.size(0)

        # Geometric Point Descriptor:
        x = geometric_point_descriptor(x) # B,14,N

        # 1st Attentional Back-projection Edge Features Module (ABEM):
        x1, x1_local = self.abem1(x) #x1:[24,64,1024] x1_local:[24,64,1024]

        # 2nd Attentional Back-projection Edge Features Module (ABEM):
        x2, x2_local = self.abem2(x1)#x2:[24,64,1024] x2_local:[24,64,1024]

        # 3rd Attentional Back-projection Edge Features Module (ABEM):
        x3, x3_local = self.abem3(x2)#x3:[24,128,1024] x3_local:[24,128,1024]

        # 4th Attentional Back-projection Edge Features Module (ABEM):
        x4, x4_local = self.abem4(x3)#x4:[24,256,1024] x4_local:[24,256,1024]

        # Concatenate both prominent and fine-grained outputs of 4 ABEMs:
        x = torch.cat((x1, x1_local, x2, x2_local, x3, x3_local, x4, x4_local), dim=1)  # B,(64+64+128+256)x2,N
        x = self.conv(x) 
        x = self.caa(x) # B,1024,1024 ,
       

        # global embedding
        global_max = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        global_avg = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((global_max, global_avg), 1) 
        globalfeat = x

        # FC layers with dropout
        x = F.leaky_relu(self.bn_linear1(self.linear1(x)), negative_slope=0.2)  # B,512
        x = self.dp1(x)
        x = F.leaky_relu(self.bn_linear2(self.linear2(x)), negative_slope=0.2)  # B,256
        x = self.dp2(x)
        x = self.linear3(x)  # B,C

        ############SEGMENTATION BRANCH
        #global_feature = F.normalize(globalfeat, p=2, dim=1)#[24,2048]
        #x1_local = F.normalize(x1_local, p=2, dim=-1)#[B,64,1024]
        #x2_local = F.normalize(x2_local, p=2, dim=-1)#[B,64,1024]
        #x3_local = F.normalize(x3_local, p=2, dim=-1)#[B,128,1024]
        #x4_local = F.normalize(x4_local, p=2, dim=-1)#[B,256,1024]
      
        #global_feature = global_feature.view(batch_size,2048,-1).repeat(1,1,1024)#[24,2048,1024]
        #Propagationfeat = torch.cat((global_feature,x1_local,x2_local,x3_local,x4_local), dim=1) #(24,2560,1024)


        #seg_x = F.relu(self.bn1(self.conv1(Propagationfeat)))#[B,2560,1024] -[B,1024,1024]
        #seg_x = F.relu(self.bn2(self.conv2(seg_x)))#[B,512,1024]
        #seg_x = F.relu(self.bn3(self.conv3(seg_x)))#[B,256,1024]
        #seg_x = F.relu(self.bn4(self.conv4(seg_x)))#[B,128,1024]
        #seg_x = F.relu(self.bn5(self.conv5(seg_x)))#[B,2,1024]
        #seg_pred = self.conv6(seg_x)#[B,1,1024]
        ##seg_x = seg_x.transpose(2,1).contiguous()
        #seg_pred = F.softmax(seg_x, dim=-1)#[24,1,1024]
        #projectionhead_feat = F.relu(self.projectionhead(globalfeat))
        return x ,globalfeat #,projectionhead_feat #,seg_pred

class BGA(nn.Module):
    def __init__(self, args, output_channels):
        super(BGA, self).__init__()
        in_channel = 3 
        #self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=64, in_channel = in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel = 128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, output_channels)
        
        self.fp3 = PointNetFeaturePropagation(512, [256, 256])
        self.fp2 = PointNetFeaturePropagation(384, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        
        self.segfc1 = nn.Conv1d(in_channels = 128, out_channels = 2,kernel_size = 1)
        self.segdrop1 = nn.Dropout(0.5) 
        self.segfc2 = nn.Conv1d(in_channels=2, out_channels=1,kernel_size = 1) 
        #self.projectionhead = nn.Linear(1024,1024)

    def forward(self, xyz):
        B, _, _ = xyz.shape 
        #if self.normal_channel:
        #    norm = xyz[:, 3:, :]
        #    xyz = xyz[:, :3, :]
        #else:
        norm = None

        l1_xyz, l1_points = self.sa1(xyz, norm)#sample 512 c:128 [b,3,512] [b,128,512]
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)#sample 128 C:256  [b,3,128] [b,256,128]
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)#poooling c 1024  [b,3,1] [b,1024,1]
   
        ###########CLASSIFICATION BRANCH
        x = l3_points.view(B, 1024)
        global_feature = x
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        class_vector = x.unsqueeze(2)#why Unsqueeze#[24,15]        
        x = self.fc3(x)
        class_pred = F.log_softmax(x, -1)#24,15
        ###########SEGMENTATION BRANCH
        # Feature Propagation layers
        #l3_points_concat = torch.cat([l3_points, class_vector], 1)

        #l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, class_vector)#l2_xyz:(24,3,128),l3_xyz::(24,3,1),12_points:(24,256,128),class_vector:[24,256,1]
        ##l2_points：[24,256,128]
        #l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)#[][24,3,512] [24,3,128][24,128,256][24,256,128]
        ## l1_points：[24,128,512]
        #l0_points = self.fp1(xyz, l1_xyz, None, l1_points) #l0_points：[24,128,1024]

        #net = self.segfc1(l0_points)
        #net = self.segdrop1(net)
        #seg_pred = self.segfc2(net)
        #seg_pred = F.softmax(seg_pred, dim=-1)
        ##projection to contrast
        #projectionhead_feat = F.relu(self.projectionhead(global_feature))
        return class_pred, global_feature#,seg_pred#[24,15] [24,1,1024]
        #return class_pred


class ContrastNet(nn.Module):
    def __init__(self, args, output_channels):
        super(ContrastNet, self).__init__()
        self.args = args
        self.k = args.k #20
        
        self.abem1 = ABEM_Module(14, 64, self.k) #input:14 output:64
        self.abem2 = ABEM_Module(64, 64, self.k)
        self.abem3 = ABEM_Module(64, 128, self.k)
        self.abem4 = ABEM_Module(128, 256, self.k)

        self.bn = nn.BatchNorm1d(args.emb_dims) #args.emb_dims:1024
        self.conv = nn.Sequential(nn.Conv1d(1024, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn,
                                   nn.LeakyReLU(negative_slope=0.2))        
        self.caa = CAA_Module(1024) #input:(B,C,N) output:(B,C,N)

        self.linear1 = nn.Linear(args.emb_dims * 2, 512, bias=False)
        self.bn_linear1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn_linear2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)
    
        
        #self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(512, [256, 256])
        self.fp2 = PointNetFeaturePropagation(384, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        
        self.segfc1 = nn.Conv1d(in_channels = 128, out_channels = 2,kernel_size = 1)
        self.segdrop1 = nn.Dropout(0.5) 
        self.segfc2 = nn.Conv1d(in_channels=2, out_channels=1,kernel_size = 1) 

    def forward(self, xyz):
         # x: B,3,N
        batch_size = x.size(0)

        # Geometric Point Descriptor:
        x = geometric_point_descriptor(x) # B,14,N

        # 1st Attentional Back-projection Edge Features Module (ABEM):
        x1, x1_local = self.abem1(x) 

        # 2nd Attentional Back-projection Edge Features Module (ABEM):
        x2, x2_local = self.abem2(x1)

        # 3rd Attentional Back-projection Edge Features Module (ABEM):
        x3, x3_local = self.abem3(x2)

        # 4th Attentional Back-projection Edge Features Module (ABEM):
        x4, x4_local = self.abem4(x3)

        # Concatenate both prominent and fine-grained outputs of 4 ABEMs:
        x = torch.cat((x1, x1_local, x2, x2_local, x3, x3_local, x4, x4_local), dim=1)  # B,(64+64+128+256)x2,N
        x = self.conv(x) 
        x = self.caa(x) # B,1024,N

        # global embedding
        global_max = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)#（8,1024）
        global_avg = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)#(8,1024)
        x = torch.cat((global_max, global_avg), 1)  # B,2048
        global_feature = x

        ###########CLASSIFICATION BRANCH
        # FC layers with dropout
        x = F.leaky_relu(self.bn_linear1(self.linear1(x)), negative_slope=0.2)  # B,512
        x = self.dp1(x)
        x = F.leaky_relu(self.bn_linear2(self.linear2(x)), negative_slope=0.2)  # B,256
        x = self.dp2(x)
        x = self.linear3(x)  # B,C
     
        B, _, _ = xyz.shape 
        #if self.normal_channel:
        #    norm = xyz[:, 3:, :]
        #    xyz = xyz[:, :3, :]
        #else:
        norm = None

        ###########CLASSIFICATION BRANCH
        l1_xyz, l1_points = self.sa1(xyz, norm)#sample 512 c:128
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)#sample 128 C:256
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)#poooling c 1024
    
   
   
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        class_vector = x.unsqueeze(2)#why Unsqueeze#[24,15]
        
        x = self.fc3(x)
        class_pred = F.log_softmax(x, -1)#24,15
       
       ###########SEGMENTATION BRANCH
        # Feature Propagation layers
        l3_points_concat = torch.cat([l3_points, class_vector], 1)

        
        # l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points_concat, [256,256], is_training, bn_decay, scope='fa_layer1')
        #l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, class_vector, [256,256], is_training, bn_decay, scope='fa_layer1')
        #l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256,128], is_training, bn_decay, scope='fa_layer2')
        #l0_points = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points, [128,128,128], is_training, bn_decay, scope='fa_layer3')
        
        #l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, class_vector)#l2_xyz:(24,3,128),l3_xyz::(24,3,1),12_points:(24,256,128),class_vector:[24,256,1]
        #l2_points：[24,256,128]
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)#[][24,3,512] [24,3,128][24,128,256][24,256,128]
        # l1_points：[24,128,512]
        l0_points = self.fp1(xyz, l1_xyz, None, l1_points)#l0_points：[24,128,1024]

        #print('l0_points:',l0_points.shape)
        #net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='seg_fc1', bn_decay=bn_decay)
        net = self.segfc1(l0_points)
        net = self.segdrop1(net)
        seg_pred = self.segfc2(net)#
        seg_pred = F.softmax(seg_pred, dim=-1)
    
        #end_points['feats'] = net 
        #net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='seg_dp1')
        #seg_pred = tf_util.conv1d(net, 2, 1, padding='VALID', activation_fn=None, scope='seg_fc2')

        return class_pred, global_feature ,seg_pred #[24,15] [24,1,1024]

