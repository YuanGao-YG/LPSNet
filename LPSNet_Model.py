# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 16:14:37 2021

@author: Administrator
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from thop import profile


class LPSNet(nn.Module):  #Learning Parameter Sharing-based Low Visibility Image Enhancement in Maritime Surveillance System
	def __init__(self,channel = 8):
		super(LPSNet,self).__init__()

		self.Haze_E = Encoder(channel)#channel = 16
		self.Low_E  = Encoder(channel)#channel = 16

		self.Share  = ShareNet(channel)#channel = 16

		self.Haze_D = Decoder(channel)#channel = 16
		self.Low_D  = Decoder(channel)#channel = 16

		self.Haze_in = nn.Conv2d(3,channel,kernel_size=1,stride=1,padding=0,bias=False)#3 16 对由多个输入平面组成的输入信号进行二维卷积
		self.Haze_out = nn.Conv2d(channel,3,kernel_size=1,stride=1,padding=0,bias=False)#16 3

		self.Low_in = nn.Conv2d(3,channel,kernel_size=1,stride=1,padding=0,bias=False)#3 16
		self.Low_out = nn.Conv2d(channel,3,kernel_size=1,stride=1,padding=0,bias=False)#16 3

	def forward(self,x,Type):

		if   Type == 2:
			x_in = self.Haze_in(x)#3 16
			L,M,S,SS = self.Haze_E(x_in)
			Share = self.Share(SS)
			x_out = self.Haze_D(Share,SS,S,M,L)
			out = self.Haze_out(x_out) + x
            
		elif Type == 1:
			x_in = self.Low_in(x)             
			L,M,S,SS = self.Low_E(x_in)
			Share = self.Share(SS)
			x_out = self.Low_D(Share,SS,S,M,L)
			out = self.Low_out(x_out) + x

		return out

class Encoder(nn.Module):
	def __init__(self,channel):
		super(Encoder,self).__init__()    

		self.el = ResidualBlock(channel)#16
		self.em = ResidualBlock(channel*2)#32
		self.es = ResidualBlock(channel*4)#64
		self.ess = ResidualBlock(channel*8)#128
		self.esss = ResidualBlock(channel*16)#256
        
		self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
		self.conv_eltem = nn.Conv2d(channel,2*channel,kernel_size=1,stride=1,padding=0,bias=False)#16 32
		self.conv_emtes = nn.Conv2d(2*channel,4*channel,kernel_size=1,stride=1,padding=0,bias=False)#32 64
		self.conv_estess = nn.Conv2d(4*channel,8*channel,kernel_size=1,stride=1,padding=0,bias=False)#64 128
		self.conv_esstesss = nn.Conv2d(8*channel,16*channel,kernel_size=1,stride=1,padding=0,bias=False)#128 256
        
	def forward(self,x):
        
		elout = self.el(x)#16
		x_emin = self.conv_eltem(self.maxpool(elout))#32
		emout = self.em(x_emin)
		x_esin = self.conv_emtes(self.maxpool(emout))        
		esout = self.es(x_esin)
		x_esin = self.conv_estess(self.maxpool(esout))        
		essout = self.ess(x_esin)#128

		return elout,emout,esout,essout#,esssout

class ShareNet(nn.Module):
	def __init__(self,channel):
		super(ShareNet,self).__init__()    

		self.s1 = Dense(channel*8)#128
		self.s2 = Dense(channel*8)#128

	def forward(self,x):

		share1 = self.s1(x)
		share2 = self.s2(share1+x)

		return share2

class Decoder(nn.Module):
	def __init__(self,channel):
		super(Decoder,self).__init__()    

		self.dss = ResidualBlock(channel*8)#128
		self.ds = ResidualBlock(channel*4)#64
		self.dm = ResidualBlock(channel*2)#32
		self.dl = ResidualBlock(channel)#16
        
		self.conv_dssstdss = nn.Conv2d(16*channel,8*channel,kernel_size=1,stride=1,padding=0,bias=False)#256 128
		self.conv_dsstds = nn.Conv2d(8*channel,4*channel,kernel_size=1,stride=1,padding=0,bias=False)#128 64
		self.conv_dstdm = nn.Conv2d(4*channel,2*channel,kernel_size=1,stride=1,padding=0,bias=False)#64 32
		self.conv_dmtdl = nn.Conv2d(2*channel,channel,kernel_size=1,stride=1,padding=0,bias=False)#32 16
        
	def _upsample(self,x):
		_,_,H,W = x.size()
		return F.upsample(x,size=(2*H,2*W),mode='bilinear')
    
	def forward(self,x,ss,s,m,l):

		dssout = self.dss(x+ss)
		x_dsin = self.conv_dsstds(self._upsample(dssout))        
		dsout = self.ds(x_dsin+s)
		x_dmin = self.conv_dstdm(self._upsample(dsout))
		dmout = self.dm(x_dmin+m)
		x_dlin = self.conv_dmtdl(self._upsample(dmout))
		dlout = self.dl(x_dlin+l)
        
		return dlout

class Dense(nn.Module):
	def __init__(self,channel):
		super(Dense,self).__init__()
				   
		self.conv_32_32_1 = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False)#16 16
		self.conv_32_32_2 = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False)#16 16
		self.conv_64_32 = nn.Conv2d(channel*2,channel,kernel_size=3,stride=1,padding=1,bias=False)#32 16
		self.conv_96_32 = nn.Conv2d(channel*3,channel,kernel_size=3,stride=1,padding=1,bias=False)#48 16
		self.conv_128_32 = nn.Conv2d(channel*4,channel,kernel_size=3,stride=1,padding=1,bias=False)#64 16
		self.relu = nn.ReLU(inplace=True) #节省计算量  避免梯度消失 缓解过拟
		self.norm = nn.GroupNorm(num_channels=channel,num_groups=1) #nn.LocalResponseNorm(32,alpha = 0.0001,beta = 0.75, k=1.0)
        #将channel切分成许多组进行归一化

	def forward(self,x_in):
		x1_2 = self.relu(self.norm(self.conv_32_32_1(x_in)))  
		x1_3 = self.relu(self.norm(self.conv_32_32_2(x1_2)))  	   
		x1_4 = self.relu(self.norm(self.conv_64_32(torch.cat((x1_2,x1_3),1))))  
		x1_5 = self.relu(self.norm(self.conv_96_32(torch.cat((x1_2,x1_3,x1_4),1))))  
		x1_6 = self.relu(self.norm(self.conv_128_32(torch.cat((x1_2,x1_3,x1_4,x1_5),1))))   
		
		return x1_6    
    
class ResidualBlock(nn.Module):# Edge-oriented Residual Convolution Block 面向边缘的残差网络块 解决梯度消失的问题
	def __init__(self,channel,norm=False):                                
		super(ResidualBlock,self).__init__()

		self.conv_1_1 = nn.Conv2d(channel,  channel,kernel_size=3,stride=1,padding=1,bias=False)#16 16
		self.conv_2_1 = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False)#16 16
		self.conv_out = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False)#16 16
		self.act = nn.PReLU(channel)#16
		self.norm =nn.GroupNorm(num_channels=channel,num_groups=1)# nn.InstanceNorm2d(channel)#
   
	def _upsample(self,x,y): #上采样->放大图片
		_,_,H,W = y.size()
		return F.upsample(x,size=(H,W),mode='bilinear')


	def forward(self,x):
        
		x_1 = self.act(self.norm(self.conv_1_1(x)))
		x_2 = self.act(self.norm(self.conv_2_1(x_1)))
		x_out = self.act(self.norm(self.conv_out(x_2)) + x)


		return	x_out        
    
