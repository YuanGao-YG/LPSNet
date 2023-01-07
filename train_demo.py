# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 20:48:05 2020

@author: Administrator
"""

import os, time, scipy.io, shutil
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
#from tensorboardX import SummaryWriter
import numpy as np
import cv2
import scipy.misc
from LPSLE_Model import *
from makedataset import Dataset
import utils_train
from Test_SSIM import *
from Syn_Type import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_checkpoint(checkpoint_dir):
	if os.path.exists(checkpoint_dir + 'checkpoint.pth.tar'):
		model_info = torch.load(checkpoint_dir + 'checkpoint.pth.tar')
		print('loading existing model ......', checkpoint_dir + 'checkpoint.pth.tar')
		net = LPSLE()
		device_ids = [0]
		model = nn.DataParallel(net, device_ids=device_ids).to(device)
		model.load_state_dict(model_info['state_dict'])
		optimizer = torch.optim.Adam(model.parameters())
		optimizer.load_state_dict(model_info['optimizer'])
		cur_epoch = model_info['epoch']
		
	else:
		net = LPSLE()
		device_ids = [0]
		model = nn.DataParallel(net, device_ids=device_ids).to(device)
		optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
		cur_epoch = 0
		
	return model, optimizer,cur_epoch


def save_checkpoint(state, is_best, PSNR,SSIM,filename='checkpoint.pth.tar'):
	torch.save(state, checkpoint_dir + 'PSNR_%.4f_SSIM_%.4f_'%(PSNR,SSIM) + 'checkpoint.pth.tar')
	if is_best:
		shutil.copyfile(checkpoint_dir + 'checkpoint.pth.tar',checkpoint_dir + 'model_best.pth.tar')


        
def adjust_learning_rate(optimizer, epoch, lr_update_freq):
	if not epoch % lr_update_freq and epoch:
		for param_group in optimizer.param_groups:
			param_group['lr'] = param_group['lr'] * 0.1
			print( param_group['lr'])
	return optimizer

def train_psnr(train_in,train_out):
	
	psnr = utils_train.batch_psnr(train_in,train_out,1.)
	return psnr


def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1])

def chw_to_hwc(img):
    return np.transpose(img, axes=[1, 2, 0])
	

if __name__ == '__main__': 	
	checkpoint_dir = './checkpoint/'
	test_dir1 = './dataset/Test_Lowlight'
	test_dir2 = './dataset/Test_Hazey'    
	result_dir = './result'
    
	testfiles1 = os.listdir(test_dir1)
	testfiles2 = os.listdir(test_dir2)
    
	maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
    
	print('> Loading dataset ...')
	dataset = Dataset(trainrgb=True, trainsyn=True, shuffle=True)
	loader_dataset = DataLoader(dataset=dataset, num_workers=0, batch_size=16, shuffle=False)
	count = len(loader_dataset)
	
	lr_update_freq = 20
	model,optimizer,cur_epoch = load_checkpoint(checkpoint_dir)
    
	L1_loss = torch.nn.L1Loss(reduce=True, size_average=True).to(device)
	L2_loss = torch.nn.MSELoss(reduce=True, size_average=True).to(device)
	
	for epoch in range(cur_epoch,200):
		optimizer = adjust_learning_rate(optimizer, epoch, lr_update_freq)
		learnrate = optimizer.param_groups[-1]['lr']
		model.train()

        
		aaa = 0
		for i,data in enumerate(loader_dataset,0):

			img_c = torch.zeros(data[:,0:3,:,:].size())		
			img_l = torch.zeros(data[:,0:3,:,:].size())

			for nx in range(data.shape[0]):             
				img_c[nx,:,:,:] = data[nx,0:3,:,:]
                
			sor = np.random.uniform(0,1)                   
			if sor<=0.5:
				Type = 1
				for nxx in range(data.shape[0]):
					img_l[nxx] = data[nxx,3:6,:,:]
                
			else:
				Type = 2
				for nxx in range(data.shape[0]):
					img_l[nxx] = Hazey(data[nxx,0:3,:,:])

									
			input_var = Variable(img_l.to(device), volatile=True)
			target_final = Variable(img_c.to(device), volatile=True)

			eout = model(input_var,Type)

			enloss = 0.5*L2_loss(eout,target_final) + 0.5*L1_loss(eout,target_final)
			optimizer.zero_grad()
			#Doptimizer.zero_grad()
            
			enloss.backward()
            
			optimizer.step()
			#Doptimizer.step()
            
			SN1_psnr = train_psnr(target_final,eout)		           
			print("[Epoch %d][Type--%d--][%d/%d] lr :%f loss: %.4f PSNR_train: %.4f" %(epoch+1,Type, i+1, count, learnrate, enloss.item(), SN1_psnr))
			
		for f in range(len(testfiles1)):
			model.eval()
			with torch.no_grad():
				img = cv2.imread(test_dir1 + '/' + testfiles1[f])
				h,w,c = img.shape
				img_ccc = cv2.resize(img,(512,512)) / 255.0
				img_h = hwc_to_chw(img_ccc)
				input_var = torch.from_numpy(img_h.copy()).type(torch.FloatTensor).unsqueeze(0).to(device)
				s = time.time()
				e_out = model(input_var,Type = 1)              
				e = time.time()   
				print(input_var.shape)       
				print(e-s)    
	             

				e_out = e_out.squeeze().cpu().detach().numpy()			               
				e_out = chw_to_hwc(e_out) 
			              
				e_out = cv2.resize(e_out,(w,h))				
				cv2.imwrite(result_dir + '/' + testfiles1[f][:-4] +'_%d_9'%(epoch)+'.png',np.clip(e_out*255,0.0,255.0))

		for f in range(len(testfiles2)):
			model.eval()
			with torch.no_grad():
				img = cv2.imread(test_dir2 + '/' + testfiles2[f])
				h,w,c = img.shape
				img_ccc = cv2.resize(img,(512,512)) / 255.0
				img_h = hwc_to_chw(img_ccc)
				input_var = torch.from_numpy(img_h.copy()).type(torch.FloatTensor).unsqueeze(0).to(device)
				s = time.time()
				e_out = model(input_var,Type=2)              
				e = time.time()   
				print(input_var.shape)       
				print(e-s)    
	             
				e_out = e_out.squeeze().cpu().detach().numpy()			               
				e_out = chw_to_hwc(e_out) 
                
				e_out = cv2.resize(e_out,(w,h))			              
			
				cv2.imwrite(result_dir + '/' + testfiles2[f][:-4] +'_%d_9'%(epoch)+'.png',np.clip(e_out*255,0.0,255.0))
                
		save_checkpoint({
			'epoch': epoch + 1,
			'state_dict': model.state_dict(),
			'optimizer' : optimizer.state_dict()}, is_best=0,PSNR=0,SSIM=0)
			
			

