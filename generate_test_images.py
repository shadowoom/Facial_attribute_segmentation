#Main file where semantic segmentation part is trained

import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import pickle
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from dataset_builder import HelenData
from network_model_encoder_decoder import encoder_decoder
from collections import defaultdict
from PIL import Image

parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    help='mini-batch size (default: 1)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-8, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--ngpu', default=1, type=int,
                    help='total number of gpus (default: 1)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')
parser.set_defaults(augment=True)

model_path = '/home/chen/project/face-parsing/Facial_attribute_segmentation/save/best.dict'
save_dir = '/home/chen/project/face-parsing/Facial_attribute_segmentation/save_im'

image_save_dict = {}

_PALETTE = [0, 0, 0,#background
           255, 255, 0,#face skin(excluding ears and neck)
           139, 76, 57,#left eyebrow
           139, 139, 139,#right eyebrow(in picture)
           0, 205, 0,#left eye
           0, 138, 0,#right eye
           154, 50, 205,#nose
           0, 0, 139,#upper lip
           255, 165, 0,#inner lip
           72, 118, 255,#lower lip
           255, 0, 0]#hair

def main():
	global args
	args = parser.parse_args()
	kwargs = {'num_workers': 1, 'pin_memory': True}
	
	#Dataset for train and test	
	transform = transforms.RandomHorizontalFlip()
	dataset_test = HelenData('/home/chen/project/face-parsing/Facial_attribute_segmentation/dataset/SmithCVPR2013_dataset_resized/','testing.txt', transform=transform)
	val_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=True, **kwargs)
	model = encoder_decoder()
	model = model.cuda()
	model.load_state_dict(torch.load(model_path))
	cudnn.benchmark = True
	# define loss function (criterion) and optimizer
	criterion = nn.CrossEntropyLoss().cuda()
	validate(val_loader, model, criterion)
		
		
def validate(val_loader, model, criterion):
	#batch_time = AverageMeter()
	#losses = AverageMeter()
	top1 = AverageMeter()
	

	#end = time.time()
	#labels_dict = defaultdict(list)
	for i, (name, input_img, target) in enumerate(val_loader):
		target = target.cuda(async=True)
		input_img = input_img.float()
		input_img = input_img.cuda()
		input_var = torch.autograd.Variable(input_img)
		target_var = torch.autograd.Variable(target)
		output = model(input_var)
		#labels_dict[name[0]].append(output)
		#loss = criterion(output, target_var)
		prec1 = accuracy(output.data, target, name, topk=(1,))[0]
		#losses.update(loss.item(), input.size(0))
		top1.update(prec1.item(), input_img.size(0))
		#batch_time.update(time.time() - end)
		#end = time.time()
	print('Validate * Prec@1 {top1.avg:.3f}'.format(top1=top1))
	generate_image()

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count =0

	def update(self, val, n=1):
		self.val = val
		self.sum += val*n
		self.count += n
		self.avg = self.sum / self.count


def accuracy(output, target, name, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)
	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	file_name = name[0].split('/')[-1].split('.')[0] + '.png'
	save_mat = pred.cpu().numpy()
	image_save_dict[file_name] = save_mat	
	correct = pred.eq(target.unsqueeze(0))
	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res

def get_label_image(probs, img_h, img_w):
	labels = probs[:img_h, :img_w]
	label_im = Image.fromarray(labels, "P")
	label_im.putpalette(_PALETTE)
	return label_im

def generate_image():
	for key in image_save_dict.keys():
		mat = image_save_dict[key]
		resized_mat = mat[0,0,:,:]
		print(resized_mat[100:120,100:120])
		label_im = Image.fromarray(resized_mat, "P")
		label_im.putpalette(_PALETTE)
		label_im.save(os.path.join(save_dir, key))

if __name__ == '__main__':
	main()

