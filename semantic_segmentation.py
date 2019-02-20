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
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from dataset_builder import HelenData
from network_model_encoder_decoder import encoder_decoder
from collections import defaultdict


parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
parser.add_argument('--epochs', default=100, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
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

model_path = "./save/"

def main():
	global args
	args = parser.parse_args()
	kwargs = {'num_workers': 1, 'pin_memory': True}
	
	#Dataset for train and test	
	transform = transforms.RandomHorizontalFlip()
	dataset_train = HelenData('./dataset/SmithCVPR2013_dataset_resized/','exemplars.txt',transform=transform)
	train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, **kwargs)
	dataset_test = HelenData('./dataset/SmithCVPR2013_dataset_resized/','tuning.txt', transform=transform)
	val_loader = torch.utils.data.DataLoader(dataset_test, batch_size=2, shuffle=True, **kwargs)

	model = encoder_decoder()

	# get the number of model parameters
	print('Number of model parameters: {}'.format(
		sum([p.data.nelement() for p in model.parameters()])))

	model = model.cuda()

	# optionally resume from a checkpoint
	if args.resume:
		if os.path.isfile(args.resume):
			print("=> loading checkpoint '{}'".format(args.resume))	
			checkpoint = torch.load(args.resume)
			args.start_epoch = checkpoint['epoch']
			best_prec1 = checkpoint['best_prec1']
			model.load_state_dict(checkpoint['state_dict'])
			print("=> loaded checkpoint '{}' (epoch {})"
				.format(args.resume, checkpoint['epoch']))
		else:
			print("=> no checkpoint found at '{}'".format(args.resume))

	cudnn.benchmark = True
	# define loss function (criterion) and optimizer
	criterion = nn.CrossEntropyLoss().cuda()
	optimizer = torch.optim.SGD(model.parameters(), args.lr,
			momentum=args.momentum, nesterov = args.nesterov,
			weight_decay=args.weight_decay)

		
	best = 0
	no_improvement_time = 0
	for epoch in range(args.start_epoch, args.epochs):
		#train(train_loader, model, criterion, optimizer,epoch)
		#result = validate(val_loader, model, criterion, epoch)
		
		# training
		training_top1 = AverageMeter()
		training_losses = AverageMeter()
		training_batch_time = AverageMeter()
		model.train()
		end = time.time()
		for i, (name, input_img, target) in enumerate(train_loader):
			target = target.cuda(async=True)
			input_img = input_img.float().cuda()
			input_var = torch.autograd.Variable(input_img)
			target_var = torch.autograd.Variable(target)
			output = model(input_var)
			#output = output.permute(0,2,3,1)
			#output = torch.max(output,dim=3)[0]
			loss = criterion(output, target_var)

			prec1 = accuracy(output.data, target, topk=(1,))[0]
			training_losses.update(loss.item(), input_img.size(0))
			training_top1.update(prec1.item(), input_img.size(0))
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
	
			training_batch_time.update(time.time() - end)
			end = time.time()

			print('Epoch: [{0}][{1}/{2}]\t'
				'Time {training_batch_time.val:.3f} ({training_batch_time.avg:.3f})\t'
				'Loss {training_losses.val:.4f} ({training_losses.avg:.4f})\t'
				'Prec@1 {training_top1.val:.3f} ({training_top1.avg:.3f})'.format(
					epoch, i, len(train_loader), training_batch_time=training_batch_time,
					training_losses=training_losses, training_top1=training_top1))
		# evaluation

		eval_top1 = AverageMeter()
		eval_losses = AverageMeter()
		model_eval = encoder_decoder().cuda()
		best = 0
		no_improvement = 0
		#model.eval()
		model_eval.load_state_dict(model.state_dict())
		for i, (name, input_img, target) in enumerate(val_loader):
			target = target.cuda(async=True)
			input_img = input_img.float().cuda()
			input_var = torch.autograd.Variable(input_img)
			target_var = torch.autograd.Variable(target)
			output = model_eval(input_var)
			loss = criterion(output, target_var)	
			prec1 = accuracy(output.data, target, topk=(1,))[0]
			eval_losses.update(loss.item(), input_img.size(0))
			eval_top1.update(prec1.item(), input_img.size(0))
			optimizer.step()
		print('Validate * Prec@1 {eval_top1.avg:.3f}'.format(eval_top1=eval_top1))
			#print('Epoch: [{0}][{1}/{2}]\t'
                        #       'Loss {training_losses.val:.4f} ({training_losses.avg:.4f})\t'
                        #       'Prec@1 {training_top1.val:.3f} ({training_top1.avg:.3f})'.format(
                        #               epoch, i, len(train_loader),
                        #               training_losses=eval_losses, training_top1=eval_top1))
		if epoch == 10:	
			torch.save(model.state_dict(), os.path.join(model_path, str(epoch)+'.dict'))
		if eval_top1.avg > best:
			best = eval_top1.avg
			torch.save(model.state_dict(), os.path.join(model_path, 'best.dict'))
		else:
			no_improvement_time += 1
		if no_improvement_time == 10:
			break
		

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

def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)
	_, pred = output.topk(maxk, 1, True, True)
#	pred = pred.t()
#	print(pred[0,0,100:105,100:105])
	#correct = pred.eq(target.view(1, -1).expand_as(pred))
	correct = pred.eq(target.unsqueeze(0))
	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0)
		#total_size = list(correct[:k].view(-1).size())[0]
		#print(correct_k.item())
		#print(total_size)
		#print(batch_size)
	        #res.append(correct_k.item() * 100.0 / batch_size / total_size)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res

if __name__ == '__main__':
	main()

