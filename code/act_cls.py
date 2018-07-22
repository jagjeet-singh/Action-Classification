import pickle
import pdb
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from termcolor import cprint
import math
import shutil
import os
from logger import Logger
from utils import *
import sklearn.metrics
from termcolor import cprint
from random import shuffle

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
global_step = 0
best_acc = 0

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-epoch',type=int, default=None,help="Starting epoch")
    parser.add_argument('--end-epoch',type=int, default=200,help="Last epoch")
    parser.add_argument('--lr',type=float, default=0.0001,help="Learning rate")
    parser.add_argument('--print-freq',type=int, default=10,help="Print frequency")
    parser.add_argument('--valid-freq',type=int, default=1,help="Validation frequency")
    parser.add_argument('--plot-freq',type=int, default=100,help="Frequency of plotting on Tensorboard")
    parser.add_argument('--batch-size',type=int, default=32,help="Training batch size")
    parser.add_argument('--workers',type=int, default=4,help="Number of workers")
    parser.add_argument('--train-val-split',type=float, default=0.7,help="Training validation split")
    parser.add_argument('--network',type=str, default='mlp',help="mlp or rnn?")
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument('--test', action='store_true', help='Test the trained model')
    return parser.parse_args()

class MLPNetwork(nn.Module):

	def __init__(self):
		super(MLPNetwork, self).__init__()
		self.linear1 = nn.Linear(512, 256)
		self.linear2 = nn.Linear(256, 256)
		self.linear3 = nn.Linear(256, 51)

	def forward(self, x, batch_size):
		x = F.tanh(self.linear1(x))
		x = F.tanh(self.linear2(x))
		x = self.linear3(x)
		return x

	def initialize(self):
		nn.init.xavier_normal(self.linear1.weight.data)
		nn.init.xavier_normal(self.linear2.weight.data)
		nn.init.xavier_normal(self.linear3.weight.data)

class RNNNetwork(nn.Module):

	def __init__(self, hidden_size):
		super(RNNNetwork, self).__init__()
		self.hidden_size = hidden_size
		self.lstm1 = nn.LSTM(input_size=512, hidden_size=self.hidden_size[0], 
			num_layers=1, batch_first=True, bidirectional=False)
		self.lstm2 = nn.LSTM(input_size=self.hidden_size[0], hidden_size=self.hidden_size[1], 
			num_layers=2, batch_first=True, bidirectional=False)
		self.lstm3 = nn.LSTM(input_size=self.hidden_size[1], hidden_size=self.hidden_size[2], 
			num_layers=1, batch_first=True, bidirectional=False)
		self.linear1 = nn.Linear(512, 256)
		self.linear3 = nn.Linear(256, 51)

	def forward(self, x, batch_size):
		self.batch_size = batch_size
		self.hidden_init = self.init_hidden()
		x = self.linear1(x)
		out,_ = self.lstm2(x, self.hidden_init[1])
		# out,_ = self.lstm3(out, self.hidden_init[2])
		out = self.linear3(out)
		return out

	def init_hidden(self):
		return [(Variable(torch.zeros((1, self.batch_size,self.hidden_size[0]))),
			Variable(torch.zeros((1, self.batch_size,self.hidden_size[0])))),
		(Variable(torch.zeros((2, self.batch_size,self.hidden_size[1]))),
			Variable(torch.zeros((2, self.batch_size,self.hidden_size[1])))),
		(Variable(torch.zeros((1, self.batch_size,self.hidden_size[2]))),
			Variable(torch.zeros((1, self.batch_size,self.hidden_size[2]))))]

def loadData():

	with open('data/annotated_train_set.p', 'rb') as f:
	    train = pickle.load(f, encoding='latin')

	with open('data/randomized_annotated_test_set_no_name_no_num.p', 'rb') as f:
	    test = pickle.load(f, encoding='latin')
	    
	return train, test


def validate(val_loader, model, epoch, logger, optimizer, test_size):

	args = parse_arguments()

	global best_acc
	for i, (input, target) in enumerate(val_loader):
		
		input_var = Variable(input)
		target_var = Variable(target)
		output = model(input_var, test_size)
		output = torch.mean(output, 1)
		prob, pred  = torch.max(output.data, 1)
		print(pred, target)
		acc = sklearn.metrics.accuracy_score(target, pred)
		cprint("Epoch:{}, Accuracy:{}".format(epoch, acc), color='green')	
		logger.scalar_summary(tag='Valid/accuracy', value=acc, step=epoch)
		if acc>best_acc:
			best_acc = acc
			save_checkpoint({
				'epoch': epoch + 1,
				'state_dict': model.state_dict(),
				'best_acc': best_acc,
				'optimizer' :  optimizer.state_dict()
			})

def test(test_loader, model, logger, optimizer, test_size):

	args = parse_arguments()
	pred_list = np.zeros((test_size,))
	for i, input in enumerate(test_loader):
		
		input_var = Variable(input)
		output = model(input_var, 1)
		output = torch.mean(output, 1)
		prob, pred  = torch.max(output.data, 1)
		print(pred.numpy()[0])
		pred_list[i] = int(pred.numpy()[0])
	np.savetxt(args.network+'_test.txt',pred_list.astype(int), fmt='%d')
		

def train(train_loader, model, epoch, criterion, optimizer, logger, network):

	args = parse_arguments()
	global global_step
	for i, (input, target) in enumerate(train_loader):
		input_var = Variable(input)
		target_var = Variable(target)
		output = model(input_var, args.batch_size) #batchx10x51
		output = torch.mean(output, 1)
		loss = criterion(output, target_var)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if global_step % args.plot_freq == 0:
			logger.scalar_summary(tag='Train/Loss', value=loss.data[0], step=global_step)
			logger.model_param_histo_summary(model=model, step=global_step)		

		if i % args.print_freq==0:
			print("Epoch:{}, iter:{}, loss:{}".format(epoch, i, loss.data[0]))
		global_step+=1

def save_checkpoint(state):
	args = parse_arguments()
	filename = 'saved_models/'+args.network+'-model-best.pth.tar'
	torch.save(state, filename)

def load_checkpoint(checkpoint_file, model, optimizer=None):
    args = parse_arguments()
    if os.path.isfile(checkpoint_file):
        print("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(checkpoint_file, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint_file))

    return start_epoch, best_acc

def main():

	args = parse_arguments()

	trainval_dict, test_dict = loadData()
	trainval_size = len(trainval_dict['data'])
	test_size = len(test_dict['data'])

	trainval_list = trainval_dict['data']
	shuffle(trainval_list)
	train_list = trainval_list[:int(0.7*trainval_size)]
	val_list = trainval_list[int(0.7*trainval_size):]
	val_size = len(val_list)
	test_list = test_dict['data']


	if args.network == 'mlp':
		model = MLPNetwork()

	elif args.network == 'rnn':
		model = RNNNetwork(hidden_size=[256, 256, 51])

	train_dataset = ActionClsDataset(train_list, type='train')
	val_dataset = ActionClsDataset(val_list, type='val')
	test_dataset = ActionClsDataset(test_list, type='test')

	# Setting Dataloaders
	train_loader = torch.utils.data.DataLoader(
		train_dataset, batch_size=args.batch_size, shuffle=True, 
		num_workers=args.workers, pin_memory=True, drop_last=True)

	val_loader = torch.utils.data.DataLoader(
		val_dataset,num_workers=args.workers, batch_size=val_size,
		pin_memory=True, drop_last=True, shuffle=True)

	test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=1)

	logger = Logger('hw3_logs', name=args.network)

	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

	start_epoch = 0
	if args.resume:
		start_epoch, best_acc = load_checkpoint(args.resume, model,  optimizer)
	if args.start_epoch:
		start_epoch=args.start_epoch

	if args.test:

		test(test_loader, model, logger, optimizer, test_size)

	else:
		for epoch in range(start_epoch, args.end_epoch):

			train(train_loader, model, epoch, criterion, optimizer, logger, network=args.network)

			if epoch % args.valid_freq == 0:

				validate(val_loader, model, epoch, logger, optimizer, val_size)

if __name__ == '__main__':
    main()