from __future__ import print_function, division

import os
import argparse
import errno
import shutil

import torch
import numpy as np
import torchvision

from torchvision import datasets, models, transforms
from networks import *
from torch.autograd import Variable
import util


parser = argparse.ArgumentParser(description='PyTorch script for chest xray view classification')
parser.add_argument('-o', '--output', help='output folder where classified images are stored', required=True)
parser.add_argument('-i', '--input', help='input folder where chest x-rays are stored', required=True)

args = parser.parse_args()

def mkdir_p(path):
#function  by @tzot from stackoverflow
	try:
		os.makedirs(path)
	except OSError as exc:  # Python >2.5
		if exc.errno == errno.EEXIST and os.path.isdir(path):
			pass
		else:
			raise

def softmax(x):
	return np.exp(x) / np.sum(np.exp(x), axis=0)






root_front = os.path.join(args.output, 'front')
root_side = os.path.join(args.output, 'side')

mkdir_p(root_front)
mkdir_p(root_side)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

views = {0: 'front',
		 1:	'side'}


data_transforms = {
	'test': transforms.Compose([
		transforms.Scale(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean, std)
	]),
}



dataset_dir = args.input


print("| Loading chestViewNet for chest x-ray view classification...")
checkpoint = torch.load('./models/'+'resnet-50.t7')
model = checkpoint['model']

use_gpu = torch.cuda.is_available()
if use_gpu:
	model.cuda()


model.eval()

testsets = util.MyFolder(dataset_dir, data_transforms['test'])

testloader = torch.utils.data.DataLoader(
	testsets,
	batch_size = 1,
	shuffle = False,
	num_workers=1
)







print("\n| classifying %s..." %dataset_dir)
for batch_idx, (inputs, path) in enumerate(testloader):
	if use_gpu:
		inputs = inputs.cuda()
	inputs = Variable(inputs, volatile=True)
	outputs = model(inputs)

	softmax_res = softmax(outputs.data.cpu().numpy()[0])

	_, predicted = torch.max(outputs.data, 1)
	print('%s is %s view' % (path[0], views[predicted.cpu().numpy()[0]]))

	if predicted.cpu().numpy()[0] == 0:
		shutil.copy2(path[0], root_front)
	else:
		shutil.copy2(path[0], root_side)