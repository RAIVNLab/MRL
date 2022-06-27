'''
Code to evaluate MRL models on different validation benchmarks. 
'''
import sys 
sys.path.append("../") # adding root folder to the path


import torch 
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import *
from torchvision import datasets
from tqdm import tqdm
from timeit import default_timer as timer
import math
import numpy as np
from MRL import *
from imagenetv2_pytorch import ImageNetV2Dataset
from torch.utils.data import DataLoader
import pandas as pd
from argparse import ArgumentParser
from utils import *
import calibration_tools
import sys

# nesting list is by default from 8 to 2048 in powers of 2, can be modified from here. 
NESTING_LIST=[2**i for i in range(3, 12)] 

parser=ArgumentParser()

# model args
parser.add_argument('--efficient', action='store_true', help='Efficient Flag')
parser.add_argument('--mrl', action='store_true', help='To use MRL')
parser.add_argument('--rep_size', default=None, help='This needs to be used in case we want to evaluate fixed feature baseline, for a particular rep. size')
parser.add_argument('--path', type=str, required=True, help='Path to .pt checkpoint')

parser.add_argument('--old_ckpt', action='store_true', help='To use our checkpoints')

# dataset/eval args
parser.add_argument('--tta', action='store_true', help='Test Time Augmentation Flag')
parser.add_argument('--dataset', type=str, default='V1', help='Benchmarks')

args = parser.parse_args()


model = resnet50(False)
if not args.old_ckpt:
	if args.mrl:
		model.fc = MRL_Linear_Layer(NESTING_LIST, efficient=args.efficient)	
	else:
		model.fc=FixedFeatureLayer(args.rep_size, 1000)
else:
	if args.mrl:	
		model = load_from_old_ckpt(model, args.efficient, NESTING_LIST)
	else:
		model.fc=FixedFeatureLayer(args.rep_size, 1000)	


apply_blurpool(model)	
model.load_state_dict(get_ckpt(args.path)) # Since our models have a torch DDP wrapper, we modify keys to exclude first 7 chars. 
model = model.cuda()
model.eval()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
resize_size = 256
center_crop_size = 224
t = transforms.Compose([
				transforms.Resize(resize_size),
				transforms.CenterCrop(center_crop_size),
				transforms.ToTensor(),
				normalize])

if args.dataset == 'V2':
	print("Loading Robustness Dataset")
	dataset = ImageNetV2Dataset("matched-frequency", transform=t)
elif args.dataset == 'A':
	print("Loading true Imagenet-A val set")
	dataset = torchvision.datasets.ImageFolder('imagenet-a/', transform=t)
elif args.dataset == 'R':
	print("Loading true Imagenet-R val set")
	dataset = torchvision.datasets.ImageFolder('imagenet-r_/', transform=t)
elif args.dataset == 'sketch':
	print("Loading Imagenet-Sketch dataset")
	dataset = torchvision.datasets.ImageFolder('sketch/', transform=t)
else:
	print("Loading true Imagenet 1K val set")
	dataset = torchvision.datasets.ImageFolder('val/', transform=t)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, num_workers=12, shuffle=False)

if args.mrl:
	_, top1_acc, top5_acc, total_time, num_images, m_score_dict, softmax_probs, gt, logits = evaluate_model(
			model, dataloader, show_progress_bar=True, nesting_list=NESTING_LIST, tta=args.tta, imagenetA= args.dataset == 'A', imagenetR=args.dataset == 'R')
else:
	_, top1_acc, top5_acc, total_time, num_images, m_score_dict, softmax_probs, gt = evaluate_model(
			model, dataloader, show_progress_bar=True, nesting_list=None, tta=args.tta, imagenetA= args.dataset == 'A', imagenetR=args.dataset == 'R')

tqdm.write('Evaluated {} images'.format(num_images))
confidence, predictions = torch.max(softmax_probs, dim=-1)
if args.mrl:
	for i, nesting in enumerate(NESTING_LIST):
		print("Rep. Size", "\t", nesting, "\n")
		tqdm.write('    Top-1 accuracy for {} : {:.2f}'.format(nesting, 100.0 * top1_acc[nesting]))
		tqdm.write('    Top-5 accuracy for {} : {:.2f}'.format(nesting, 100.0 * top5_acc[nesting]))
		tqdm.write('    Total time: {:.1f}  (average time per image: {:.2f} ms)'.format(total_time, 1000.0 * total_time / num_images))
else:
	print("Rep. Size", "\t", args.rep_size, "\n")
	tqdm.write('    Evaluated {} images'.format(num_images))
	tqdm.write('    Top-1 accuracy: {:.2f}%'.format(100.0 * top1_acc))
	tqdm.write('    Top-5 accuracy: {:.2f}%'.format(100.0 * top5_acc))
	tqdm.write('    Total time: {:.1f}  (average time per image: {:.2f} ms)'.format(total_time, 1000.0 * total_time / num_images))