'''
Code to evaluate Fixed Feature baseline on different validation benchmarks. 
It manually iterates over nesting list; loads corresponding models and evaluate them.
'''
import sys 
sys.path.append("../../") # adding root folder to the path

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
from imagenetv2_pytorch import ImageNetV2Dataset
from torch.utils.data import DataLoader
import pandas as pd
from argparse import ArgumentParser
from utils import *
import torchvision.models as models
import calibration_tools

parser=ArgumentParser()
parser.add_argument('--path', type=str, required=True, help="Path to the directory which contains the ckpt folder. Ckpt has training jsons as well.")
parser.add_argument('--tta', action='store_true')
parser.add_argument('--dataset', type=str, default='V1')

args = parser.parse_args()

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
elif args.dataset == 'sketch':
	print("Loading Imagenet-Sketch dataset")
	dataset = torchvision.datasets.ImageFolder('sketch/', transform=t)
elif args.dataset == 'R':
	print("Loading true Imagenet-R val set")
	dataset = torchvision.datasets.ImageFolder('imagenet-r_/', transform=t)
else:
	print("Loading true Imagenet 1K val set")
	dataset = torchvision.datasets.ImageFolder('val/', transform=t)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, num_workers=12, shuffle=False)

for ff in [2**i for i in range(3,12)]:
	print("Rep. Size: ", ff)
	args.feat_dim=ff
	model = resnet50(False)
	model.fc=FixedFeatureLayer(args.feat_dim, 1000)
	apply_blurpool(model)		
	model.load_state_dict(get_ckpt(args.path+f"sh=0_mh=0_nesting_start=3_fixed_feature={ff}/final_weights.pt")) 
	model = model.cuda()
	model.eval()

	_, top1_acc, top5_acc, total_time, num_images, m_score_dict, softmax_probs, gt = evaluate_model(
			model, dataloader, show_progress_bar=True, nesting_list=None, tta=args.tta, imagenetA= args.dataset == 'A', imagenetR=args.dataset == 'R')


	confidence, predictions = torch.max(softmax_probs, dim=-1)
	max_uncertainity=0
	worst_class=None
	sum_=0
	for k in m_score_dict:
		m_score_dict[k]= (m_score_dict[k].mean()).item()
		sum_+=m_score_dict[k]
		if m_score_dict[k]>max_uncertainity:
			worst_class=k
			max_uncertainity=m_score_dict[k]	

	tqdm.write('    Evaluated {} images'.format(num_images))
	tqdm.write('    Top-1 accuracy: {:.2f}%'.format(100.0 * top1_acc))
	tqdm.write('    Top-5 accuracy: {:.2f}%'.format(100.0 * top5_acc))
	tqdm.write('    Total time: {:.1f}  (average time per image: {:.2f} ms)'.format(total_time, 1000.0 * total_time / num_images))
