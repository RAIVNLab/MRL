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
import sys
from MRL import *
from imagenetv2_pytorch import ImageNetV2Dataset
from torch.utils.data import DataLoader
import pandas as pd
from argparse import ArgumentParser
from utils import *
import calibration_tools
import sys

parser=ArgumentParser()
parser.add_argument('--efficient', action='store_true')
parser.add_argument('--path', type=str, required=True)
parser.add_argument('--tta', action='store_true')
parser.add_argument('--dataset', type=str, default='V1')


args = parser.parse_args()
model = resnet50(False)
nesting_list=[2**i for i in range(3, 12)] # BY DEFAULT STARTING FROM 8 DIM...
model.fc = MRL_Linear_Layer(nesting_list, efficient=args.efficient)
apply_blurpool(model)	
model.load_state_dict(get_ckpt(args.path)) # Since our models have a torch DDP wrapper. 
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
elif args.dataset == 'C':
	print("Not implemented Yet~"); sys.exit()
elif args.dataset == 'R':
	print("Loading true Imagenet-R val set")
	dataset = torchvision.datasets.ImageFolder('imagenet-r_/', transform=t)
elif args.dataset == 'sketch':
	print("Loading Imagenet-Sketch dataset")
	dataset = torchvision.datasets.ImageFolder('sketch/', transform=t)
elif args.dataset == 'O':
	print("Loading true Imagenet-O val set")
	dataset_o = torchvision.datasets.ImageFolder('imagenet-o/', transform=t)
	dataset_in = torchvision.datasets.ImageFolder('imagenet_val_for_imagenet_o_ood_/', transform=t)
else:
	print("Loading true Imagenet 1K val set")
	dataset = torchvision.datasets.ImageFolder('val/', transform=t)

if args.dataset=='O':
	dataloader = torch.utils.data.DataLoader(dataset_in, batch_size=128, num_workers=12, shuffle=False)
	_, top1_acc, top5_acc, total_time, num_images, m_score_dict, softmax_probs, gt = evaluate_model(
			model, dataloader, show_progress_bar=True, nesting_list=nesting_list, tta=args.tta, imagenetA= args.dataset == 'A', imagenetO=True)

	confidence_in, predictions_in = torch.max(softmax_probs, dim=-1)
	in_score = -confidence_in

	dataloader = torch.utils.data.DataLoader(dataset_o, batch_size=128, num_workers=12, shuffle=False)
	_, top1_acc, top5_acc, total_time, num_images, m_score_dict, softmax_probs, gt = evaluate_model(
			model, dataloader, show_progress_bar=True, nesting_list=nesting_list, tta=args.tta, imagenetA= args.dataset == 'A', imagenetO=True)

	confidence_out, predictions_out = torch.max(softmax_probs, dim=-1)
	out_score = -confidence_out

	for i, nesting in enumerate(nesting_list):
		print("Feature dim: ", nesting)
		aurocs, auprs, fprs = [], [], []
		measures = calibration_tools.get_measures(out_score[i].cpu().numpy(), in_score[i].cpu().numpy())
		aurocs = measures[0]; auprs = measures[1]; fprs = measures[2]
		calibration_tools.print_measures_old(aurocs, auprs, fprs, method_name='MSP')
else:
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, num_workers=12, shuffle=False)
	_, top1_acc, top5_acc, total_time, num_images, m_score_dict, softmax_probs, gt, logits = evaluate_model(
			model, dataloader, show_progress_bar=True, nesting_list=nesting_list, tta=args.tta, imagenetA= args.dataset == 'A', imagenetO=args.dataset == 'A', imagenetR=args.dataset == 'R')

	tqdm.write('    Evaluated {} images'.format(num_images))
	confidence, predictions = torch.max(softmax_probs, dim=-1)
	for i, nesting in enumerate(nesting_list):
		max_uncertainity=0
		worst_class=None

		for k in m_score_dict[nesting]:
			if m_score_dict[nesting][k]>max_uncertainity:
				worst_class=k
				max_uncertainity=m_score_dict[nesting][k]

		tqdm.write('    Top-1 accuracy for {} : {:.2f}'.format(nesting, 100.0 * top1_acc[nesting]))
		tqdm.write('    Top-5 accuracy for {} : {:.2f}'.format(nesting, 100.0 * top5_acc[nesting]))
		tqdm.write('    Total time: {:.1f}  (average time per image: {:.2f} ms)'.format(total_time, 1000.0 * total_time / num_images))

