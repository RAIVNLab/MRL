import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from typing import Type, Any, Callable, Union, List, Optional
from torchvision.models import *
from tqdm import tqdm
from timeit import default_timer as timer
import math
import numpy as np
from imagenet_id import indices_in_1k_a, indices_in_1k_o, indices_in_1k_r


def get_ckpt(path):
	ckpt=path 
	ckpt = torch.load(ckpt, map_location='cpu')
	plain_ckpt={}
	for k in ckpt.keys():
		plain_ckpt[k[7:]] = ckpt[k] # torch DDP models have a extra wrapper of module., so to remove that... 
	return plain_ckpt


class BlurPoolConv2d(torch.nn.Module):
	def __init__(self, conv):
		super().__init__()
		default_filter = torch.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]) / 16.0
		filt = default_filter.repeat(conv.in_channels, 1, 1, 1)
		self.conv = conv
		self.register_buffer('blur_filter', filt)

	def forward(self, x):
		blurred = F.conv2d(x, self.blur_filter, stride=1, padding=(1, 1),
						   groups=self.conv.in_channels, bias=None)
		return self.conv.forward(blurred)


def apply_blurpool(mod: torch.nn.Module):
	for (name, child) in mod.named_children():
		if isinstance(child, torch.nn.Conv2d) and (np.max(child.stride) > 1 and child.in_channels >= 16):
			setattr(mod, name, BlurPoolConv2d(child))
		else: apply_blurpool(child)


def evaluate_model(model, dataloader, show_progress_bar=True, notebook_progress_bar=False, nesting_list=None, tta=False, imagenetA=False, imagenetO=False, imagenetR=False):
	if nesting_list is None:
		return evaluate_model_ff(model, dataloader, show_progress_bar, notebook_progress_bar, tta=tta, imagenetA=imagenetA, imagenetO=imagenetO, imagenetR=imagenetR)
	else:
		return evaluate_model_nesting(model, dataloader, show_progress_bar=True, nesting_list=nesting_list, tta=tta, imagenetA=imagenetA, imagenetO=imagenetO, imagenetR=imagenetR)


def evaluate_model_ff(model, data_loader, show_progress_bar=False, notebook_progress_bar=False, tta=False, imagenetA=False, imagenetO=False, imagenetR=False):

	torch.backends.cudnn.benchmark = True
	num_images = 0
	num_top1_correct = 0
	num_top5_correct = 0
	predictions = []; m_score_dict={}; softmax=[]; gt=[]; all_logits=[]
	start = timer()
	with torch.no_grad():
		enumerable = enumerate(data_loader)
		if show_progress_bar:
			total = int(math.ceil(len(data_loader.dataset) / data_loader.batch_size))
			desc = 'Batch'
			if notebook_progress_bar:
				enumerable = tqdm.tqdm_notebook(enumerable, total=total, desc=desc)
			else:
				enumerable = tqdm(enumerable, total=total, desc=desc)
		for ii, (img_input, target) in enumerable:
			gt.append(target)
			unique_labels= torch.unique(target)
			img_input = img_input.cuda(non_blocking=True)
			logits = model(img_input)
			if tta:
				logits+= model(torch.flip(img_input, dims=[3]))
			# Getting the margin scores...
			if imagenetA:
				logits = logits[:, indices_in_1k_a]
			elif imagenetO:
				logits = logits[:, indices_in_1k_o]
			elif imagenetR:
				logits = logits[:, indices_in_1k_r]

			probs=F.softmax(logits, dim=-1); softmax.append(probs)
			m_score = margin_score(logits)
			for y in unique_labels:
				y=y.item()
				m_ = m_score[target==y]
				if not (y in m_score_dict.keys()):
					m_score_dict[y]=[]
				m_score_dict[y].append(m_)

			_, output_index = logits.topk(k=5, dim=1, largest=True, sorted=True)
			output_index = output_index.cpu().numpy()
			predictions.append(output_index)
			for jj, correct_class in enumerate(target.cpu().numpy()):
				if correct_class == output_index[jj, 0]:
					num_top1_correct += 1
				if correct_class in output_index[jj, :]:
					num_top5_correct += 1
			num_images += len(target)
			all_logits.append(logits.cpu())
	end = timer()
	predictions = np.vstack(predictions)
	for k in m_score_dict.keys():
		m_score_dict[k]=torch.cat(m_score_dict[k])

	assert predictions.shape == (num_images, 5)
	return predictions, num_top1_correct / num_images, num_top5_correct / num_images, end - start, num_images, m_score_dict, torch.cat(softmax, dim=0), torch.cat(gt, dim=0), torch.cat(all_logits, dim=0)


def evaluate_model_nesting(model, data_loader, show_progress_bar=False, notebook_progress_bar=False, nesting_list=[2**i for i in range(3, 12)], tta=False, imagenetA= False, imagenetO=False, imagenetR=False):
	torch.backends.cudnn.benchmark = True

	num_images = 0
	num_top1_correct = {}
	num_top5_correct = {}
	predictions = {}; m_score_dict={};softmax=[]; gt=[]; all_logits=[]
	for i in nesting_list:
		m_score_dict[i]={}
		predictions[i]=[]
		num_top5_correct[i], num_top1_correct[i]=0,0
	start = timer()
	with torch.no_grad():
		enumerable = enumerate(data_loader)
		if show_progress_bar:
			total = int(math.ceil(len(data_loader.dataset) / data_loader.batch_size))
			desc = 'Batch'
			if notebook_progress_bar:
				enumerable = tqdm.tqdm_notebook(enumerable, total=total, desc=desc)
			else:
				enumerable = tqdm(enumerable, total=total, desc=desc)
		for ii, (img_input, target) in enumerable:
			gt.append(target)
			unique_labels= torch.unique(target)
			img_input = img_input.cuda(non_blocking=True)
			logits = model(img_input); logits=torch.stack(logits, dim=0)
			if tta:
				logits+= torch.stack(model(torch.flip(img_input, dims=[3])), dim=0)
			# We have many logits here.... 
			# Getting the margin scores...
			if imagenetA:
				logits = logits[:, :, indices_in_1k_a]
			elif imagenetO:
				logits = logits[:, :, indices_in_1k_o]
			elif imagenetR:
				logits = logits[:, :, indices_in_1k_r]
			probs=F.softmax(logits, dim=-1); softmax.append(probs.cpu())

			m_score = margin_score(logits)
			for k, nesting in enumerate(nesting_list):
				for y in unique_labels:
					y=y.item()
					m_ = (m_score[k])[target==y]
					if not (y in m_score_dict[nesting].keys()):
						m_score_dict[nesting][y]=[]
					m_score_dict[nesting][y].append(m_)

				_, output_index = logits[k].topk(k=5, dim=1, largest=True, sorted=True)
				output_index = output_index.cpu().numpy()
				predictions[nesting].append(output_index)
				for jj, correct_class in enumerate(target.cpu().numpy()):
					if correct_class == output_index[jj, 0]:
						num_top1_correct[nesting] += 1
					if correct_class in output_index[jj, :]:
						num_top5_correct[nesting] += 1
			num_images += len(target)
			all_logits.append(logits.cpu())

	end = timer()
	for nesting in nesting_list:
		predictions[nesting] = np.vstack(predictions[nesting])
		for k in m_score_dict[nesting].keys():
			m_score_dict[nesting][k]=torch.cat(m_score_dict[nesting][k])
			m_score_dict[nesting][k]=(m_score_dict[nesting][k].mean()).item()

		num_top5_correct[nesting]=num_top5_correct[nesting]/num_images
		num_top1_correct[nesting]=num_top1_correct[nesting]/num_images

		assert predictions[nesting].shape == (num_images, 5)
	return predictions, num_top1_correct, num_top5_correct, end - start, num_images, m_score_dict,torch.cat(softmax, dim=1), torch.cat(gt, dim=0), torch.cat(all_logits, dim=1)


def margin_score(y_pred):
	top_2 = torch.topk(F.softmax(y_pred, dim=-1), k=2, dim=-1)[0]
	if len(top_2.shape)>2:
		margin_score = 1- (top_2[:, :, 0]-top_2[:, :, 1])
	else:
		margin_score = 1- (top_2[:, 0]-top_2[:, 1])
	return margin_score


'''
Retrieval utility methods
'''
activation = {}
fwd_pass_x_list = []
fwd_pass_y_list = []

def get_activation(name):
	"""
	Get the activation from an intermediate point in the network
	:param name: layer whose activation is to be returned
	:return: activation of layer
	"""
	def hook(model, input, output):
		activation[name] = output.detach()
	return hook


def append_feature_vector_to_list(activation, label, rep_size):
	"""
	Append the feature vector to a list to later write to disk
	:param activation: image feature vector from network
	:param label: ground truth label
	:param rep_size: representation size to be stored
	"""
	for i in range (activation.shape[0]):
		x = activation[i].cpu().detach().numpy()
		y = label[i].cpu().detach().numpy()
		fwd_pass_y_list.append(y)
		fwd_pass_x_list.append(x[:rep_size])


def dump_feature_vector_array_lists(config_name, random_sample_dim, output_path):
	"""
	Save the database and query vector array lists to disk
	:param config_name: config to specify during file write
	:param random_sample_dim: to write a subset of database if required, e.g. to train an SVM on 100K samples
	:param output_path: path to dump database and query arrays after inference
	"""

	# save X (n x 2048), y (n x 1) to disk, where n = num_samples
	args = parser.parse_args()
	X_fwd_pass = np.asarray(fwd_pass_x_list, dtype=np.float32)
	y_fwd_pass = np.asarray(fwd_pass_y_list, dtype=np.float16).reshape(-1,1)

	if random_sample_dim < X_fwd_pass.shape[0]:
		random_indices = np.random.choice(X_fwd_pass.shape[0], size=random_sample_dim, replace=False)
		random_X = X_fwd_pass[random_indices, :]
		random_y = y_fwd_pass[random_indices, :]
		print("Writing random samples to disk with dim [%d x 2048] " % random_sample_dim)
	else:
		random_X = X_fwd_pass
		random_y = y_fwd_pass
		print("Writing %s to disk with dim [%d x %d]" % (str(config_name), X_fwd_pass.shape[0], args.fixed_feature))

	np.save(output_path+str(config_name)+'-X.npy', random_X)
	np.save(output_path+str(config_name)+'-y.npy', random_y)


def generate_retrieval_data(model, data_loader, config, distributed, random_sample_dim, rep_size, output_path):
	"""
	Iterate over data in dataloader, get feature vector from model inference, and save to array to dump to disk
	:param model: ResNet50 model loaded from disk
	:param data_loader: loader for database or query set
	:param config: name of configuration for writing arrays to disk
	:param distributed: if model was trained with PyTorch DDP
	:param random_sample_dim: to write a subset of database if required, e.g. to train an SVM on 100K samples
	:param rep_size: representation size for fixed feature model
	:param output_path: path to dump database and query arrays after inference
	"""
	if distributed:
		model.module.avgpool.register_forward_hook(get_activation('avgpool'))
	else:
		model.avgpool.register_forward_hook(get_activation('avgpool'))

	with torch.no_grad():
		with autocast():
				for i_batch, images, target in enumerate(data_loader):
					output = model(images)
					append_feature_vector_to_list(activation['avgpool'].squeeze(), target, rep_size)
				dump_feature_vector_array_lists(config, random_sample_dim, output_path)

	# re-initialize empty lists
	global fwd_pass_x_list
	global fwd_pass_y_list
	fwd_pass_x_list = []

'''
Load pretrained models saved with old notation
'''
class SingleHeadNestedLinear(nn.Linear):
	"""
	Class for MRL-E model.
	"""

	def __init__(self, nesting_list: List, num_classes=1000, **kwargs):
		super(SingleHeadNestedLinear, self).__init__(nesting_list[-1], num_classes, **kwargs)
		self.nesting_list=nesting_list
		self.num_classes=num_classes # Number of classes for classification

	def forward(self, x):
		nesting_logits = ()
		for i, num_feat in enumerate(self.nesting_list):
			if not (self.bias is None):
				logit = torch.matmul(x[:, :num_feat], (self.weight[:, :num_feat]).t()) + self.bias
			else:
				logit = torch.matmul(x[:, :num_feat], (self.weight[:, :num_feat]).t())
			nesting_logits+= (logit,)
		return nesting_logits

class MultiHeadNestedLinear(nn.Module):
	"""
	Class for MRL model.
	"""
	def __init__(self, nesting_list: List, num_classes=1000, **kwargs):
		super(MultiHeadNestedLinear, self).__init__()
		self.nesting_list=nesting_list
		self.num_classes=num_classes # Number of classes for classification
		for i, num_feat in enumerate(self.nesting_list):
			setattr(self, f"nesting_classifier_{i}", nn.Linear(num_feat, self.num_classes, **kwargs))

	def forward(self, x):
		nesting_logits = ()
		for i, num_feat in enumerate(self.nesting_list):
			nesting_logits +=  (getattr(self, f"nesting_classifier_{i}")(x[:, :num_feat]),)
		return nesting_logits

def load_from_old_ckpt(model, efficient, nesting_list):
		if efficient:
			model.fc=SingleHeadNestedLinear(nesting_list)
		else:
			model.fc=MultiHeadNestedLinear(nesting_list)

		return model