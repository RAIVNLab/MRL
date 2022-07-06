'''
Code to evaluate MRL models on different validation benchmarks. 
'''
import sys 
sys.path.append("../") # adding root folder to the path

import torch 
import torchvision
from torchvision import transforms
from torchvision.models import *
from torchvision import datasets
from tqdm import tqdm

from MRL import *
from imagenetv2_pytorch import ImageNetV2Dataset
from argparse import ArgumentParser
from utils import *

# nesting list is by default from 8 to 2048 in powers of 2, can be modified from here.
BATCH_SIZE = 256
IMG_SIZE = 256
CENTER_CROP_SIZE = 224
NESTING_LIST=[2**i for i in range(3, 12)]
ROOT="../" # path to validation datasets

parser=ArgumentParser()

# model args
parser.add_argument('--efficient', action='store_true', help='Efficient Flag')
parser.add_argument('--mrl', action='store_true', help='To use MRL')
parser.add_argument('--rep_size', default=None, help='Rep. size for fixed feature model')
parser.add_argument('--path', type=str, required=True, help='Path to .pt checkpoint')
parser.add_argument('--old_ckpt', action='store_true', help='To use our checkpoints')
parser.add_argument('--workers', default=12, help='workers for dataloader', type=int)
parser.add_argument('--distributed', default=0, help='is model DistributedDataParallel')
# dataset/eval args
parser.add_argument('--tta', action='store_true', help='Test Time Augmentation Flag')
parser.add_argument('--dataset', type=str, default='V1', help='Benchmarks')
parser.add_argument('--save_logits', action='store_true', help='To save logits for model analysis')
parser.add_argument('--save_softmax', action='store_true', help='To save softmax_probs for model analysis')
parser.add_argument('--save_gt', action='store_true', help='To save ground truth for model analysis')
parser.add_argument('--save_predictions', action='store_true', help='To save predicted labels for model analysis')
# retrieval args
parser.add_argument('--retrieval', default=0, help='flag for image retrieval array dumps')
parser.add_argument('--random_sample_dim', default=4202000, help='number of random samples to slice from retrieval database', type=int)
parser.add_argument('--retrieval_array_path', default='', help='path to save database and query arrays for retrieval', type=str)


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
test_transform = transforms.Compose([
				transforms.Resize(IMG_SIZE),
				transforms.CenterCrop(CENTER_CROP_SIZE),
				transforms.ToTensor(),
				normalize])

# Model Eval
if args.retrieval == 0:
	if args.dataset == 'V2':
		print("Loading Robustness Dataset")
		dataset = ImageNetV2Dataset("matched-frequency", transform=test_transform)
	elif args.dataset == 'A':
		print("Loading true Imagenet-A val set")
		dataset = torchvision.datasets.ImageFolder(ROOT+'imagenet-a/', transform=test_transform)
	elif args.dataset == 'R':
		print("Loading true Imagenet-R val set")
		dataset = torchvision.datasets.ImageFolder(ROOT+'imagenet-r_/', transform=test_transform)
	elif args.dataset == 'sketch':
		print("Loading Imagenet-Sketch dataset")
		dataset = torchvision.datasets.ImageFolder(ROOT+'sketch/', transform=test_transform)
	else:
		print("Loading true Imagenet 1K val set")
		dataset = torchvision.datasets.ImageFolder(ROOT+'val/', transform=test_transform)

	dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=args.workers, shuffle=False)

	if args.mrl:
		_, top1_acc, top5_acc, total_time, num_images, m_score_dict, softmax_probs, gt, logits = evaluate_model(
				model, dataloader, show_progress_bar=True, nesting_list=NESTING_LIST, tta=args.tta, imagenetA=args.dataset == 'A', imagenetR=args.dataset == 'R')
	else:
		_, top1_acc, top5_acc, total_time, num_images, m_score_dict, softmax_probs, gt, logits = evaluate_model(
				model, dataloader, show_progress_bar=True, nesting_list=None, tta=args.tta, imagenetA=args.dataset == 'A', imagenetR=args.dataset == 'R')

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


	# saving torch tensor for model analysis... 
	if args.save_logits or args.save_softmax or args.save_predictions:
		save_string = f"mrl={args.mrl}_efficient={args.efficient}_dataset={args.dataset}_tta={args.tta}"
		if args.save_logits:
			torch.save(logits, save_string+"_logits.pth")
		if args.save_predictions:
			torch.save(predictions, save_string+"_predictions.pth")
		if args.save_softmax:
			torch.save(softmax_probs, save_string+"_softmax.pth")

	if args.save_gt:
		torch.save(gt, f"gt_dataset={args.dataset}.pth")


# Image Retrieval Inference
elif args.retrieval == 1:
	if args.dataset_name == '1K':
		train_path = 'path_to_imagenet1k_train/'
		train_dataset = datasets.ImageFolder(train_path, transform=test_transform)
		test_dataset = datasets.ImageFolder(ROOT+"val/", transform=test_transform)
	elif args.dataset_name == 'V2':
		train_dataset = None  # V2 has only a test set
		test_dataset = ImageNetV2Dataset("matched-frequency", transform=test_transform)
	elif args.dataset_name == '4K':
		train_path = 'path_to_imagenet4k_train/'
		test_path = 'path_to_imagenet4k_test/'
		train_dataset = datasets.ImageFolder(train_path, transform=test_transform)
		test_dataset = datasets.ImageFolder(test_path, transform=test_transform)
	else:
		print("Error: unsupported dataset!")

	database_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=args.workers, shuffle=False)
	queryset_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=args.workers, shuffle=False)

	config = args.dataset_name + "_val_mrl" + str(args.mrl) + "_e" + str(args.efficient) + "_ff" + str(args.rep_size)
	print("Retrieval Config: " + config)
	generate_retrieval_data(model, queryset_loader, config, args.distributed, args.dataset_name, args.random_sample_dim, args.retrieval_array_path)
	config = args.dataset_name + "_train_mrl" + str(args.mrl) + "_e" + str(args.efficient) + "_ff" + str(args.rep_size)
	print("Retrieval Config: " + config)
	generate_retrieval_data(model, database_loader, config, args.distributed, args.dataset_name, args.random_sample_dim, args.retrieval_array_path)