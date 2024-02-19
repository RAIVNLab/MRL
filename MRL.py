from typing import List

import torch
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional

'''
Loss function for Matryoshka Representation Learning 
'''

class Matryoshka_CE_Loss(nn.Module):
	def __init__(self, relative_importance: List[float]=None, **kwargs):
		super(Matryoshka_CE_Loss, self).__init__()
		self.criterion = nn.CrossEntropyLoss(**kwargs)
		# relative importance shape: [G]
		self.relative_importance = relative_importance

	def forward(self, output, target):
		# output shape: [G granularities, N batch size, C number of classes]
		# target shape: [N batch size]

		# Calculate losses for each output and stack them. This is still O(N)
		losses = torch.stack([self.criterion(output_i, target) for output_i in output])
		
		# Set relative_importance to 1 if not specified
		rel_importance = torch.ones_like(losses) if self.relative_importance is None else torch.tensor(self.relative_importance)
		
		# Apply relative importance weights
		weighted_losses = rel_importance * losses
		return weighted_losses.sum()

class Matryoshka_MSE_Loss(nn.Module):
	def __init__(self, relative_importance: List[float]=None, **kwargs):
		super(Matryoshka_MSE_Loss, self).__init__()
		self.criterion = nn.MSELoss(**kwargs)
		# relative importance shape: [G]
		self.relative_importance = relative_importance

	def forward(self, output, target):
		
		# output shape: [G granularities, N batch size, C number of classes]
		# target shape: [N batch size]

		# Calculate losses for each output and stack them. This is still O(N)
		losses = torch.stack([self.criterion(output_i, target) for output_i in output])
		
		# Set relative_importance to 1 if not specified
		rel_importance = torch.ones_like(losses) if self.relative_importance is None else torch.tensor(self.relative_importance)
		
		# Apply relative importance weights
		weighted_losses = rel_importance * losses
		
		return weighted_losses.sum()


class MRL_Linear_Layer(nn.Module):
	def __init__(self, nesting_list: List, num_classes=1000, efficient=False, **kwargs):
		super(MRL_Linear_Layer, self).__init__()
		self.nesting_list = nesting_list
		self.num_classes = num_classes # Number of classes for classification
		self.efficient = efficient
		if self.efficient:
			setattr(self, f"nesting_classifier_{0}", nn.Linear(nesting_list[-1], self.num_classes, **kwargs))		
		else:	
			for i, num_feat in enumerate(self.nesting_list):
				setattr(self, f"nesting_classifier_{i}", nn.Linear(num_feat, self.num_classes, **kwargs))	

	def reset_parameters(self):
		if self.efficient:
			self.nesting_classifier_0.reset_parameters()
		else:
			for i in range(len(self.nesting_list)):
				getattr(self, f"nesting_classifier_{i}").reset_parameters()


	def forward(self, x):
		nesting_logits = ()
		for i, num_feat in enumerate(self.nesting_list):
			if self.efficient:
				if self.nesting_classifier_0.bias is None:
					nesting_logits += (torch.matmul(x[:, :num_feat], (self.nesting_classifier_0.weight[:, :num_feat]).t()), )
				else:
					nesting_logits += (torch.matmul(x[:, :num_feat], (self.nesting_classifier_0.weight[:, :num_feat]).t()) + self.nesting_classifier_0.bias, )
			else:
				nesting_logits +=  (getattr(self, f"nesting_classifier_{i}")(x[:, :num_feat]),)

		return nesting_logits


class FixedFeatureLayer(nn.Linear):
    '''
    For our fixed feature baseline, we just replace the classification layer with the following. 
    It effectively just look at the first "in_features" for the classification. 
    '''

    def __init__(self, in_features, out_features, **kwargs):
        super(FixedFeatureLayer, self).__init__(in_features, out_features, **kwargs)

    def forward(self, x):
        if not (self.bias is None):
            out = torch.matmul(x[:, :self.in_features], self.weight.t()) + self.bias
        else:
            out = torch.matmul(x[:, :self.in_features], self.weight.t())
        return out
        
