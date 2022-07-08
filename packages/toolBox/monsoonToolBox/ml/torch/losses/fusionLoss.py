# +==***---------------------------------------------------------***==+ #
# |                                                                   | #
# |  Filename: fusionLoss.py                                          | #
# |  Copyright (C)  - All Rights Reserved                             | #
# |  The code presented in this file is part of an unpublished paper  | #
# |  Unauthorized copying of this file, via any medium is strictly    | #
# |  prohibited                                                       | #
# |  Proprietary and confidential                                     | #
# |  Written by Mengxun Li <mengxunli@whu.edu.cn>, June 2022          | #
# |                                                                   | #
# +==***---------------------------------------------------------***==+ #
import torch.nn as nn
import torch

class FusionLoss(nn.Module):
	def __init__(self, losses: list, weights = None, device = "cuda"):
		super().__init__()
		self.losses = losses
		self.device = device
		if weights is None:
			weights = torch.ones(len(losses))
		else:
			weights = torch.tensor(weights)
		self.weights = weights/weights.sum()
	
	def forward(self, input, target):
		losses_ = [loss_fn(input, target) for loss_fn in self.losses]
		loss = torch.tensor(0, dtype = torch.float32).to(self.device)
		for l, w in zip(losses_, self.weights):
			loss += l*w
		return loss
	
	def __str__(self) -> str:
		return "FusionLoss-({})".format("+".join(["{}-{}".format(str(loss), str(weight)) for loss, weight in zip(self.losses, self.weights)]))
	
	__repr__ = __str__

