# +==***---------------------------------------------------------***==+ #
# |                                                                   | #
# |  Filename: oneHot.py                                              | #
# |  Copyright (C)  - All Rights Reserved                             | #
# |  The code presented in this file is part of an unpublished paper  | #
# |  Unauthorized copying of this file, via any medium is strictly    | #
# |  prohibited                                                       | #
# |  Proprietary and confidential                                     | #
# |  Written by Mengxun Li <mengxunli@whu.edu.cn>, June 2022          | #
# |                                                                   | #
# +==***---------------------------------------------------------***==+ #
import torch

def oneHot2D(targets, n_classes, device = "cpu"):    
	"""
    - targets: shape - [N, H, W]
    return one_hot: shape - [N, C, H, W]
    """
	targets_extend=targets.clone()
	targets_extend.unsqueeze_(1) # convert to Nx1xHxW
	one_hot = torch.FloatTensor(targets_extend.size(0), n_classes, targets_extend.size(2), targets_extend.size(3)).zero_().to(device)
	one_hot.scatter_(1, targets_extend, 1) 
	return one_hot