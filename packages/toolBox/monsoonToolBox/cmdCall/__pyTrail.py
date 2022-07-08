# +==***---------------------------------------------------------***==+ #
# |                                                                   | #
# |  Filename: __pyTrail.py                                           | #
# |  Copyright (C)  - All Rights Reserved                             | #
# |  The code presented in this file is part of an unpublished paper  | #
# |  Unauthorized copying of this file, via any medium is strictly    | #
# |  prohibited                                                       | #
# |  Proprietary and confidential                                     | #
# |  Written by Mengxun Li <mengxunli@whu.edu.cn>, June 2022          | #
# |                                                                   | #
# +==***---------------------------------------------------------***==+ #
import os, sys, re, time, math
from math import pi, sin, cos, sqrt, log, log2, log10, ceil, floor

import monsoonToolBox as tbx

__maybe_exec = [
	"import numpy as np",
	"import cv2 as cv",
	"import matplotlib.pyplot as plt"
]

for __command in __maybe_exec:
	try:
		exec(__command)
	except ModuleNotFoundError:
		pass
