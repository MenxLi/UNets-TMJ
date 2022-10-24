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
