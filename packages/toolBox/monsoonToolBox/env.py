# +==***---------------------------------------------------------***==+ #
# |                                                                   | #
# |  Filename: env.py                                                 | #
# |  Copyright (C)  - All Rights Reserved                             | #
# |  The code presented in this file is part of an unpublished paper  | #
# |  Unauthorized copying of this file, via any medium is strictly    | #
# |  prohibited                                                       | #
# |  Proprietary and confidential                                     | #
# |  Written by Mengxun Li <mengxunli@whu.edu.cn>, June 2022          | #
# |                                                                   | #
# +==***---------------------------------------------------------***==+ #
import os
from .filetools import parDir, pJoin

def setupEnvVar():
	main_dir = parDir(__file__)
	# os.environ("TBX_LOGFILE_PATH") = pJoin(main_dir, "log.txt")

def getEnvVar(name: str):
	var = os.getenv(name)
	if var is None:
		raise Exception("Env {} not set".format(name))
	else:
		return var