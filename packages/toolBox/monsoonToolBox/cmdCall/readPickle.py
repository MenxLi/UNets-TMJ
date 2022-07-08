# +==***---------------------------------------------------------***==+ #
# |                                                                   | #
# |  Filename: readPickle.py                                          | #
# |  Copyright (C)  - All Rights Reserved                             | #
# |  The code presented in this file is part of an unpublished paper  | #
# |  Unauthorized copying of this file, via any medium is strictly    | #
# |  prohibited                                                       | #
# |  Proprietary and confidential                                     | #
# |  Written by Mengxun Li <mengxunli@whu.edu.cn>, June 2022          | #
# |                                                                   | #
# +==***---------------------------------------------------------***==+ #

import argparse
from ..misc import printPickle

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("file")
	args = parser.parse_args()

	printPickle(args.file)