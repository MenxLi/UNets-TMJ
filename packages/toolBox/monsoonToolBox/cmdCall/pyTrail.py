# +==***---------------------------------------------------------***==+ #
# |                                                                   | #
# |  Filename: pyTrail.py                                             | #
# |  Copyright (C)  - All Rights Reserved                             | #
# |  The code presented in this file is part of an unpublished paper  | #
# |  Unauthorized copying of this file, via any medium is strictly    | #
# |  prohibited                                                       | #
# |  Proprietary and confidential                                     | #
# |  Written by Mengxun Li <mengxunli@whu.edu.cn>, June 2022          | #
# |                                                                   | #
# +==***---------------------------------------------------------***==+ #

import argparse, os
from monsoonToolBox.filetools import pJoin

def main():
	parser = argparse.ArgumentParser("Run python command or interactive mode with pre-imported modules")
	parser.add_argument("-c", "--command", dest="command", default= None, help="run command instead of interactive mode")
	args = parser.parse_args()

	curr_path = os.path.dirname(__file__)
	base_script = pJoin(curr_path, "__pyTrail.py")

	if not args.command:
		os.system(f"python -i {base_script}")
	else:
		with open(base_script, "r") as fp:
			command_to_run = fp.read()
		command_to_run += args.command
		command_to_run = command_to_run.replace("\"", "'")
		command_to_run = "\"{}\"".format(command_to_run)
		os.system(f"python -c {command_to_run}")

if __name__ == "__main__":
	main()
