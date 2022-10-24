
import argparse
from ..misc import printPickle

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("file")
	args = parser.parse_args()

	printPickle(args.file)