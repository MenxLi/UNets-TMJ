import pickle
import argparse
import pprint

def readPickle(fpath: str):
	assert fpath.endswith(".pkl"), "The file should have an extension of .pkl"
	with open(fpath, "rb") as f:
		data = pickle.load(f)
	return data

def printPickle(fpath:str):
	pprint.pprint(readPickle(fpath))

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-p", "--path")
	parser.add_argument("-r", "--read", action="store_true")
	args = parser.parse_args()

	if args.read:
		printPickle(args.path)