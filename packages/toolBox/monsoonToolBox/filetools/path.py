import os
from typing import List, Union
from pathlib import Path

pJoin = os.path.join
parDir = os.path.dirname

	
def subDirAndFiles(dir_path: str):
	assert os.path.isdir(dir_path), "Input path is not a directory, subFiles function only accept directory path as input argument"
	return [os.path.join(dir_path, f) for f in os.listdir(dir_path)]

def subFiles(dir_path: str):
	dir_and_file = subDirAndFiles(dir_path)
	return [i for i in dir_and_file if os.path.isfile(i)]

def subDirs(dir_path: str) -> List[str]:
	"""List sub-directories

	Args:
		dir_path (str): directory path to be searched in.

	Returns:
		List[str]: list of sub-directories
	"""
	dir_and_file = subDirAndFiles(dir_path)
	return [i for i in dir_and_file if os.path.isdir(i)]

def mkdiR(dir_path: str) -> None:
	"""Recursively make directories, if dir_path not exists.

	Args:
		dir_path (str): path of the directory.

	Returns:
		bool: True if create the path, False if the path exists
	"""
	dir_path = Path.expanduser(Path(dir_path))
	if os.path.exists(dir_path):
		return False
	
	parent_dir = os.path.dirname(dir_path)
	if os.path.exists(parent_dir):
		os.mkdir(dir_path)
		return True
	else:
		mkdiR(parent_dir)
		os.mkdir(dir_path)
		return True