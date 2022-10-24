import platform, os, subprocess, shutil
from .path import subDirAndFiles
from typing import List

def openFile(filepath):
	"""Use system application to open a file"""
	# https://stackoverflow.com/questions/434597/open-document-with-default-os-application-in-python-both-in-windows-and-mac-os
	if platform.system() == 'Darwin':       # macOS
		subprocess.call(('open', filepath))
	elif platform.system() == 'Windows':    # Windows
		os.startfile(filepath)
	else:                                   # linux variants
		subprocess.call(('xdg-open', filepath))

def clearDir(dir_path: str):
	"""
	Delete every files or sub-directories under dir_path	
	"""
	assert os.path.isdir(dir_path), "Input path is not a directory, clearDir function only accept directory path as input argument"
	for p in subDirAndFiles(dir_path):
		if os.path.isdir(p):
			shutil.rmtree(p)
		else:
			os.remove(p)
		print("{} cleared.".format(dir_path))

def recursivlyFindFilesByExtension(pth: str, suffix: List[str], ignore: List[str] = []) -> List[str]:
	file_valid = []
	assert os.path.isdir(pth), "input should be a directory."
	for f in os.listdir(pth):
		if f in ignore:
			continue
		f_path = os.path.join(pth, f)
		if os.path.isfile(f_path):
			for suffix_ in suffix:
				if f.endswith(suffix_):
					file_valid.append(f_path)
		elif os.path.isdir(f_path):
			file_valid += recursivlyFindFilesByExtension(f_path, suffix)
	return file_valid