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