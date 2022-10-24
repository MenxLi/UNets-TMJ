import sys
try:
	from notebook import transutils as _
	from notebook.services.contents.filemanager import FileContentsManager as FCM
except ModuleNotFoundError:
	print("The notebook module is not installed on this enviroment.")
	sys.exit()

def main():
	try:
		notebook_fname = sys.argv[1]
		if notebook_fname.endswith(".ipynb"):
			notebook_fname = notebook_fname[:-6]
	except IndexError:
		print("Usage: create-notebook <notebook>")
		exit()

	notebook_fname += '.ipynb'  # ensure .ipynb suffix is added
	FCM().new(path=notebook_fname)

if __name__=="__main__":
	main()