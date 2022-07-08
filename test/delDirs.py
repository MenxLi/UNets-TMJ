# Delete directories

import shutil

to_remove = [
    "./.TempDir",
    "./Database",
    "./UNetPPTMJ/.TempDir",
    "./nnUNetTMJ/.TempDir",
    "./evaluation/.TempDir",
]

if input("Delete directories?: \n{}\n (y/n) ".format(to_remove)) == "y":
    for path in to_remove:
        try:
            shutil.rmtree(path)
        except:
            pass