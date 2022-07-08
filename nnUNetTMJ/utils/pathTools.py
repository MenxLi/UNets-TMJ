# +==***---------------------------------------------------------***==+ #
# |                                                                   | #
# |  Filename: pathTools.py                                           | #
# |  Copyright (C)  - All Rights Reserved                             | #
# |  The code presented in this file is part of an unpublished paper  | #
# |  Unauthorized copying of this file, via any medium is strictly    | #
# |  prohibited                                                       | #
# |  Proprietary and confidential                                     | #
# |  Written by Mengxun Li <mengxunli@whu.edu.cn>, June 2022          | #
# |                                                                   | #
# +==***---------------------------------------------------------***==+ #
import os, typing, shutil


def subDirAndFiles(dir_path: str):
    assert os.path.isdir(
        dir_path
    ), "Input path is not a directory, subFiles function only accept directory path as input argument"
    return [os.path.join(dir_path, f) for f in os.listdir(dir_path)]


def clearDir(dir_path: str):
    assert os.path.isdir(
        dir_path
    ), "Input path is not a directory, clearDir function only accept directory path as input argument"
    for p in subDirAndFiles(dir_path):
        if os.path.isdir(p):
            shutil.rmtree(p)
        else:
            os.remove(p)
        print("{} cleared.".format(dir_path))
