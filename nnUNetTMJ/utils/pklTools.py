# +==***---------------------------------------------------------***==+ #
# |                                                                   | #
# |  Filename: pklTools.py                                            | #
# |  Copyright (C)  - All Rights Reserved                             | #
# |  The code presented in this file is part of an unpublished paper  | #
# |  Unauthorized copying of this file, via any medium is strictly    | #
# |  prohibited                                                       | #
# |  Proprietary and confidential                                     | #
# |  Written by Mengxun Li <mengxunli@whu.edu.cn>, June 2022          | #
# |                                                                   | #
# +==***---------------------------------------------------------***==+ #
import pickle
import argparse


def readPickle(fpath: str):
    assert fpath.endswith(".pkl"), "The file should have an extension of .pkl"
    with open(fpath, "rb") as f:
        data = pickle.load(f)
    return data


def printPickle(fpath: str):
    print(readPickle(fpath))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path")
    parser.add_argument("-r", "--read", action="store_true")
    args = parser.parse_args()

    if args.read:
        printPickle(args.path)
