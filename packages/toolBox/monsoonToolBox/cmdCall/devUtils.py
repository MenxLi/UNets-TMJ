
import os, pprint
import argparse
from typing import List

def countLine() -> None:
    _description = "\
    Count file lines from a directory of specific suffix\
    i.e. countLine . -s .py .json -i .git build\
    "
    parser = argparse.ArgumentParser(description=_description)
    parser.add_argument("path", type = str)
    parser.add_argument("-s", "--suffix", nargs="+", default=[".txt", ".py", ".js", ".c", ".h", ".html", ".json"])
    parser.add_argument("-i", "--ignore", nargs="+", default=["dist", "__pycache__", ".git", "build"])
    args = parser.parse_args()

    def _getFile_recursive(pth: str, suffix: List[str]) -> List[str]:
        file_valid = []
        assert os.path.isdir(pth), "input should be a directory."
        for f in os.listdir(pth):
            if f in args.ignore:
                continue
            f_path = os.path.join(pth, f)
            if os.path.isfile(f_path):
                for suffix_ in suffix:
                    if f.endswith(suffix_):
                        file_valid.append(f_path)
            elif os.path.isdir(f_path):
                file_valid += _getFile_recursive(f_path, suffix)
        return file_valid
    
    valid_files = _getFile_recursive(args.path, args.suffix)
    outcome = dict()
    for k in args.suffix:
        outcome[k] = {
            "files":[],
            "count":0
        }
    total_count = 0
    for f in valid_files:
        suffix_ = "."+f.split(".")[-1]
        with open(f, "r") as fp:
            count_ = len(fp.readlines())
            total_count += count_
        outcome[suffix_]["files"].append(f)
        outcome[suffix_]["count"] += count_
    pprint.pprint(outcome)
    print("Total: ", total_count)


def crun() -> None:
    """Run c program with 'gcc <file> -o tbx_tmp.o; ./tbx_tmp.o; rm tbx_tmp.o'
    """
    _description = "\
    Run c program with 'gcc <file> -o tbx_tmp.o; ./tbx_tmp.o; rm tbx_tmp.o'\
    "
    parser = argparse.ArgumentParser(description=_description)
    parser.add_argument("file", type = str)
    args = parser.parse_args()
    os.system("gcc -o ./tbx_tmp.o {}".format(args.file))
    os.system("./tbx_tmp.o")
    os.system("rm ./tbx_tmp.o")