# +==***---------------------------------------------------------***==+ #
# |                                                                   | #
# |  Filename: ioUtils.py                                             | #
# |  Copyright (C)  - All Rights Reserved                             | #
# |  The code presented in this file is part of an unpublished paper  | #
# |  Unauthorized copying of this file, via any medium is strictly    | #
# |  prohibited                                                       | #
# |  Proprietary and confidential                                     | #
# |  Written by Mengxun Li <mengxunli@whu.edu.cn>, June 2022          | #
# |                                                                   | #
# +==***---------------------------------------------------------***==+ #
import sys, os, io
from typing import Union

class RedirectStdout(object):
    def __init__(self, outfile: Union[io.TextIOWrapper, None] = None) -> None:
        super().__init__()
        if outfile:
            self.out = outfile
            self.__no_outfile = True
        else:
            self.out = open(os.devnull, "w")
            self.__no_outfile = False
    
    def __enter__(self):
        self.ori_stdout = sys.stdout
        sys.stdout = self.out

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.ori_stdout
        if self.__no_outfile:
            self.out.close()