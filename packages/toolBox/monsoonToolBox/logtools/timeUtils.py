# +==***---------------------------------------------------------***==+ #
# |                                                                   | #
# |  Filename: timeUtils.py                                           | #
# |  Copyright (C)  - All Rights Reserved                             | #
# |  The code presented in this file is part of an unpublished paper  | #
# |  Unauthorized copying of this file, via any medium is strictly    | #
# |  prohibited                                                       | #
# |  Proprietary and confidential                                     | #
# |  Written by Mengxun Li <mengxunli@whu.edu.cn>, June 2022          | #
# |                                                                   | #
# +==***---------------------------------------------------------***==+ #
import time
import datetime
from functools import wraps

def getDateTimeNumStr():
    return str(datetime.datetime.now())[:-7]

class Timer(object):
    def __init__(self, flag = "this operation") -> None:
        super().__init__()
        self.flag = flag

    def __enter__(self):
        self.start_time = time.time()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        t = self.end_time - self.start_time
        time_str = "======> Time for {flag} is: {time}s".format(\
            flag = self.flag, time = t)
        print(time_str)
