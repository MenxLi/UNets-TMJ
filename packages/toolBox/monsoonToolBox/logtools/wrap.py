from functools import wraps
import sys
import time

from .logger import Logger
from .timeUtils import getDateTimeNumStr


def logFuncOutput(log_path, flag:str = "", mode = "a", terminal:bool = True):
    def wapper(func):
        @wraps(func)
        def _func(*args, **kwargs):
            std_out = sys.stdout
            std_err = sys.stderr
            with open(log_path, mode) as log_file:
                sys.stdout = Logger(log_file, write_to_terminal = terminal)
                sys.stderr = Logger(log_file, write_to_terminal = terminal)
                print("{time}: {name} - {flag}".format(time = getDateTimeNumStr(), name = func.__name__, flag = flag))
                output = func(*args, **kwargs)
            sys.stdout = std_out
            sys.stderr = std_err
            return output
        return _func
    return wapper

def timedFunc(flag = ""):
    def wrap(func):
        @wraps(func)
        def _func(*args, **kwargs):
            t = time.time()
            out = func(*args, **kwargs)
            t = time.time() - t
            time_str = "======> Time for function {name} (flag: {flag}) is: {time}s".format(\
                name = func.__name__, flag = flag, time = t)
            print(time_str)
            return out
        return _func
    return wrap
