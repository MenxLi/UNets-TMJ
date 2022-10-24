import time
import datetime
from functools import wraps
import warnings

def getDateTimeNumStr():
    warnings.warn("getDateTimeNumStr in misc will be moved to logtools in the future, use 'from monsoonToolBox.logtools import ...' instead", DeprecationWarning)
    return str(datetime.datetime.now())[:-7]

def timedFunc(flag = ""):
    warnings.warn("timedFunc in misc will be moved to logtools in the future, use 'from monsoonToolBox.logtools import ...' instead", DeprecationWarning)
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
