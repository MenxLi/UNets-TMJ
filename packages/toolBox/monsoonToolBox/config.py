# +==***---------------------------------------------------------***==+ #
# |                                                                   | #
# |  Filename: config.py                                              | #
# |  Copyright (C)  - All Rights Reserved                             | #
# |  The code presented in this file is part of an unpublished paper  | #
# |  Unauthorized copying of this file, via any medium is strictly    | #
# |  prohibited                                                       | #
# |  Proprietary and confidential                                     | #
# |  Written by Mengxun Li <mengxunli@whu.edu.cn>, June 2022          | #
# |                                                                   | #
# +==***---------------------------------------------------------***==+ #
from ._version import __version__

def init():
    print("Using monsoonToolBox v{}".format(__version__))

def setMatplotlibBackend(backend: str):
    """set matplotlib backend, avaliable backends:
    https://matplotlib.org/stable/users/explain/backends.html

    Args:
        backend (str)
    """
    import matplotlib
    matplotlib.use(backend)

def setPltFiguresize(w: int, h: int):
    """set matplotlib backend, avaliable backends:
    https://matplotlib.org/stable/users/explain/backends.html

    Args:
        backend (str)
    """
    import matplotlib.pyplot as plt
    plt.rcParams["figure.figsize"] = (w, h)