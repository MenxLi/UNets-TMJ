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