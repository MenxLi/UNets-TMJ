from setuptools import setup, find_packages
from monsoonToolBox._version import __version__

setup(
    name="monsoonToolbox",
    version=__version__,
    author="Li, Mengxun",
    author_email="mengxunli@whu.edu.cn",
    description="My toolbox",

    url="https://github.com/MenxLi/toolBox", 

    packages=find_packages(),

	install_requires=['numpy>=1.19', 'scipy', "matplotlib"],

    entry_points = {
        # "console_scripts":[
            # "tbx_readPickle=monsoonToolBox.cmdCall.readPickle:main",
            # "tbx_createIpynb=monsoonToolBox.cmdCall.createIpynb:main",
            # "tbx_pyTrail=monsoonToolBox.cmdCall.pyTrail:main",
            # "tbx_countLine=monsoonToolBox.cmdCall.devUtils:countLine",
            # "tbx_crun=monsoonToolBox.cmdCall.devUtils:crun",
            # "tbx_findVTKModules=monsoonToolBox.cmdCall.vtkUtils:callFindVTKModules",
        # ]
    }
)