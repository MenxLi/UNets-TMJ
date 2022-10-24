from setuptools import setup, find_packages
import importlib
from labelSys.version import __version__

# Do not install opencv if any of cv variation exists
# e.g. opencv-headless, opencv-contrib
install_requires = ["numpy", "vtk", "scipy"]
cv_spec = importlib.util.find_spec("cv2")
if cv_spec is None:
    install_requires.append("opencv-python")

setup(
    name="LabelSys",
    version=__version__,
    author="Mengxun Li",
    author_email="mengxunli@whu.edu.cn",
    description="A segmentation labeling software",
    url="https://github.com/MenxLi/LabelSys",
    packages=find_packages(),
    classifiers=[
        #   Development Status
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 4",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.5",
    include_package_data=True,
    install_requires=install_requires,
    entry_points={
        "console_scripts": [
            # "labelSys=labelSys.exec:main",
            # "labelSys_=labelSys.exec:main_",
        ]
    },
)
