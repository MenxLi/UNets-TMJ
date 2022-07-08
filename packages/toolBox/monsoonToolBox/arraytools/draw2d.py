# +==***---------------------------------------------------------***==+ #
# |                                                                   | #
# |  Filename: draw2d.py                                              | #
# |  Copyright (C)  - All Rights Reserved                             | #
# |  The code presented in this file is part of an unpublished paper  | #
# |  Unauthorized copying of this file, via any medium is strictly    | #
# |  prohibited                                                       | #
# |  Proprietary and confidential                                     | #
# |  Written by Mengxun Li <mengxunli@whu.edu.cn>, June 2022          | #
# |                                                                   | #
# +==***---------------------------------------------------------***==+ #
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.figure
from typing import List, Union
from .mask2d import Mask2D
from .img2d import Img2D

class Drawer2D(Mask2D):
    LINE_THICKNESS = 1
    DRAWING_MODE = "outline"
    DEFAULT_COLOR = (255,0,0)

    @staticmethod
    def imshowGrid(imgs: List[np.ndarray], n_col:int = 2, **imshow_params):
        n_imgs = len(imgs)
        n_row = np.ceil(n_imgs/n_col)
        for i in range(n_imgs):
            plt.subplot(int(n_row), int(n_col), i+1)
            plt.imshow(imgs[i], **imshow_params)

    @staticmethod
    def overlapMask(img: np.ndarray, mask: np.ndarray, color: Union[tuple, list] = (255,0,0), alpha = 1) -> np.ndarray:
        if Drawer2D.imgChannel(img) == 1:
            img = Drawer2D.gray2rgb(img)
        if Drawer2D.imgChannel(mask) == 1:
            mask = Drawer2D.gray2rgb(mask)
        im = img.astype(float)
        channel = np.ones(img.shape[:2], np.float)
        color_ = np.concatenate((channel[:,:,np.newaxis]*color[0],channel[:,:,np.newaxis]*color[1],channel[:,:,np.newaxis]*color[2]), axis = 2)
        f_im = im*(1-mask) + im*mask*(1-alpha) + color_*alpha*mask
        return f_im.astype(np.uint8)
    
    @staticmethod
    def drawMasks(img: np.ndarray, masks: List[np.ndarray], alpha: float = 1, colors : Union[None, List[tuple], tuple] = None):
        """
        - alpha: transparency
        """
        if colors is None:
            colors = [Drawer2D.DEFAULT_COLOR]*len(masks)
        assert len(colors) == len(masks),"mask and color should have the same length" 
        resultant = Drawer2D.mapMatUint8(img)
        for mask, color in zip(masks, colors):
            resultant = Drawer2D.overlapMask(resultant, mask, color, alpha = alpha)
        return resultant

    @staticmethod
    def drawOutline(img, mask, color, mode, line_thickness = 1):
        pass
    
    @staticmethod
    def _visualCompareSegmentations(img: np.ndarray, masks: List[List[np.ndarray]], alpha:float = 1.0, colors = None, tags = None):
        """
        generate comprehensive images of the labels to compare segemtation results
        - masks: list of per image masks
        """
        if tags is None:
            tags = [""]*len(masks)
        if len(masks) != len(tags):
            raise Exception("mask and tag should have same dimension")
        ims = []
        img = Drawer2D.mapMatUint8(img)
        if Drawer2D.imgChannel(img) == 1:
            img = Drawer2D.gray2rgb(img)
        im0 = Drawer2D.addTagToImg(img, "Original image"); ims.append(im0)
        for mask, tag in zip(masks, tags):
            im = Drawer2D.addTagToImg(Drawer2D.drawMasks(img, mask, alpha, colors), tag); ims.append(im)
        return np.concatenate(ims, axis = 1)
    
    @staticmethod
    def visualCompareLocation(self, img, masks):
        pass
    
    @staticmethod
    def addTagToImg(img, txt, font = cv.FONT_HERSHEY_SIMPLEX, org = (5, 20), fontScale = 0.5, color = (255,255,255), **kwargs):
        im = img.copy()
        im = cv.putText(im, txt, org = (5,20), fontFace=font, fontScale=fontScale, color = color, **kwargs)
        return im
    
    @staticmethod
    def visualCompareSegmentations(img: np.ndarray, masks_ori: List[np.ndarray], color_dict: dict, alpha: float = 1.0, tags: Union[List[str], None] = None):
        """
        generate comprehensive images of the labels to compare segemtation results
        - masks_ori
        - color_dict: 
            etc. color_dict = {
                1: (255, 0, 0),
                2: (0, 255, 0),
                3: (0, 100, 255)
            }
            (when unique(masks_ori) == (0, 1, 2, 3))
        """
        ms = np.zeros(shape = (len(masks_ori), len(color_dict.keys()), *masks_ori[0].shape))
        colors = []
        idx = 0
        for k,v in color_dict.items():
            colors.append(v)
            for i in range(len(masks_ori)):
                ms[i, idx] = (masks_ori[i] == k)
            idx += 1
        return Drawer2D._visualCompareSegmentations(img, ms, tags = tags, alpha =alpha, colors = colors)
    
    def gridImgs(imgs: List[np.array], n_col: int, **figure_kwargs) -> matplotlib.figure.Figure:
        n_im = len(imgs)
        n_row = np.ceil(n_im/n_col)
        fig: matplotlib.figure.Figure = plt.figure(**figure_kwargs)
        for i in range(n_im):
            ax = fig.add_subplot(int(n_row), int(n_col), i+1)
            ax.imshow(imgs[i])
        return fig
    
