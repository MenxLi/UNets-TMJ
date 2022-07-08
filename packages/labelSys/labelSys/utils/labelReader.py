# +==***---------------------------------------------------------***==+ #
# |                                                                   | #
# |  Filename: labelReader.py                                         | #
# |  Copyright (C)  - All Rights Reserved                             | #
# |  The code presented in this file is part of an unpublished paper  | #
# |  Unauthorized copying of this file, via any medium is strictly    | #
# |  prohibited                                                       | #
# |  Proprietary and confidential                                     | #
# |  Written by Mengxun Li <mengxunli@whu.edu.cn>, June 2022          | #
# |                                                                   | #
# +==***---------------------------------------------------------***==+ #
from .base64ImageConverter import imgDecodeB64
import numpy as np
import cv2 as cv
import re, os, json, sys

try:
    import skimage.transform

    USE_SKIMAGE = True
except ModuleNotFoundError:
    USE_SKIMAGE = False
import vtk


def checkFolderEligibility(fpath: str):
    flag = False
    for fname in os.listdir(fpath):
        if fname == "HEAD_0.json":
            flag = True
    return flag


def readOneFolder(path, magnification=1, line_thickness=1):  # {{{
    imgs = []
    masks = []
    file_list = [x for x in os.listdir(path) if x.endswith(".json")]
    for file_name in sorted(file_list, key=lambda x: int(re.findall("\d+|$", x)[0])):
        file_path = os.path.join(path, file_name)
        if file_name == "HEAD_0.json":
            with open(file_path, "r") as _f:
                header_data = json.load(_f)
        else:
            img, mask = readDataSlice(file_path, magnification, line_thickness)
            imgs.append(img)
            masks.append(mask)
    return imgs, masks


# }}}
def readDataSlice(file_path, magnification=1, line_thickness=1):  # {{{
    """
    - line_thickness: the line thickness of the interpreted mask, if the label is open contour
    read one data slice (i.e. not header file), the image and masks are magified with masks obtained\
        with VTK finely interpreted.
    return image, masks
    the masks is saved as dictionary
    """
    with open(file_path, "r") as _f:
        slice_data = json.load(_f)

    # resize image
    img_ori = imgDecodeB64(slice_data["Image"], accelerate=True)
    if len(img_ori.shape) == 2:
        im_channel = 1
    elif len(img_ori.shape) == 3:
        im_channel = img_ori.shape[2]
    new_im_size = (
        (np.array(img_ori.shape[:2]) * magnification).astype(np.int).tolist()
    )  # H, W
    if USE_SKIMAGE:
        if im_channel == 1:
            img = skimage.transform.resize(img_ori, new_im_size)
        else:
            img = skimage.transform.resize(img_ori, new_im_size + [im_channel])
    else:
        print("Opencv found, using opencv instead")
        img_ori_uint8 = (
            (img_ori - img_ori.min()) / (img_ori.max() - img_ori.min()) * 255
        )
        img_ori_uint8 = img_ori_uint8.astype(np.uint8)
        #  if im_channel == 1:
        #      img_ori_uint8 = np.concatenate((img_ori_uint8[:,:,np.newaxis], img_ori_uint8[:,:,np.newaxis], img_ori_uint8[:,:,np.newaxis]), axis = 2)
        img = cv.resize(
            img_ori_uint8, tuple(new_im_size[::-1]), interpolation=cv.INTER_LINEAR
        )

    masks = dict()
    for label, data in slice_data["Data"].items():
        if not isinstance(data, list):
            # SOPInstanceUID
            continue
        if data == []:
            masks[label] = np.zeros(img.shape[:2])
        else:
            masks[label] = _readOneLabel(
                img_ori.shape[:2], data, magnification, line_thickness
            )
    return img, masks


# }}}
def _readOneLabel(ori_im_size, data, magnification, line_thickness):  # {{{
    """
    will be called by readDataSlice(...)
    - ori_im_size: original image size (H, W)
    - data: data from one label
    return one mask
    """
    new_im_size = (np.array(ori_im_size) * magnification).astype(np.int)
    mask = np.zeros(new_im_size[:2], np.uint8)
    for d in data:
        open_curve = d["Open"]
        pts = d["Points"]
        #  contour_widget.AddObserver('EndInteractionEvent', contourWidgetEndInteraction)
        #  contour_widget.AddObserver('WidgetValueChangedEvent', contourWidgetEndInteraction)

        pd = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        for i in range(len(pts)):
            points.InsertPoint(i, pts[i])

        if not open_curve:
            vertex_ids = list(range(len(pts))) + [0]
            lines.InsertNextCell(len(pts) + 1, vertex_ids)
        else:
            vertex_ids = list(range(len(pts)))
            lines.InsertNextCell(len(pts), vertex_ids)

        pd.SetPoints(points)
        pd.SetLines(lines)

        # create a contour widget
        renderer = vtk.vtkRenderer()
        ren_win = vtk.vtkRenderWindow()
        ren_win.AddRenderer(renderer)

        contourRep = vtk.vtkOrientedGlyphContourRepresentation()
        contour_widget = vtk.vtkContourWidget()

        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(ren_win)
        contour_widget.SetInteractor(iren)
        contour_widget.SetRepresentation(contourRep)

        contour_widget.On()
        contour_widget.Initialize(pd, 1)

        cnt = _getFullCnt(contour_widget, ori_im_size)  # All points on the contour
        mask = drawMask(mask, np.array(cnt) * magnification, open_curve, line_thickness)
    return mask


# }}}
def _getFullCnt(contour_widget, img_shape):  # {{{{{{
    """
    Get all point position in a contour widget
    return point in (col, row)
    -img_shape : (H, W) - original image shape
    """
    if len(img_shape) == 3:
        img_shape = img_shape[:2]
    cnt = contour_widget
    rep = cnt.GetContourRepresentation()
    all_pts = []
    point = np.empty((3))
    for pt_id in range(rep.GetNumberOfNodes()):
        rep.GetNthNodeWorldPosition(pt_id, point)
        all_pts.append(_getBackCvCoord(*point[:2], img_shape))
        for ipt_id in range(rep.GetNumberOfIntermediatePoints(pt_id)):
            rep.GetIntermediatePointWorldPosition(pt_id, ipt_id, point)
            all_pts.append(_getBackCvCoord(*point[:2], img_shape))
    all_pts = np.array(all_pts)
    return all_pts.tolist()


# }}}
def _getBackCvCoord(x, y, img_shape):  # {{{{{{
    """Get coordinate in (col, row)
    - img_shape: (W, H)"""
    return np.array([x, img_shape[0] - 1 - y])


# }}}
def drawMask(mask, cnt, open_curve, line_thickness):  # {{{
    """
    draw the mask from contour points
    will be called by _readOneLabel()
    - mask: mask to be drawn
    - cnt: all points on the contour in (x, y)
    - open_curve: indicate if the contour is open
    - line_thickness: the line thickness if the contour is open
    """
    if not isinstance(cnt, np.ndarray):
        cnt = np.array(cnt)
    cnt = _removeDuplicate2d(cnt.astype(int))
    if open_curve:
        cnt = np.array(cnt)
        cv.polylines(mask, [cnt], False, 1, thickness=line_thickness)
    else:
        cnt = np.array([[arr] for arr in cnt])
        cv.fillPoly(mask, pts=[cnt], color=1)
    return mask


# }}}
def gray2rgb(im):  # {{{
    return np.concatenate(
        (im[:, :, np.newaxis], im[:, :, np.newaxis], im[:, :, np.newaxis]), axis=2
    )


# }}}
def format2Uint8(im):  # {{{
    return ((im - im.min()) / (im.max() - im.min()) * 255).astype(np.uint8)


# }}}
def inspectOneSlice(slice_path):  # {{{
    img, masks = readDataSlice(slice_path, magnification=1)
    img = format2Uint8(img)

    if len(img.shape) == 2:
        im_channel = 1
    elif len(img.shape) == 3:
        im_channel = img.shape[2]
    mask = np.zeros(img.shape[:2], np.float)
    value = 0
    for label, _mask in masks.items():
        value += 1
        mask = mask * (1 - _mask) + value * _mask
    mask = (mask / value) * 255
    mask = mask.astype(np.uint8)

    if im_channel == 1:
        show_im = np.concatenate((img, mask), axis=1)
    else:
        show_im = np.concatenate((img, gray2rgb(mask)), axis=1)
    cv.imshow(slice_path, cv.cvtColor(show_im, cv.COLOR_RGB2BGR))
    cv.waitKey(0)
    return masks


# }}}


def _removeDuplicate2d(duplicate):  # {{{
    final_list = []
    flag = True
    for num in duplicate:
        for num0 in final_list:
            if num[0] == num0[0] and num[1] == num0[1]:
                flag = False
        if flag:
            final_list.append(num)
        flag = True
    return final_list


# }}}

if __name__ == "__main__":
    slice_path = sys.argv[1]
    inspectOneSlice(slice_path)
