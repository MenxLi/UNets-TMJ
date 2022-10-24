#
# Copyright (c) 2020 Mengxun Li.
#
# This file is part of LabelSys
# (see https://bitbucket.org/Mons00n/mrilabelsys/).
#
"""
Useful functions
"""

# import{{{
import numpy as np

try:
    import scipy.ndimage
except:
    pass
try:
    import cv2 as cv
except:
    pass
try:
    import matplotlib.pyplot as plt
except:
    pass
import uuid

# }}}

import platform, os, subprocess


def sUUID() -> str:
    """generate short UUID of length 21
    Returns:
        str: short string uuid
    """
    B16_TABLE = {
        "0": "0000",
        "1": "0001",
        "2": "0010",
        "3": "0011",
        "4": "0100",
        "5": "0101",
        "6": "0110",
        "7": "0111",
        "8": "1000",
        "9": "1001",
        "a": "1010",
        "b": "1011",
        "c": "1100",
        "d": "1101",
        "e": "1110",
        "f": "1111",
    }
    B64_LIST =[
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",\
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",\
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "/"
    ]
    uuid_long = uuid.uuid4().hex
    # b_asciis = " ".join(["{0:08b}".format(ord(u), "b") for u in uuid_long])
    binary = "".join([B16_TABLE[u] for u in uuid_long])
    # len(binary) = 128
    uuid_short = ["0"] * 21
    for i in range(0, len(binary) - 2, 6):
        b = binary[i : i + 6]
        integer = int(b, 2)
        uuid_short[i // 6] = B64_LIST[integer]
    return "".join(uuid_short)


def ssUUID() -> str:
    """Short uuid of length 16

    Returns:
        str: uuid
    """
    suid = sUUID()
    return suid[:8] + suid[-8:]


def cv_imread(file_path):
    """Read image even if path contain 中文"""
    cv_img = cv.imdecode(np.fromfile(file_path, dtype=np.uint8), cv.IMREAD_COLOR)
    return cv_img


def openFile(filepath):
    """Use system application to open a file"""
    # https://stackoverflow.com/questions/434597/open-document-with-default-os-application-in-python-both-in-windows-and-mac-os
    if platform.system() == 'Darwin':  # macOS
        subprocess.call(('open', filepath))
    elif platform.system() == 'Windows':  # Windows
        os.startfile(filepath)
    else:  # linux variants
        subprocess.call(('xdg-open', filepath))


def gray2rgb_(img):  # {{{
    new_img = np.concatenate(
        (
            img.copy()[:, :, np.newaxis],
            img.copy()[:, :, np.newaxis],
            img.copy()[:, :, np.newaxis],
        ),
        axis=2,
    )
    return new_img


# }}}
def removeDuplicate2d(duplicate):  # {{{
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
def plot_imgs(imgs, n_col=3, gray=False, titles=None):  # {{{
    """Take a list of images and plot them in grids"""
    if titles == None:
        print_title = False
    else:
        print_title = True
    if len(imgs) // n_col == len(imgs) / n_col:
        n_row = len(imgs) // n_col
    else:
        n_row = len(imgs) // n_col + 1
    _, axs = plt.subplots(n_row, n_col)
    axs = axs.flatten()
    count = 0
    for img, ax in zip(imgs, axs):
        if print_title:
            ax.set_title(titles[count])
            count += 1
        if gray:
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img)
    plt.show()


# }}}
def normalize_mat(mat, minimum="mean"):  # {{{
    if minimum == "zero":
        return (mat - mat.min()) / (mat.max() - mat.min())
    if minimum == "mean":
        return (mat - mat.mean()) / (mat.max() - mat.min())


# }}}
def map_mat_255(img):  # {{{
    img = img.astype(np.float)
    if (img == 0).all():
        return img.astype(np.uint8)
    result = normalize_mat(img, minimum="zero") * 255
    return result.astype(np.uint8)


# }}}
def img_channel(img):  # {{{
    if len(img.shape) == 3:
        return img.shape[2]
    if len(img.shape) == 2:
        return 1


# }}}
def equal_multiChannel(mat, template):  # {{{
    """
    To find template in a mat
    tamplate must be 1D and len(template) == mat.shape[-1]
    mat is a 2D image with multiple channels
    """
    template = np.array(template)
    if not (len(template.shape) == 1 and len(template) == mat.shape[-1]):
        raise Exception("illegal parameter, see --help")
    x = mat == template
    bools = [x[:, :, i] for i in range(len(template))]
    result = np.ones(mat.shape[:2], np.bool)
    for bl in bools:
        result = np.logical_and(result, bl)
    return result


# }}}
def overlap_(fg_img, bg_img, mask):  # {{{
    """
    按蒙版重叠图像fg_img和bg_img，fg_img在蒙版的白色区bg_img在黑色区
    @fg_img: 前景图，三通道
    @bg_img: 背景图，三通道
    @mask: 蒙版，最大值为1, 三通道
    三图要有相同大小
    """
    # mask = gray2rgb(mask)

    new_img = fg_img * mask + bg_img * (1 - mask)
    return new_img


# }}}
def returned_img(patch, img, pos):  # {{{
    """
    return an image patch into the original image
    pos: upper left corner (row, col)
    """
    if img_channel(img) == 1:
        img = gray2rgb_(img)
    if img_channel(patch) == 1:
        patch = gray2rgb_(patch)
    patch_h = patch.shape[0]
    patch_w = patch.shape[1]
    img[pos[0] : pos[0] + patch_h, pos[1] : pos[1] + patch_w] = patch
    return img


# }}}
def get_region(img, h_range, w_range):  # {{{
    """
    fetch a image patch without worrying about get out of the image dimension
    """
    return [
        [max(0, h_range[0]), min(h_range[1], img.shape[0])],
        [max(0, w_range[0]), min(w_range[1], img.shape[1])],
    ]


# }}}
def find_region(mask, value=1):  # {{{
    """Find the row & col image region that the mask == value"""
    coord = np.where(mask == value)
    row_start = coord[0].min()
    row_end = coord[0].max()
    col_start = coord[1].min()
    col_end = coord[1].max()
    return ((row_start, row_end), (col_start, col_end))


# }}}
def find_max_areaContour(img, approx=cv.RETR_TREE):  # {{{
    """
    用opencv的contour找到最大面积的轮廓
    返回轮廓
    """
    contours, _ = cv.findContours(img, cv.RETR_TREE, approx)

    max_area_id = 0  # 记录最大轮廓的id
    max_area = 0  # 记录最大长度
    for i in range(len(contours)):
        current_area = cv.contourArea(contours[i])
        if current_area > max_area:
            max_area = current_area
            max_area_id = i
    return contours[max_area_id]


# }}}
def resampleSpacing(imgs, old_spacing, new_spacing=[1, 1, 1]):  # {{{
    """Resample /dicom/ images"""
    spacing = np.array(old_spacing)
    resize_factor = spacing / new_spacing
    new_real_shape = imgs.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / imgs.shape
    new_spacing = spacing / real_resize_factor
    imgs = scipy.ndimage.interpolation.zoom(imgs, real_resize_factor)
    return imgs, new_spacing


# }}}
def overlap_mask(img, mask, color=(255, 0, 0), alpha=1):  # {{{
    if img_channel(img) == 1:
        img = gray2rgb_(img)
    if img_channel(mask) == 1:
        mask = gray2rgb_(mask)
    im = img.astype(float)
    channel = np.ones(img.shape[:2], np.float)
    color_ = np.concatenate(
        (
            channel[:, :, np.newaxis] * color[0],
            channel[:, :, np.newaxis] * color[1],
            channel[:, :, np.newaxis] * color[2],
        ),
        axis=2,
    )
    f_im = im * (1 - mask) + im * mask * (1 - alpha) + color_ * alpha * mask
    return f_im.astype(np.uint8)


# }}}
class Interpolate_mask_init:  # {{{
    """
    get initial contour from two masks: msk1, msk2
    @msk1, msk2: masks (binary)
    @num_interp: # of masks to interpolate
    """

    def __init__(self, msk1, msk2, num_interp, accuracy=100):
        if msk1.shape != msk2.shape:
            raise Exception("Unmatched shape for two images, check image demension")
        elif num_interp < 1:
            raise Exception("Cannot interpolate sccessive masks  ")

        self.msk1 = msk1.astype(np.uint8)
        self.msk2 = msk2.astype(np.uint8)
        self.msks = None
        self.num = num_interp
        self.cnt1 = find_max_areaContour(self.msk1, cv.CHAIN_APPROX_NONE)
        self.cnt2 = find_max_areaContour(self.msk2, cv.CHAIN_APPROX_NONE)
        self.cnt1_unif = self.resample(self.cnt1, accuracy)  # uniformed
        self.cnt2_unif = self.resample(self.cnt2, accuracy)

    def run(self):
        NI = self.num
        weights = [((NI - pos) / (NI + 1), (pos + 1) / (NI + 1)) for pos in range(NI)]
        cnts = [
            (w1 * self.cnt1_unif + w2 * self.cnt2_unif).astype(int)
            for w1, w2 in weights
        ]
        self.msks = [self.get_mask(cnt, self.msk1.shape[:2]) for cnt in cnts]
        return self.msks

    # =========================Utils===================================
    def resample(self, ori_seq, length):
        """
        index_seq: range sequence
        """
        id_index = range(length)
        id_corr_ori = (len(ori_seq) - 1) / (length - 1) * np.array(id_index)
        new_seq = []
        for i in id_corr_ori:
            if i == int(i):
                new_seq.append(ori_seq[int(i)])
            else:
                floor = max((np.floor(i)).astype(int), 0)
                ceil = min((np.ceil(i)).astype(int), len(ori_seq) - 1)
                val = (i - floor) * ori_seq[ceil] + (ceil - i) * ori_seq[floor]
                new_seq.append(val)
        return np.array(new_seq)

    def get_mask(self, cnt, shape):
        img = np.zeros(shape, np.uint8)
        cv.fillPoly(img, pts=[cnt], color=1)
        return img


# }}}
def otsuThresh(arr):  # {{{{{{
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    min_value = arr.min()
    arr = arr - min_value

    p_dic = dict()
    for px in arr:
        try:
            p_dic[px] += 1
        except:
            p_dic[px] = 0
    values = np.sort(np.unique(arr))
    count = np.zeros(values.shape, np.int)
    for i in range(len(values)):
        count[i] = p_dic[values[i]]

    p = count / count.sum()  # possibility
    ip = p * values
    mu_total = ip.sum()
    var = np.zeros(values.shape, np.int)  # between class variance
    for i in range(len(values)):
        if i == 0:
            sum_ip = ip[0]
            w0 = p[0]
        else:
            sum_ip += ip[i]
            w0 += p[i]
        w1 = 1 - w0
        mu0 = sum_ip / (w0 + 0.000000001)
        mu1 = (mu_total - sum_ip) / (w1 + 0.000000001)
        var[i] = w0 * w1 * (mu1 - mu0) ** 2
    thresh = np.where(var == np.amax(var))[0][0]
    return thresh + min_value


# }}}
