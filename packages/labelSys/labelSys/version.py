# +==***---------------------------------------------------------***==+ #
# |                                                                   | #
# |  Filename: version.py                                             | #
# |  Copyright (C)  - All Rights Reserved                             | #
# |  The code presented in this file is part of an unpublished paper  | #
# |  Unauthorized copying of this file, via any medium is strictly    | #
# |  prohibited                                                       | #
# |  Proprietary and confidential                                     | #
# |  Written by Mengxun Li <mengxunli@whu.edu.cn>, June 2022          | #
# |                                                                   | #
# +==***---------------------------------------------------------***==+ #
#
# Copyright (c) 2020-2022 Mengxun Li.
#
# This file is part of LabelSys
# (see ...).
#

__author__ = "Mengxun Li"
__copyright__ = "Copyright 2020-2022"
__credits__ = ["Kumaradevan Punithakumar"]
__license__ = "All right reserved"
__maintainer__ = "Mengxun Li"
__email__ = "mengxunli@whu.edu.cn | mengxun1@ualberta.ca"

__license_full__ = "\
Copyright (c) 2020-2022, Mengxun Li\n\
All rights reserved.\n\
\n\
Redistribution and use in source and binary forms, with or without\n\
modification, are not permitted except that permission from the copyright holder\n\
was obtained and the following conditions are met:\n\
\n\
1. Redistributions of source code must retain the above copyright notice, this\n\
   list of conditions and the following disclaimer.\n\
\n\
2. Redistributions in binary form must reproduce the above copyright notice,\n\
   this list of conditions and the following disclaimer in the documentation\n\
   and/or other materials provided with the distribution.\n\
\n\
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS \"AS IS\"\n\
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE\n\
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE\n\
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE\n\
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL\n\
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR\n\
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER\n\
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,\n\
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE\n\
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."

__VERSIONS__ = [
    ["0.X", "Implemented with Tkinter"],
    ["1.0-alpha", "Re-write the whole program using PyQt5 and vtk, under develpment"],
    ["1.0.0", "For MRI labeling - disc, condyle and eminence"],
    [
        "1.0.1",
        "Fix curve initialization crash bug; Add individual curve initialization step; Add image/video loading support",
    ],
    [
        "1.1.0",
        "Speed up loading and exporting process; Change data storing header file rule to include configration; Add compare widget; Add labeling panel preview; Change version naming rule",
    ],
    ["1.2.0", "Support Color image reading, license change"],
    [
        "1.2.1",
        "Bug fix - label interpretation error when dealing with non-square image",
    ],
    [
        "1.2.2",
        "Now support color panel preview, cursor can move out of the image when labeling; Add max_im_height to config file",
    ],
    [
        "1.3.0",
        "Add rotation function oin operation, support default label, support reverse switch label",
    ],
    ["1.3.1", "Add on-panel label"],
    ["1.3.2", "Bug fix: rotate will not clear all labels"],
    [
        "1.3.3",
        "Add known issue into source code, add preview on panel shortcut and manual button, preview on panel will update on contour end interaction",
    ],
    [
        "1.3.4",
        "Change interpolation method to equally-spaced points on curvature integral.",
    ],
    [
        "1.3.5",
        "Liscense change; Add load file selection to console args; Default label can be set to \"\" to prevent label change while changing slices; Not raise error when select incorrect loading directory.",
    ],
    ["1.4.0", "Using setup.py for distribution"],
    ["1.4.1", "Bug fix - relative path"],
    ["1.5.0", "UI updates, add comment functionality"],
    ["1.5.1", "Performance optimization with better color image loading."],
    ["1.5.2", "Resample stratagy update"],
    ["1.5.3", "Add classification functionality"],
    [
        "1.5.4",
        "Saving format change, now using .npz for image saving; Add dtype to on-panel img info",
    ],
    ["1.5.5", "Add -c to argparse"],
    [
        "1.6.0",
        "Add new click interaction style, new labelReader API, more entry on config file",
    ],
    ["1.6.1", "Set configure file as optional positional argument"],
    ["1.6.2", "Able to generate default conf; double the default label step;"],
    [
        "1.6.3",
        "Add 'to another label' in 'operation', old label loading compatability; set default labeler name to platform.node()",
    ],
    ["1.6.4", "LabelReader API update, doc-zh update"],
    ["1.6.5", "Add License to help menu"],
    ["1.6.6", "Logfile update"],
    [
        "1.7.0",
        "Move SOPInstanceUID out of Data when saving, use uid instead, move logfile to $HOME. Bug fix: not enable widgets when open files after loading",
    ],
    ["1.7.1", "Better debugging when open general image"],
    ["1.7.2", "Bug fix: opencv read image path contain 中文"],
    [
        "1.7.3",
        "Delete vim markers, bug fix: crash when opening DICOM from CLI. support for DICOM without series and InstanceNumber",
    ],
    [
        "1.7.4",
        "More accurate labeling by vtk-cv coordinate transform; Mark labeled folder in qlabeltext",
    ],
    [
        "1.8.0",
        "Add crop and rotate window (achieved with immarker project); Bug fix: classification not update when reopen classification GUI; Change classification config file naming: short_title -> full_name ",
    ],
    ["1.8.1", "Interaction style change, use space to move image"],
    ["1.8.2", "Bug fix: deal with image change while crop panel being opened"],
]
__version__, __description__ = __VERSIONS__[-1]
