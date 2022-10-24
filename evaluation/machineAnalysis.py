import os, re, json
from typing import Dict, List, Set, Tuple, Union
import pydicom.errors
import pydicom.dataset
import pydicom.dataelem
from pydicom import dcmread
import numpy as np
from monsoonToolBox.filetools import pJoin, subDirAndFiles, subDirs, subFiles
from monsoonToolBox.misc import lisJobParallel, lisJobParallel_
from monsoonToolBox.logtools import Timer
from immarker.sc import showRGBIms
from labelSys.utils.labelReaderV2 import LabelSysReader

DICOM_DIR = "/home/monsoon/Documents/Data/TMJ-ML/drive-download-20220322T053732Z-001"
LABELED_DATA_DIR = "/home/monsoon/Documents/Data/TMJ-ML/Data_total"
JSON_SAVE_PATH = "/home/monsoon/Documents/Data/TMJ-ML/machine.json"
LABEL_UID_JSON_PATH = "/home/monsoon/Documents/Data/TMJ-ML/label_uid.json"


class PatientData:
    def __init__(self, idx: int, dir_path: str):
        self.idx: int = idx
        self.dir_path: str = dir_path
        self.ds: List[pydicom.dataset.FileDataset] = []

    def meta(self, tag0: int, tag1: int) -> pydicom.dataelem.DataElement:
        try:
            out = self.ds[0][tag0, tag1]
        except KeyError:
            # 0x0021 private tag
            out = pydicom.dataelem.DataElement(
                (0x0021, 0x0001), VR="UN", value="Unknown"
            )
        return out

    @property
    def arrays(self) -> Dict[str, List[np.ndarray]]:
        out = {}
        for d in self.ds:
            series = self.getSeriesDescription(d)
            out.setdefault(series, []).append(d.pixel_array)
        return out

    @property
    def series_descriptions(self) -> Set[str]:
        out = set()
        for d in self.ds:
            series = self.getSeriesDescription(d)
            out.add(series)
        return out

    @staticmethod
    def getSeriesDescription(ds: pydicom.dataset.FileDataset) -> str:
        if "SeriesDescription" in ds:
            seriesDescript = ds["SeriesDescription"].value
            seriesDescript = str(seriesDescript)
        else:
            try:
                seriesDescript = str(ds[0x0020, 0x0011].value)  # Series Number
            except KeyError:
                seriesDescript = "UnKnown"
        return seriesDescript

    @property
    def uids(self):
        uids = []
        for d in self.ds:
            # SOPInstanceUID
            uid = d[0x0008, 0x0018].value
            uids.append(str(uid))
        return uids

    @property
    def info(self) -> dict:
        return {
            "id": self.idx,
            "series_descriptions": self.series_descriptions,
            "modality": self.meta(0x0008, 0x0060).value,
            "modality_in_study": self.meta(0x0008, 0x0061).value,
            "equipment_modality": self.meta(0x0008, 0x0221).value,
            #  "ManufacturerRelatedModelGroup": self.meta(0x0008, 0x0222),
            "manufacture": self.meta(0x0008, 0x0070).value,
            "manufacture_model": self.meta(0x0008, 0x1090).value,
            "institution_name": self.meta(0x0008, 0x0080).value,
            "device_serial_number": self.meta(0x0018, 0x1000).value,
        }

    def __str__(self):
        out_lines = []
        for k, v in self.info.items():
            out_lines.append("{}: {}".format(k, v))
        return "\n".join(out_lines)

    __repr__ = __str__


def findAllPatients(paths: List[str]) -> List[PatientData]:
    all_patients: List[PatientData] = []
    for p in paths:
        idx = int(os.path.basename(p))
        ps = findPatients(p, idx)
        all_patients += ps
    return all_patients


def findPatients(f_path: str, idx: int) -> List[PatientData]:
    out: List[PatientData] = []
    p_data = PatientData(idx, f_path)
    for f in subFiles(f_path):
        try:
            slic = dcmread(f)
            seriesDescript = p_data.getSeriesDescription(slic)
            #  if re.search("SAG *PD.*", seriesDescript):
            #      p_data.ds.append(slic)
            p_data.ds.append(slic)
        except pydicom.errors.InvalidDicomError:
            #  print("invalid files: {}".format(f))
            continue

    if p_data.ds:
        out.append(p_data)

    for d in subDirs(f_path):
        out += findPatients(d, idx)
    return out


def groupMachines(patient_data: List[PatientData]):
    grouping = dict()
    for p in patient_data:
        serial_number = p.info["device_serial_number"]
        manufacture = p.info["manufacture"]
        manufacture_model = p.info["manufacture_model"]
        total = ".".join((manufacture, manufacture_model, serial_number))
        grouping.setdefault(total, []).append(p)
    return grouping


def groupMachines_(patient_data: List[PatientData]):
    return [groupMachines(patient_data)]


def mergeDicts(*dicts: dict):
    new_d = {}
    for d in dicts:
        for k, v in d.items():
            new_v = new_d.setdefault(k, [])
            new_d[k] += new_v
    return new_d


def generateMachineTable():
    global DICOM_DIR, LABELED_DATA_DIR
    print("Loading...")
    valid_patients = lisJobParallel_(findAllPatients, subDirs(DICOM_DIR))
    #  groups = lisJobParallel(groupMachines, valid_patients, True)
    group = groupMachines(valid_patients)
    print("Total valid DICOM: ", len(valid_patients))

    reader = LabelSysReader(subDirs(LABELED_DATA_DIR))
    print("Reading labels...")
    label_uids: Dict[str, List[str]] = dict()
    if not os.path.exists(LABEL_UID_JSON_PATH):
        for i in range(len(reader)):
            base_name = os.path.basename(reader[i].path)
            label_uids[base_name] = reader[i].uids
        with open(LABEL_UID_JSON_PATH, "w") as fp:
            json.dump(label_uids, fp, indent=1)
    else:
        with open(LABEL_UID_JSON_PATH, "r") as fp:
            label_uids = json.load(fp)

    label_machine: Dict[str, str] = {}
    for machine, patients in group.items():
        for p in patients:
            find_ = False
            for label_fname, uids in label_uids.items():
                print(label_fname)
                if find_:
                    break
                for _uid in p.uids:
                    if _uid in uids:
                        label_machine[label_fname] = machine
                        find_ = True
                        break
    # print missing data:
    print("Missing DICOM: ")
    for f in os.listdir(LABELED_DATA_DIR):
        if f not in label_machine.keys():
            print(f)
            label_machine[f] = ""

    with open(JSON_SAVE_PATH, "w") as fp:
        json.dump(label_machine, fp, indent=1)


if __name__ == "__main__":
    #  with Timer("Paralleled_reading"):
    #      valid_patients = lisJobParallel_(findAllPatients, subDirs(DICOM_DIR)[::1])
    #  for v in valid_patients:
    #      print(v)
    #  group = groupMachines(valid_patients)
    #  for k in group.keys():
    #      print("{} - count: {}".format(k, len(group[k])))
    generateMachineTable()
