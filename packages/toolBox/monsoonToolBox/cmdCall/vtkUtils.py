from typing import List
import argparse
import vtk
import sys

def _findModule(module_name: str):
    m = getattr(vtk, module_name)
    return m.__module__

def findVTKModules(module_list: List[str]):
    modules = dict()
    for mod in module_list:
        module_path = _findModule(mod)
        modules.setdefault(module_path, []).append(mod)
    return modules

def callFindVTKModules():
    parser = argparse.ArgumentParser(description="Find VTK modules position for importing.")
    parser.add_argument("m", nargs="+", help = "VTK module name(s)")
    args = parser.parse_args()
    module_names = args.m
    mods = findVTKModules(module_names)
    for module_path, _modules in mods.items():
        print("from {} import {}".format(module_path, ", ".join(set(_modules))))

if __name__ == "__main__":
    callFindVTKModules()
