"""handle all modules using importlib"""
import importlib
from pathlib import Path

import toml
cmods = []
mapping = {}

#to whoever has to improve this code, i am so sorry

def import_all_custom_modules_needed():
    """import all custom modules"""
    global cmods # pylint: disable=global-statement
    cfgs = []
    for config in Path("./models").glob("*.toml"):
        with open(config, "r", encoding="utf-8") as _:
            cfgs.append(toml.load(_))
    for config in cfgs:
        cmods.append(config["module"])
    cmods = list(set(cmods))
    for cmod in cmods:
        _tmod = importlib.import_module(f"modules.{cmod}")
        ali = _tmod.MODEL_NAME
        mapping[ali] = _tmod.MODEL[0]
def import_custom_module(cmod):
    """get the model class for the module called `cmod`"""
    _tmod = importlib.import_module(f"modules.{cmod}")
    ali:str = _tmod.MODEL_NAME
    return mapping[ali]
def get_required_model_class(modpath:Path):
    """get the model class for the model at `modpath`"""
    with open(modpath.with_suffix("".join(modpath.suffix+".toml")), "r", encoding="utf-8") as _:
        _cfg = toml.load(_)
    _cfg['module'] = modpath
    return import_custom_module("ctransformers")
