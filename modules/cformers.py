"""implement ctransformers as a custom module"""
from ctransformers import AutoModelForCausalLM #pylint: disable=import-error
import toml
from modules.basemodel import Model
BASECFG = {
    "max_new_tokens": 75,
    "top_k": 50,
    "top_p": 0.9,
    "temperature": 0.5,
    "repetition_penalty": 1.0,
    "threads": 4
}


class CformerModel(Model):
    """Model class for ggml/ctransformers based models."""
    def __init__(self, model: str):
        #sanity check
        model = str(model)
        self.originalname = model
        self.gen_cfg = BASECFG
        #open the toml located at "{modelname}.toml"
        with open(f"{model}.toml", "r", encoding="utf-8") as _:
            self.cfg = toml.load(_)
            if "extra_data" in self.cfg.keys():
                self.gpulayers = self.cfg["extra_data"]["gpu_layers"]
            else:
                self.gpulayers = 0
        for lib in ["avx2", "avx", "basic"]:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(model,
                                lib=lib, model_type=self.cfg["model_arch"], local_files_only=True,
                                gpu_layers=self.gpulayers)
                break
            except FileNotFoundError:
                pass
            finally:
                pass
        self.human_name = f"{model.replace('.bin', '').replace('_', ' ')}"
    def __call__(self, prompt, cfg:dict = None) -> str:
        if cfg is None:
            cfg = {}
        config = BASECFG.copy()
        config.update(self.gen_cfg)
        config.update(cfg)
        new_tokens = self.model(
            prompt,
            max_new_tokens=config["max_new_tokens"],
            top_k=config["top_k"],
            top_p=config["top_p"],
            temperature=config["temperature"],
            repetition_penalty=config["repetition_penalty"],
            threads=config["threads"]
        )
        return new_tokens
    def generate(self, *args, **kwargs):
        return self(*args, **kwargs)
    def switch_model(self, model_name):
        self.model = None
        if model_name == self.originalname:
            return True
        with open(f"{model_name}.toml", "r", encoding="utf-8") as _:
            self.cfg = toml.load(_)
            if "extra_data" in self.cfg.keys():
                self.gpulayers = self.cfg["extra_data"]["gpu_layers"]
            else:
                self.gpulayers = 0
        libraries = ["avx2", "avx", "basic"]
        for lib in libraries:
            try:
                model_file = f"{model_name.replace('.bin', '')}.bin"
                self.model = AutoModelForCausalLM.from_pretrained(model_file, lib=lib,
                                model_type=self.cfg["model_arch"], local_files_only=True,
                                gpu_layers=self.gpulayers)
                break
            except FileNotFoundError:
                pass
            finally:
                pass
        self.originalname = model_name
        return True

MODEL_NAME = "CformerModel"
MODEL = [CformerModel]
