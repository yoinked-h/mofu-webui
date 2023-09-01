import ctransformers, requests
from pathlib import Path
import toml
BASECFG = {
    "max_new_tokens": 75,
    "top_k": 50,
    "top_p": 0.9,
    "temperature": 0.5,
    "repetition_penalty": 1.0,
    "threads": 4
}
def download_model(url, arch, name=None, temp=False):
    if name is None:
        # get name from url
        name = url.split("/")[-1]
    whereto = "models"
    if temp:
        whereto = "temp"
    if not name.endswith(".bin"):
        name += ".bin"
    with open(Path(f"./{whereto}/{name}"), "wb") as f:
        f.write(requests.get(url).content)
    with open(Path(f"./{whereto}/{name}.toml"), "w") as f:
        f.write(f"model_arch = \"{arch}\"")
    return

def get_smallest_model():
    allof = Path("./models").glob("*.bin")
    models = []
    for i in allof:
        #get file size
        size = Path(i).stat().st_size
        models.append((size, i))
    try:
        return min(models, key=lambda x: x[0])
    except ValueError:
        download_model("https://huggingface.co/ggerganov/ggml/resolve/main/ggml-model-gpt-2-117M.bin", "gpt2", "gpt2")
        get_smallest_model()
class CformerModel:
    def __init__(self, model: str, gen_cfg:dict = None):
        #sanity check
        model = str(model)
        self.originalname = model
        self.gen_cfg = gen_cfg if gen_cfg is not None else BASECFG
        #open the toml located at "{modelname}.toml"
        with open(f"{model}.toml", "r") as f:
            self.cfg = toml.load(f)
        for lib in ["avx2", "avx", "basic"]:
            try:
                self.model = ctransformers.AutoModelForCausalLM.from_pretrained(f"{model.replace('.bin', '')}.bin", lib=lib, model_type=self.cfg["model_arch"], local_files_only=True)
                break
            except:
                pass
        self.human_name = f"{model.replace('.bin', '').replace('_', ' ')}"
    def __call__(self, prompt, cfg:dict = dict(), append:bool = False):
        config = BASECFG
        config.update(self.gen_cfg)
        config.update(cfg)
        n = self.model(
            prompt,
            max_new_tokens=config["max_new_tokens"],
            top_k=config["top_k"],
            top_p=config["top_p"],
            temperature=config["temperature"],
            repetition_penalty=config["repetition_penalty"],
            threads=config["threads"],
        )
        if append:
            prompt = prompt + n
        return n, prompt
    def switch_model(self, model_to):
        try:
            del self.model
        except AttributeError:
            self.model = None
        model_name = str(model_to)
        print("loading model", model_name)
        if model_name == self.originalname:
            return True
        with open(f"{model_name}.toml", "r") as f:
            self.cfg = toml.load(f)
        for lib in ["avx2", "avx", "basic"]:
            try:
                self.model = ctransformers.AutoModelForCausalLM.from_pretrained(f"{model_name.replace('.bin', '')}.bin", lib=lib, model_type=self.cfg["model_arch"], local_files_only=True)
                break
            except:
                pass
        model_to = Path(model_to).name
        self.originalname = model_to
        print("loaded model", self.originalname)
        return True
        