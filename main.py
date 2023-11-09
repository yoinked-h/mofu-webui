"""mofu webui"""
from pathlib import Path

import argparse
import requests
import toml
import gradio as gr

import custom_modules as cm
model, params = None, None
cm.import_all_custom_modules_needed()
def get_smallest_model():
    """Get the smallest model in the models directory."""
    models = []
    for path in Path("./models").glob("*.*"):
        if path.suffix == ".toml":
            continue
        size = path.stat().st_size
        models.append((size, path))
    return min(models, key=lambda x: x[0])
def initialize():
    """Initialize the model and parameters."""
    global model, params # pylint: disable=global-statement
    mname = get_smallest_model()[1]
    model_class = cm.get_required_model_class(mname)
    model = model_class(mname)
    params = {
        "max_new_tokens": 75,
        "top_k": 50,
        "top_p": 0.9,
        "temperature": 0.5,
        "repetition_penalty": 1.0,
    }
initialize()
def inference(text, new, should=False):
    """Generate text using the model."""
    params.update({"max_new_tokens": 1})
    x = text
    for n in range(new):
        txt = model.generate(x, params)
        x += txt
        if should:
            yield x, x
        else:
            yield x, text
    
def download(modurl, name, really, arch, modname):
    """Download a model file."""
    print(modurl)
    if really:
        name = name or modurl.split("/")[-1]

        name = name if name.endswith(".*") else name + ".bin"
        with open(Path(f"./models/{name}"), "wb") as _:
            _.write(requests.get(modurl, timeout=30).content)

        with open(Path(f"./models/{name}.toml"), "w", encoding="utf-8") as _:
            prm = {"model_arch": arch, "module": modname}
            toml.dump(prm, _)
def swap_models(modeloc):
    """Switch to another model."""
    global model # pylint: disable=global-statement
    model_class = cm.get_required_model_class(modeloc)
    model = model_class(modeloc)
    return True
def get_models():
    """
    Returns a list of all the model files in the `models` directory.
    """
    fullname = Path("./models").glob("*.*")
    sname = []
    for i in fullname:
        if i.suffix == ".toml":
            continue
        sname.append(i)
    return sname

with gr.Blocks(analytics_enabled=False, theme="NoCrypt/miku") as demo:
    gr.Markdown("# mofu-webui | もふ インタフェース")
    with gr.Tab("Inference"):
        with gr.Column():
            prompt = gr.Textbox(label="input text", lines=20)
        with gr.Column():
            modeloutput = gr.Textbox(label="output text", lines=15,)
            with gr.Row():
                infer = gr.Button("Inference")
                append = gr.Checkbox(label="append output?", interactive=True)
                maxnewtokens = gr.Slider(minimum=20, maximum=250, step=5,
                                        label="max new tokens", value=75)
        infer.click(inference, inputs=[prompt, maxnewtokens, append], outputs=[modeloutput, prompt])
    with gr.Tab("Model"):
        with gr.Row():
            drop = gr.Dropdown(label="model", choices=get_models(),
                            multiselect=False, value=model.originalname, allow_custom_value=True)
            hiddenCheckbox = gr.Checkbox(label="switch model", value=False, visible=False)
            refreshlist = gr.Button("refresh list")
            refreshlist.click(
                fn=lambda: drop.update(choices=get_models(), value=model.originalname),
                outputs=[drop]
            )
            drop.input(swap_models, inputs=[drop], outputs=[hiddenCheckbox])
        with gr.Row():
            gr.Markdown("## Model generation parameters")
            top_k = gr.Slider(minimum=0, maximum=100, step=1, label="top k",
                            value=50, interactive=True)
            top_p = gr.Slider(minimum=0, maximum=1, step=0.05, label="top p",
                            value=0.9, interactive=True)
            temperature = gr.Slider(minimum=0, maximum=1, step=0.05, label="temperature",
                                    value=0.5, interactive=True)
            repetition_penalty = gr.Slider(minimum=0, maximum=2, step=0.1,
                                label="repetition penalty", value=1.0, interactive=True)
            top_k.input(
                fn=lambda: params.update({"top_k": top_k.value}),
            )
            top_p.input(
                fn=lambda: params.update({"top_p": top_p.value}),
            )
            temperature.input(
                fn=lambda: params.update({"temperature": temperature.value}),
            )
            repetition_penalty.input(
                fn=lambda: params.update({"repetition_penalty": repetition_penalty.value}),
            )
    with gr.Tab("Download"):
        with gr.Row():
            with gr.Column():
                url = gr.Textbox(label="new model url", value="", placeholder=
                    "https://huggingface.co/ggerganov/ggml/resolve/main/ggml-model-gpt-2-117M.bin")
                modelname = gr.Textbox(label="new model name", value="", placeholder="gpt2-tiny")
                modelarch = gr.Dropdown(label="model arch", value="gpt2",
                                        choices=["gpt2", "gptj", "gpt_neox", "falcon", "llama",
                                                "mpt", "gpt_bigcode", "dolly-v2", "replit"])
                modulename = gr.Dropdown(label="module name", choices=["ctransformers"],
                                        value="ctransformers", allow_custom_value=True)
            with gr.Column():
                modeldownload = gr.Button("download model")
                areyousure = gr.Checkbox(label="are you sure?", value=False)
            modeldownload.click(
                download,
                inputs=[url, modelname, areyousure, modelarch, modulename],
            )

parser = argparse.ArgumentParser()
parser.add_argument('--share', '-s', action='store_true', default=False)
parser.add_argument('--api', '-a', action='store_true', default=False)
args = parser.parse_args()
demo.queue(max_size=20).launch(share=args.share, show_api=args.api, max_threads=1)
