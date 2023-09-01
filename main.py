import gradio as gr
from pathlib import Path
from modelhandlers import *
mname = get_smallest_model()[1]
model = CformerModel(mname)

params = {
    "max_new_tokens": 75,
    "top_k": 50,
    "top_p": 0.9,
    "temperature": 0.5,
    "repetition_penalty": 1.0,
    "threads": 6
}
def inference(text, shoudappend, new):
    global model, params
    params.update({"max_new_tokens": new})
    return model.__call__(
        text,
        params,
        shoudappend
    )
def wrapdownload(url, name, areyousure, arch, temporary=False):
    if areyousure:
        return download_model(url, arch, name, temporary)
    
def get_models():
    fullname = Path("./models").glob("*.bin")
    sname = []
    for i in fullname:
        sname.append(i)
    return sname
#gradio interface もふ
# with gr.Blocks(analytics_enabled=False, theme=random.choice(["gstaff/xkcd", "ParityError/Anime"])) as demo:
with gr.Blocks(analytics_enabled=False, theme="NoCrypt/miku") as demo:
    gr.Markdown("# mofu-webui | もふ インタフェース")
    with gr.Tab("Inference"):
        with gr.Column():
            text = gr.Textbox(label="input text", lines=20)
        with gr.Column():
            modeloutput = gr.Textbox(label="output text", lines=15)
            with gr.Row():
                infer = gr.Button("Inference")
                shoudappend = gr.Checkbox(label="append output to input?", value=False)
                maxnewtokens = gr.Slider(minimum=20, maximum=250, step=5, label="max new tokens", value=75)
                infer.click(inference, inputs=[text, shoudappend, maxnewtokens], outputs=[modeloutput, text])
    with gr.Tab("Model"):
        with gr.Row():
            drop = gr.Dropdown(label="model", choices=get_models(), multiselect=False, value=model.originalname)
            hiddenCheckbox = gr.Checkbox(label="switch model", value=False, visible=False)
            refreshlist = gr.Button("refresh list")
            refreshlist.click(
                fn=lambda: drop.update(choices=get_models(), value=model.originalname), 
                outputs=[drop]
            )
            drop.input(model.switch_model, inputs=[drop], outputs=[hiddenCheckbox])
        with gr.Row():
            gr.Markdown("## Model generation parameters")
            top_k = gr.Slider(minimum=0, maximum=100, step=1, label="top k", value=50, interactive=True)
            top_p = gr.Slider(minimum=0, maximum=1, step=0.05, label="top p", value=0.9, interactive=True)
            temperature = gr.Slider(minimum=0, maximum=1, step=0.05, label="temperature", value=0.5, interactive=True)
            repetition_penalty = gr.Slider(minimum=0, maximum=2, step=0.1, label="repetition penalty", value=1.0, interactive=True)
            #on change of top_k
            top_k.input(
                fn=lambda: params.update({"top_k": top_k.value}),
            )
            #on change of top_p
            top_p.input(
                fn=lambda: params.update({"top_p": top_p.value}),
            )
            #on change of temperature
            temperature.input(
                fn=lambda: params.update({"temperature": temperature.value}),
            )
            #on change of repetition_penalty
            repetition_penalty.input(
                fn=lambda: params.update({"repetition_penalty": repetition_penalty.value}),
            )
    with gr.Tab("Download"):
        with gr.Row():
            with gr.Column():
                modelurl = gr.Textbox(label="new model url", value="", placeholder="https://huggingface.co/ggerganov/ggml/resolve/main/ggml-model-gpt-2-117M.bin")
                modelname = gr.Textbox(label="new model name", value="", placeholder="gpt2-tiny")
                modelarch = gr.Dropdown(label="model arch", choices=["gpt2", "gptj", "gpt_neox", "falcon", "llama", "mpt", "gpt_bigcode", "dolly-v2", "replit"], value="gpt2")
            with gr.Column():
                modeldownload = gr.Button("download model")
                areyousure = gr.Checkbox(label="are you sure?", value=False)
                
            modeldownload.click(
                wrapdownload,
                inputs=[modelurl, modelname, areyousure, modelarch],
            )

demo.launch()