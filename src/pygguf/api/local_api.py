import requests
import subprocess
from pathlib import Path
import time
import json
from http.client import responses
import io
from pygguf.api.img_utils import image_to_url, process_image
import webbrowser


from pygguf.api.settings import LLAMAEXE, MODELS, HOME, DATA_PATH


OAI_ENDPOINT = "/v1/chat/completions"
LLAMA_ENDPOINT = "/completion"


def load_json(fpath: Path) -> dict:
    with io.open(fpath, "r", encoding="utf8") as f:
        jobj = json.loads(f.read())
    return jobj


def open_grammar(filename: str) -> str:
    with io.open(
        Path(HOME, "../grammars", f"{filename}.gbnf"), "r", encoding="utf8"
    ) as f:
        s = f.read()
    return s


def moving_dots(n: int, N: int) -> str:
    s = "." * n
    return s.ljust(N)


def model_fpath(model_name: str) -> Path:
    return Path(DATA_PATH, rf"models\{model_name}").absolute()


def launch_server(
    port: int = 8080,
    ctx: int = int(2**13),
    verbose: bool = False,
    model_name: str = "gemma",
    open_browser: bool = False,
):
    exe = LLAMAEXE

    if verbose:
        kwargs = {"stdout": subprocess.PIPE}
    else:
        kwargs = {"stderr": subprocess.DEVNULL, "stdout": subprocess.DEVNULL}

    mmproj_model = None
    if model_name == "gemma":
        model_name = "gemma/gemma-3-4b-it-Q4_K_M.gguf"
        mmproj_model = "gemma/mmproj-F16.gguf"
    elif model_name == "smolvlm":
        model_name = "smolvlm/SmolVLM-Instruct-Q8_0.gguf"
        mmproj_model = "smolvlm/mmproj-SmolVLM-Instruct-Q8_0.gguf"

    model = model_fpath(model_name)
    if mmproj_model is not None:
        mmproj = f"--mmproj {model_fpath(mmproj_model)}"
    else:
        mmproj = ""
    cmd = f"{exe} -m {model} --port {port} --offline -c {ctx} {mmproj} -ngl 99"

    print(cmd)

    server = subprocess.Popen(cmd, **kwargs)
    host = f"http://localhost:{port}"
    r = requests.get(host)
    n = 0
    n_dots = 5
    while r.status_code == 503:
        try:
            time.sleep(0.2)
            r = requests.get(host)
            if not verbose:
                print(
                    f"Status code {r.status_code} ({responses[r.status_code]}){moving_dots(n,n_dots)} on localhost:{port} model: {model_name}",
                    end="\r",
                )
            n = (n + 1) % n_dots
        except Exception:
            time.sleep(0.2)
            pass
    print("\n")

    if open_browser:
        print(f"Opening {host}")
        webbrowser.open(host)


def build_payload_oai(
    prompt_msg: str, image: Path, system_prompt: str, json_schema: str
) -> dict:
    content = [{"type": "text", "text": prompt_msg}]
    if image:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_to_url(image)},
            }
        )
    user = {"role": "user", "content": content}

    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            user,
        ]
    }

    if json_schema:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "chat_response",
                "strict": True,
                "schema": json_schema,
            },
        }
    return payload


def build_payload_llama(prompt_msg: str, image: Path, grammar: str):
    payload = {}
    if image:
        payload["prompt"] = {
            "prompt_string": prompt_msg,
            "multimodal_data": process_image(image),
        }
    else:
        payload["prompt"] = prompt_msg
    if grammar:
        payload["grammar"] = grammar

    return payload


def load_schema(filename: Path) -> str:
    with io.open(Path(HOME, "../../json_schema", filename)) as f:
        jobj = json.loads(f.read())
    return jobj


def prompt(
    prompt_msg: str,
    port: int = 8080,
    image: Path = None,
    system_prompt: str = None,
    endpoint: str = None,
    grammar: str = None,
    json_schema: dict = None,
) -> requests.Response:
    if system_prompt is None:
        system_prompt = "You are an AI assistant. You only return the requested content without making comments."
    elif isinstance(system_prompt, list):
        system_prompt = "\n".join(system_prompt)

    if endpoint is None:
        endpoint = OAI_ENDPOINT
    host = f"http://localhost:{port}{endpoint}"

    headers = {"Content-Type": "application/json", "Authorization": "Bearer no-key"}

    if endpoint == OAI_ENDPOINT:
        payload = build_payload_oai(prompt_msg, image, system_prompt, json_schema)
    elif endpoint == LLAMA_ENDPOINT:
        payload = build_payload_llama(prompt_msg, image, grammar)

    data = json.dumps(payload, ensure_ascii=False)

    res = requests.post(url=host, headers=headers, data=data)
    return res


def response_content(res: requests.Response, endpoint: str = OAI_ENDPOINT) -> dict:
    jobj = json.loads(res.content)
    if endpoint == OAI_ENDPOINT:
        if "error" in jobj.keys():
            msg = jobj["error"]
            raise RuntimeError(msg)
        else:
            return jobj["choices"][0]["message"]["content"]
    elif endpoint == LLAMA_ENDPOINT:
        if "error" in jobj.keys():
            msg = jobj["error"]
            raise RuntimeError(msg)
        else:
            return jobj["content"]


def open_for_kill():
    choice = ""
    while choice != "k":
        choice = input("Press k to kill the llama: ")
        print(f"You've pressed: {choice}")
    kill_server()


def kill_server():
    cmd = f"taskkill /IM llama-server.exe /F"
    print(cmd)
    subprocess.Popen(cmd)
    print("Killed the llama")


if __name__ == "__main__":
    try:

        available_models = [m for m in MODELS]
        for ind, m in enumerate(available_models):
            print(f"[{ind}] - {m}")

        num = input("Please pick the model's number: ")

        model_name = available_models[int(num)]
        launch_server(model_name=model_name, open_browser=True)
        open_for_kill()
    finally:
        kill_server()
