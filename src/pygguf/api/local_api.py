from multiprocessing.sharedctypes import Value
from typing import Callable
import requests
import subprocess
from pathlib import Path
import time
import json
from http.client import responses
import io
from pygguf.api.img_utils import image_to_url, process_image
import webbrowser
import atexit

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
    return Path(DATA_PATH, rf"models/{model_name}").resolve().absolute()


def wait_for_llama(
    server: subprocess.Popen[bytes],
    host: str,
    timeout: float = 30,
    interval: float = 0.5,
    verbose: bool = False,
):
    start = time.time()
    while time.time() - start < timeout:
        try:
            if server.poll() is not None:  # process has exited
                stdout, stderr = server.communicate()
                if stderr:
                    raise RuntimeError(f"llama-server crashed:\n{stderr.decode()}")
                else:
                    raise RuntimeError("llama-server crashed: reasons unknown")
            r = requests.get(f"{host}/health", timeout=2)
            if r.status_code == 200:
                if verbose:
                    print("llama-server ready!")
                return True
            else:
                if verbose:
                    print(f"Status: {r.status_code}")
        except requests.exceptions.ConnectionError as e:
            if verbose:
                print(f"Exception: {e}")
        time.sleep(interval)
    raise TimeoutError(f"llama-server didn't start within {timeout}s")


def process_arg(arg):
    if isinstance(arg, str) and arg.endswith(".gguf"):
        return model_fpath(arg)
    return arg


def launch_server(
    port: int = 8080,
    ctx: int = int(2**13),
    verbose: bool = False,
    model_name: str = "Assistant_Pepe_8B-Q8_0.gguf",
    open_browser: bool = False,
    **sup_args,
) -> subprocess.Popen[bytes]:

    exe = LLAMAEXE

    if verbose:
        kwargs = {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE}
    else:
        kwargs = {"stderr": subprocess.DEVNULL, "stdout": subprocess.DEVNULL}
    model = model_fpath(model_name)
    cmd = f"{exe} -m {model} --offline --port {port} -c {ctx}"
    if sup_args is not None:
        cmd = (
            cmd
            + " "
            + " ".join([f"--{k} {process_arg(v)}" for k, v in sup_args.items()])
        )
    print(cmd)

    cmds = [c for c in cmd.split(" ") if c != ""]

    server: subprocess.Popen[bytes] = subprocess.Popen(cmds, **kwargs)
    host = f"http://localhost:{port}"
    wait_for_llama(server, host)
    r = requests.get(host, timeout=5)
    n = 0
    n_dots = 5
    while r.status_code == 503:
        try:
            time.sleep(0.2)
            r = requests.get(host)
            if not verbose:
                print(
                    f"Status code {r.status_code} ({responses[r.status_code]}){moving_dots(n, n_dots)} on localhost:{port} model: {model_name}",
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

    ksfn = assign_server_killer(server)
    atexit.register(ksfn)
    return server


def build_payload_oai(
    prompt_msg: str,
    image: Path,
    system_prompt: str,
    json_schema: str,
    reasoning_budget: int=None,
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
        ],"id_slot":0,"cache_prompt":True
    }

    if json_schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "chat_response",
                "strict": True,
                "schema": json_schema,
            },
        }
    if reasoning_budget is not None:
        payload["thinking_budget_tokens"] = reasoning_budget
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


def stream_chat(
    prompt_msg: str,
    port: int = 8080,
    endpoint: str = None,
    system_prompt: str = None,
    on_token=None,
    on_reasoning_token=None,
    image: Path = None,
    grammar: str = None,
    json_schema: dict = None,
) -> dict:
    """
    Streams a chat completion from llama-server, printing/live-updating as it goes,
    but returns a single reconstructed JSON-like dict at the end containing:
      - content
      - reasoning_content (if the model emits it)
      - finish_reason
      - timings (llama-server specific)
      - usage (if present)
      - raw_chunks (all raw SSE chunks, in case you want to inspect anything else)

    Args:
        on_token: callback for each content token, e.g. lambda t: print(t, end="", flush=True)
        on_reasoning_token: callback for each reasoning token, same signature
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    content_parts = []
    reasoning_parts = []
    finish_reason = None
    timings = None
    usage = None
    raw_chunks = []

    if on_token is None:

        def on_token(t):
            print(t, end="", flush=True)

    if on_reasoning_token is None:

        def on_reasoning_token(t):
            print(t, end="", flush=True)

    if endpoint is None:
        endpoint = OAI_ENDPOINT
    host = f"http://localhost:{port}{endpoint}"

    if endpoint == OAI_ENDPOINT:
        payload = build_payload_oai(prompt_msg, image, system_prompt, json_schema)
    elif endpoint == LLAMA_ENDPOINT:
        payload = build_payload_llama(prompt_msg, image, grammar)

    payload["stream"] = True

    with requests.post(host, json=payload, stream=True) as response:
        response.raise_for_status()
        response.encoding = "utf-8"
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data_str = line[len("data: ") :]
            if data_str.strip() == "[DONE]":
                break

            chunk = json.loads(data_str)
            raw_chunks.append(chunk)

            choice = chunk["choices"][0]
            delta = choice.get("delta", {})

            token = delta.get("content")
            if token:
                content_parts.append(token)
                if on_token:
                    on_token(token)

            reasoning_token = delta.get("reasoning_content")
            if reasoning_token:
                reasoning_parts.append(reasoning_token)
                if on_reasoning_token:
                    on_reasoning_token(reasoning_token)
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

            # llama-server attaches "timings" to chunks as generation progresses;
            # the last one present is the most complete.
            if "timings" in chunk:
                timings = chunk["timings"]

            if "usage" in chunk and chunk["usage"] is not None:
                usage = chunk["usage"]

    return {
        "content": "".join(content_parts),
        "reasoning_content": "".join(reasoning_parts) if reasoning_parts else None,
        "finish_reason": finish_reason,
        "timings": timings,
        "usage": usage,
        "raw_chunks": raw_chunks,
    }


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


def response_timings(res:requests.Response,endpoint:str=OAI_ENDPOINT)->dict:
    jobj = res.json()
    if endpoint == OAI_ENDPOINT:
        if "error" in jobj.keys():
            msg = jobj["error"]
            raise RuntimeError(msg)
        else:
            return jobj["timings"]
    else:
        raise ValueError("Timings only work for OAI_ENDPOINT")


def response_content(res: requests.Response, endpoint: str = OAI_ENDPOINT) -> str:
    return response_message_key(res=res, key="content", endpoint=endpoint)


def response_message_key(
    res: requests.Response, key: str = "content", endpoint: str = OAI_ENDPOINT
) -> str:
    jobj = res.json()
    if endpoint == OAI_ENDPOINT:
        if "error" in jobj.keys():
            msg = jobj["error"]
            raise RuntimeError(msg)
        else:
            return jobj["choices"][0]["message"][key]
    elif endpoint == LLAMA_ENDPOINT:
        if "error" in jobj.keys():
            msg = jobj["error"]
            raise RuntimeError(msg)
        else:
            return jobj[key]
    else:
        raise ValueError(f"Endpoint not recognized: {endpoint}")


def open_for_kill(server: subprocess.Popen[bytes]):
    choice = ""
    while choice != "k":
        choice = input("Press k to kill the llama: ")
        print(f"You've pressed: {choice}")
    kill_server(server)


def assign_server_killer(server: subprocess.Popen[bytes]) -> Callable:
    def ksfn():
        kill_server(server)

    return ksfn


def kill_server(server: subprocess.Popen[bytes]):
    if server.poll() is not None:
        return  # already dead
    server.terminate()
    try:
        server.wait(timeout=10)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait()
    # if os.name=="posix":
    #     cmd = f"fuser -k {port}/tcp"
    #     print(cmd)
    #     subprocess.Popen(cmd.split(" "))
    # else:
    #     cmd = f"taskkill /IM llama-server.exe /F"
    #     print(cmd)
    #     subprocess.Popen(cmd)
    print("Killed the llama")


if __name__ == "__main__":
    available_models = [m for m in MODELS]
    for ind, m in enumerate(available_models):
        print(f"[{ind}] - {m}")

    num = input("Please pick the model's number: ")

    model_name = available_models[int(num)]
    server = launch_server(model_name=model_name, open_browser=True)
    # open_for_kill(server)
