import atexit
import json
import subprocess
import time
import webbrowser
from collections.abc import Callable
from http.client import responses
from pathlib import Path

import jinja2
import requests

from pygguf.api.img_utils import image_to_url, process_image
from pygguf.api.settings import (
    APPLY_TEMPLATE_ENDPOINT,
    DATA_PATH,
    HOME,
    LLAMA_ENDPOINT,
    LLAMAEXE,
    MODELS,
    OAI_ENDPOINT,
    SYSTEM_PROMPT,
)


def load_json(fpath: Path) -> dict:
    with open(fpath, "r", encoding="utf8") as f:
        jobj = json.loads(f.read())
    return jobj


def open_grammar(filename: str) -> str:
    with open(Path(HOME, "../grammars", f"{filename}.gbnf"), "r", encoding="utf8") as f:
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
                _, stderr = server.communicate()
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
    ctx: int = 2**13,
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
    system_prompt: str | None = None,
    assistant_prompt: str | None = None,
    json_schema: str | None = None,
    reasoning_budget: int | None = None,
    **chat_template_kwargs,
) -> dict:

    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT

    content = [{"type": "text", "text": prompt_msg}]
    if image:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_to_url(image)},
            }
        )

    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        "id_slot": 0,
        "cache_prompt": True,
    }
    if assistant_prompt is not None:
         payload["messages"].append({"role": "assistant", "content": assistant_prompt})

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
    if chat_template_kwargs:
        payload.update(chat_template_kwargs)
    return payload



def _render_initial_prompt(system_prompt: str, prompt_msg: str, port: int) -> str:
    """One /apply-template call, no generation. Gets the exact raw prompt
    this model's template produces for a fresh turn -- including whatever
    per-model reasoning-open tag it inserts -- without us needing to know
    that template's syntax at all."""
    host = f"http://localhost:{port}{APPLY_TEMPLATE_ENDPOINT}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_msg},
    ]
    return requests.post(host, json={"messages": messages}).json()["prompt"]
 
 
def build_payload_llama(
    prompt_msg: str,
    image: Path | None = None,
    grammar: str | None = None,
    system_prompt: str | None = None,
    assistant_prompt: str | None = None,
    port: int = 8080,
    **chat_template_kwargs,
) -> dict:
    """
    assistant_prompt: None for the first leg of a conversation -- the real
        starting prompt gets rendered once via /apply-template. For every
        continuation, pass the raw text generated + injected so far; it's
        used exactly as given, no template re-application, nothing appended
        or closed on your behalf. The model stays "open" for as long as you
        keep extending this string yourself.
    port: needed now because round-1 requires one live call to this
        server's /apply-template to render the starting prompt.
    """
    payload = {"id_slot": 0, "cache_prompt": True}  # cache_prompt matters more here than
                                                      # on the OAI side: every injection round
                                                      # resends a longer version of the same
                                                      # string, and this lets llama.cpp reuse
                                                      # the common prefix instead of reprocessing it
 
    if assistant_prompt is None:
        raw_prompt = _render_initial_prompt(system_prompt or SYSTEM_PROMPT, prompt_msg, port)
    else:
        raw_prompt = assistant_prompt
 
    if image:
        payload["prompt"] = {
            "prompt_string": raw_prompt,
            "multimodal_data": process_image(image),
        }
    else:
        payload["prompt"] = raw_prompt
 
    if grammar:
        payload["grammar"] = grammar
    if chat_template_kwargs:
        payload.update(chat_template_kwargs)
    return payload



def load_schema(filename: Path) -> str:
    with open(Path(HOME, "../../json_schema", filename)) as f:
        jobj = json.loads(f.read())
    return jobj


def get_special_tokens(port: int = 8080):
    props = requests.get(f"http://localhost:{port}/props").json()
    template_str = props["chat_template"]
    bos = props.get("bos_token", "")
    eos = props.get("eos_token", "")
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    template = env.from_string(template_str)

    # Render with a dummy conversation + thinking enabled, no actual generation involved
    rendered = template.render(
        messages=[{"role": "user", "content": "PLACEHOLDER"}],
        add_generation_prompt=True,
        bos_token=bos,
        eos_token=eos,
        enable_thinking=True,  # matches the kwarg the server accepts
    )
    print(rendered)


def prompt(
    prompt_msg: str,
    port: int = 8080,
    image: Path | None = None,
    system_prompt: str | None = None,
    endpoint: str | None = None,
    grammar: str | None = None,
    json_schema: dict | None = None,
    reasoning_budget: int = 2**10,
    **chat_template_kwargs,
) -> requests.Response:
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT
    elif isinstance(system_prompt, list):
        system_prompt = "\n".join(system_prompt)

    if endpoint is None:
        endpoint = OAI_ENDPOINT
    host = f"http://localhost:{port}{endpoint}"

    headers = {"Content-Type": "application/json", "Authorization": "Bearer no-key"}

    if endpoint == OAI_ENDPOINT:
        payload = build_payload_oai(
            prompt_msg,
            image=image,
            system_prompt=system_prompt,
            json_schema=json_schema,
            reasoning_budget=reasoning_budget,
            **chat_template_kwargs,
        )
    elif endpoint == LLAMA_ENDPOINT:
        payload = build_payload_llama(
            prompt_msg, image, grammar, **chat_template_kwargs
        )

    data = json.dumps(payload, ensure_ascii=False)

    res = requests.post(url=host, headers=headers, data=data)
    return res


def response_timings(res: requests.Response, endpoint: str = OAI_ENDPOINT) -> dict:
    jobj = res.json()
    if endpoint == OAI_ENDPOINT:
        if "error" in jobj:
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
        if "error" in jobj:
            msg = jobj["error"]
            raise RuntimeError(msg)
        else:
            return jobj["choices"][0]["message"][key]
    elif endpoint == LLAMA_ENDPOINT:
        if "error" in jobj:
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
