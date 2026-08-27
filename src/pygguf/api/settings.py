import os
from pathlib import Path
from typing import Literal

HOME = Path(__file__).parent

# morse code
DATA_PATH = Path(HOME, "../../../../data/")

MODELS = os.listdir(Path(DATA_PATH, "models"))


latest = os.listdir(Path(DATA_PATH,"bin"))
if os.name == "posix":
    LLAMAEXE = Path(DATA_PATH, f"bin/{latest[-1]}/llama-server").resolve().absolute()
else:
    LLAMAEXE = Path(
        DATA_PATH, fr"bin\{latest[-1]}\llama-server.exe"
    )




OAI_ENDPOINT = "/v1/chat/completions"
LLAMA_ENDPOINT = "/completion"
Endpoints = Literal[OAI_ENDPOINT,LLAMA_ENDPOINT]

SYSTEM_PROMPT  = "You are an AI assistant. You only return the requested content without making comments."
APPLY_TEMPLATE_ENDPOINT = "/apply-template"