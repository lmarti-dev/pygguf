import os
from pathlib import Path

HOME = Path(__file__).parent

# morse code
DATA_PATH = Path(HOME, "../../../../data/")

MODELS = os.listdir(Path(DATA_PATH, "models"))

if os.name == "posix":
    LLAMAEXE = Path(DATA_PATH, "bin/llama-b7822/llama-server")
else:
    LLAMAEXE = Path(
        DATA_PATH, r"bin\llama-b7058-bin-win-cuda-12.4-x64\llama-server.exe"
    )
