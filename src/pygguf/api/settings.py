import os
from pathlib import Path

HOME = Path(__file__).parent

# morse code
DATA_PATH = Path(HOME, "../../../../data/")

MODELS = os.listdir(Path(DATA_PATH, "models"))


latest = os.listdir(Path(DATA_PATH,"bin"))
if os.name == "posix":
    LLAMAEXE = Path(DATA_PATH, f"bin/{latest[-1]}/llama-server")
else:
    LLAMAEXE = Path(
        DATA_PATH, fr"bin\{latest[-1]}\llama-server.exe"
    )
