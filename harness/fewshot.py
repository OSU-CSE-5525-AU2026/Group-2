import json
import re
import subprocess
import tempfile
from pathlib import Path

from .config import OPT_LEVELS
from .prompts.few_shot import EXAMPLES

SOURCES = EXAMPLES.parent


def disassemble(source: Path, opt: str) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        obj = Path(tmp) / "f.o"
        subprocess.run(["gcc", "-c", "-fcf-protection", "-o", obj, source, f"-{opt}"], check=True)
        dump = subprocess.run(["objdump", "-d", obj], check=True, capture_output=True, text=True).stdout
    text = dump.split("Disassembly of section .text:")[-1].strip()
    lines = [line.split("\t")[-1].split("#")[0].strip() for line in text.split("\n")]
    return re.sub(r"^0+\s", "", "\n".join(lines) + "\n")


def build() -> None:
    sources = sorted(SOURCES.glob("*.c"))
    examples = {
        opt: [{"name": s.stem, "c": s.read_text().strip(), "asm": disassemble(s, opt)} for s in sources]
        for opt in OPT_LEVELS
    }
    EXAMPLES.write_text(json.dumps(examples, indent=2) + "\n")
    print(f"wrote {len(sources)} examples x {len(OPT_LEVELS)} optimization levels to {EXAMPLES}")
