INSTRUCTION = (
    "Decompile the following x86-64 assembly (compiled with gcc -{opt}) into functionally "
    "equivalent C code. The function is named func0. Return only the C function "
    "(plus any #includes it needs) in a single ```c code block.\n\n{asm}"
)


def instruction(asm: str, opt: str) -> dict:
    return {"role": "user", "content": INSTRUCTION.format(asm=asm, opt=opt)}


def assistant(text: str) -> dict:
    return {"role": "assistant", "content": text}
