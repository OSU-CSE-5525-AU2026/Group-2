PROMPT = """Decompile the following x86-64 assembly (compiled with gcc -{opt}) into functionally equivalent C code. The function is named func0.

Work through it in this order before writing any code:
1. Signature: which registers carry the arguments, their likely types and signedness, and the return type.
2. Locals: stack slots and registers that hold local variables, and their types.
3. Control flow: map every jump target to the branches and loops it forms, including loop bounds and exit conditions.
4. Memory: loads, stores, pointer arithmetic and array indexing, with element sizes.
5. Calls: library or other functions that are called and the arguments passed to them.

Then give the C function (plus any #includes it needs) in a single ```c code block at the end.

{asm}"""


async def run(task, chat):
    return [await chat([{"role": "user", "content": PROMPT.format(asm=task.asm, opt=task.opt)}])]
