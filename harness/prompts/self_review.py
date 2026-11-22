from .common import assistant, instruction

REVIEW = (
    "Review your decompilation against the assembly line by line. Check the control flow "
    "and loop bounds, argument and variable types including signedness and width, missing or "
    "extra operations, memory access sizes, and the return value on every path. Fix every "
    "mistake you find and return the complete corrected function (plus any #includes it needs) "
    "in a single ```c code block."
)


async def run(task, chat):
    messages = [instruction(task.asm, task.opt)]
    draft = await chat(messages)
    final = await chat([*messages, assistant(draft.text), {"role": "user", "content": REVIEW}])
    return [draft, final]
