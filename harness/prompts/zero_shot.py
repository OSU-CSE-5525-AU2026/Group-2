from .common import instruction


async def run(task, chat):
    return [await chat([instruction(task.asm, task.opt)])]
