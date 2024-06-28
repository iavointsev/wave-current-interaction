def print(*args, **kwargs):
    __builtins__.print(*args, **kwargs, flush = True)


display = print
Markdown = lambda *args, **kwargs: ...


def verbose_print(verbose: bool, *objs):
    return print(*objs) if verbose else None


def verbose_display(verbose: bool, *objs):
    return display(*objs) if verbose else None

