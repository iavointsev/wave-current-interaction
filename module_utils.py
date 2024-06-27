from IPython.display import display, Markdown

display = print
Markdown = Markdown

def verbose_print(verbose: bool, *objs):
    return print(*objs) if verbose else None


def verbose_display(verbose: bool, *objs):
    return display(*objs) if verbose else None

