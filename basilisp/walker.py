from functools import partial
import basilisp.lang.list as llist
import basilisp.lang.map as lmap
import basilisp.lang.set as lset
import basilisp.lang.vector as vector


def walk(inner_f, outer_f, form):
    """"""
    if isinstance(form, llist.List):
        return outer_f(llist.list(map(inner_f, form)))
    elif isinstance(form, vector.Vector):
        return outer_f(vector.vector(map(inner_f, form)))
    elif isinstance(form, lmap.Map):
        return outer_f(lmap.from_entries(map(inner_f, form)))
    elif isinstance(form, lset.Set):
        return outer_f(lset.set(map(inner_f, form)))
    else:
        return outer_f(form)


def postwalk(f, form):
    inner_f = partial(postwalk, f)
    return walk(inner_f, f, form)


def prewalk(f, form):
    inner_f = partial(prewalk, f)
    return walk(inner_f, lambda x: x, f(form))


def _print_identity(v):
    print(v)
    return v


def postwalk_demo(form):
    return postwalk(_print_identity, form)


def prewalk_demo(form):
    return prewalk(_print_identity, form)
