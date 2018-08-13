import datetime
import re
import uuid
from typing import Pattern

import dateutil.parser as dateparser

import basilisp.lang.atom as atom


def lrepr(f) -> str:
    """Return the canonical Lisp representation of an object."""
    if f is True:
        return "true"
    elif f is False:
        return "false"
    elif f is None:
        return "nil"
    elif isinstance(f, str):
        return f'"{f}"'
    elif isinstance(f, datetime.datetime):
        inst_str = f.isoformat()
        return f'#inst "{inst_str}"'
    elif isinstance(f, uuid.UUID):
        uuid_str = str(f)
        return f'#uuid "{uuid_str}"'
    elif isinstance(f, Pattern):
        return f'#"{f.pattern}"'
    else:
        return repr(f)


# Use an atomically incremented integer as a suffix for all
# user-defined function and variable names compiled into Python
# code so no conflicts occur
_NAME_COUNTER = atom.Atom(1)


def next_name_id() -> int:
    """Increment the name counter and return the next value."""
    return _NAME_COUNTER.swap(lambda x: x + 1)


def genname(prefix: str) -> str:
    """Generate a unique function name with the given prefix."""
    i = next_name_id()
    return f"{prefix}_{i}"


def inst_from_str(inst_str: str) -> datetime.datetime:
    """Create a datetime instance from an RFC 3339 formatted date string."""
    return dateparser.parse(inst_str)


def regex_from_str(regex_str: str) -> Pattern:
    """Create a new regex pattern from the input string."""
    return re.compile(regex_str)


def uuid_from_str(uuid_str: str) -> uuid.UUID:
    """Create a new UUID instance from the canonical string representation
    of a UUID."""
    return uuid.UUID(f'{{{uuid_str}}}')
