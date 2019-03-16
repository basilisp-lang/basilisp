import uuid
from datetime import datetime
from decimal import Decimal
from fractions import Fraction
from typing import Union, Pattern

import basilisp.lang.keyword as kw
import basilisp.lang.list as llist
import basilisp.lang.map as lmap
import basilisp.lang.seq as lseq
import basilisp.lang.set as lset
import basilisp.lang.symbol as sym
import basilisp.lang.vector as vec

IterableLispForm = Union[llist.List, lmap.Map, lset.Set, vec.Vector]
LispNumber = Union[int, float, Fraction]
LispForm = Union[
    bool,
    complex,
    datetime,
    Decimal,
    int,
    float,
    Fraction,
    kw.Keyword,
    llist.List,
    lmap.Map,
    None,
    Pattern,
    lset.Set,
    str,
    sym.Symbol,
    vec.Vector,
    uuid.UUID,
]
PyCollectionForm = Union[dict, list, set, tuple]
ReaderForm = Union[LispForm, lseq.Seq, PyCollectionForm]
SpecialForm = Union[llist.List, lseq.Seq]
