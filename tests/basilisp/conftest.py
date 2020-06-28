import sys
from typing import Dict, Optional

import pytest

import basilisp.lang.compiler as compiler
import basilisp.lang.reader as reader
import basilisp.lang.runtime as runtime
import basilisp.lang.symbol as sym
from tests.basilisp.helpers import CompileFn, get_or_create_ns


@pytest.fixture(params=[3, 4] if sys.version_info < (3, 8) else [3, 4, 5])
def pickle_protocol(request) -> int:
    return request.param


@pytest.fixture
def core_ns_sym() -> sym.Symbol:
    return runtime.CORE_NS_SYM


@pytest.fixture
def core_ns(core_ns_sym: sym.Symbol) -> runtime.Namespace:
    return get_or_create_ns(core_ns_sym)


@pytest.fixture
def test_ns_sym(test_ns: str) -> sym.Symbol:
    return sym.symbol(test_ns)


@pytest.fixture
def ns(test_ns: str, test_ns_sym: sym.Symbol) -> runtime.Namespace:
    get_or_create_ns(test_ns_sym)
    with runtime.ns_bindings(test_ns) as ns:
        try:
            yield ns
        finally:
            runtime.Namespace.remove(test_ns_sym)


@pytest.fixture
def lcompile(ns: runtime.Namespace, compiler_file_path: str) -> CompileFn:
    def _lcompile(
        s: str,
        resolver: Optional[reader.Resolver] = None,
        opts: Optional[Dict[str, bool]] = None,
    ):
        """Compile and execute the code in the input string.

        Return the resulting expression."""
        ctx = compiler.CompilerContext(compiler_file_path, opts=opts)

        last = None
        for form in reader.read_str(s, resolver=resolver):
            last = compiler.compile_and_exec_form(form, ctx, ns)

        return last

    return _lcompile
