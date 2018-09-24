import sys
from unittest.mock import patch

import basilisp.importer as importer


def importer_counter():
    return sum([isinstance(o, importer.BasilispImporter) for o in sys.meta_path])


def test_hook_imports():
    with patch('sys.meta_path',
               new=[]):
        assert 0 == importer_counter()
        importer.hook_imports()
        assert 1 == importer_counter()
        importer.hook_imports()
        assert 1 == importer_counter()

    with patch('sys.meta_path',
               new=[importer.BasilispImporter()]):
        assert 1 == importer_counter()
        importer.hook_imports()
        assert 1 == importer_counter()
