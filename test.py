import importlib


def test_module_import():
    """Basic import of the top-level module should succeed."""
    mod = importlib.import_module("bmap")
    assert mod is not None


def test_public_symbol_imports():
    """Import a few commonly used public symbols."""
    from bmap import BufferMap, BufferAlign, NativeTypes, bmap_todot, bmap_pyvis

    assert BufferMap
    assert BufferAlign
    assert NativeTypes
    assert callable(bmap_todot)
    assert callable(bmap_pyvis)


if __name__ == "__main__":
    test_module_import()
    test_public_symbol_imports()
    print("Import checks passed.")
