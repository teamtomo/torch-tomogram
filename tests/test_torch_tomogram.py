import torch_tomogram


def test_imports_with_version():
    assert isinstance(torch_tomogram.__version__, str)
