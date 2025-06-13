# torch-tomogram

[![License](https://img.shields.io/pypi/l/torch-tomogram.svg?color=green)](https://github.com/tlambert03/torch-tomogram/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/torch-tomogram.svg?color=green)](https://pypi.org/project/torch-tomogram)
[![Python Version](https://img.shields.io/pypi/pyversions/torch-tomogram.svg?color=green)](https://python.org)
[![CI](https://github.com/tlambert03/torch-tomogram/actions/workflows/ci.yml/badge.svg)](https://github.com/tlambert03/torch-tomogram/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/tlambert03/torch-tomogram/branch/main/graph/badge.svg)](https://codecov.io/gh/tlambert03/torch-tomogram)

Tomogram reconstruction, subtomogram reconstruction, and subtilt extraction for cryo-ET.

## Development

The easiest way to get started is to use the [github cli](https://cli.github.com)
and [uv](https://docs.astral.sh/uv/getting-started/installation/):

```sh
gh repo fork tlambert03/torch-tomogram --clone
# or just
# gh repo clone tlambert03/torch-tomogram
cd torch-tomogram
uv sync
```

Run tests:

```sh
uv run pytest
```

Lint files:

```sh
uv run pre-commit run --all-files
```
