# Installation

wellcast supports Python 3.10, 3.11, 3.12 and 3.13 on Linux, macOS and
Windows.

```bash
pip install wellcast
```

The core install depends only on NumPy, pandas and SciPy. Optional features
come as extras:

| Extra | Adds | Needed for |
|---|---|---|
| `io` | lasio, openpyxl | LAS logs and Excel files |
| `ml` | scikit-learn, LightGBM, SHAP, joblib | `wellcast.ml`, Isolation Forest anomalies |
| `viz` | plotly | `wellcast.plotting`, `DeclineModel.plot` |
| `app` | Streamlit, plotly | `wellcast app` |
| `all` | all of the above | everything |
| `dev` | pytest, ruff, mypy, pre-commit and build tools | contributing |
| `docs` | MkDocs Material, mkdocstrings, mkdocs-jupyter | building this site |

```bash
pip install "wellcast[all]"
```

Using a feature whose extra is missing raises
`MissingDependencyError` with the exact command to run, for example
`Install the ml extra: pip install 'wellcast[ml]'`.

## From source

```bash
git clone https://github.com/olaflaitinen/wellcast.git
cd wellcast
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all,dev,docs]"
pytest
```

## LightGBM on macOS

LightGBM needs the OpenMP runtime. Install it with Homebrew if importing
LightGBM fails:

```bash
brew install libomp
```
