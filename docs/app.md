# Interactive app

wellcast ships a multipage Streamlit app that runs every analysis without
writing code.

```bash
pip install "wellcast[all]"
wellcast app
```

The app opens in the browser. Press **Load demo data** to explore the
synthetic field with zero setup.

| Page | What it does |
|---|---|
| Data | Upload CSV or Excel, map columns and units to the wellcast schema, quality-control report |
| Explore | Production history, cumulative volumes, GOR and water cut per well |
| Predict | Train a virtual flow meter, time-ordered or leave-one-well-out validation, prediction intervals, SHAP |
| Decline | Arps fits and EUR for every well, bootstrap P90/P50/P10 for one well |
| Diagnose | IPR from a well test, Chan plot, Hall plot, anomaly detection |
| Report | Export a standalone HTML report |

## Deploying on Streamlit Community Cloud

The repository is ready for [Streamlit Community Cloud](https://streamlit.io/cloud):
use `app/streamlit_app.py` as the entry point. Dependencies come from
`app/requirements.txt` and the theme from `.streamlit/config.toml`.
