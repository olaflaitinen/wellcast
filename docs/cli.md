# Command-line interface

Installing wellcast adds the `wellcast` command (also available as
`python -m wellcast`).

| Command | Purpose |
|---|---|
| `wellcast --version` | Print the version |
| `wellcast qc <file>` | Print the data-quality report of a CSV or Excel file |
| `wellcast dca <file> --well <name>` | Fit a decline curve and print the EUR |
| `wellcast demo` | Write the synthetic demo field to CSV |
| `wellcast app` | Launch the Streamlit app |

## Examples

```bash
wellcast demo --output field.csv --labels labels.csv --wells 8 --years 6
wellcast qc field.csv --format csv > qc.csv
wellcast dca field.csv --well P-03 --model hyperbolic --econ-limit 15 --bootstrap 300
wellcast app --port 8600
```

Files without a well column take the well name from `--default-well`.
Text output rounds to six significant figures; `--format csv` keeps full
precision.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | A wellcast error: bad data, failed fit, missing optional dependency |
| 2 | Invalid command-line usage |

The full option list of each command is shown by `wellcast <command> --help`.
