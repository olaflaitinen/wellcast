# Data input

## Canonical schema

Every analysis in wellcast works on one table layout: a DataFrame indexed by a
`DatetimeIndex` named `date`, sorted by well and date, with these columns
(only `well` is required):

| Column | Unit | Description |
|---|---|---|
| `well` | | Well identifier |
| `oil_rate` | bbl/d | Oil rate at stock-tank conditions |
| `gas_rate` | Mscf/d | Gas rate at standard conditions |
| `water_rate` | bbl/d | Water rate |
| `water_inj_rate` | bbl/d | Water injection rate |
| `gas_inj_rate` | Mscf/d | Gas injection or gas lift rate |
| `whp` | psi | Wellhead (tubing head) pressure |
| `wht` | degF | Wellhead temperature |
| `bhp` | psi | Bottomhole (downhole gauge) pressure |
| `bht` | degF | Bottomhole temperature |
| `choke` | in/64 | Choke bean size |
| `on_stream_hours` | h | Producing hours in the period |

Other columns are kept after the canonical ones.

## Reading production data

```python
from wellcast.io import read_production

df = read_production("daily_production.xlsx", units={"oil_rate": "sm3/d"})
```

Column names are matched case-insensitively and without punctuation against a
built-in alias map that covers common database exports, for example
`DATEPRD` to `date`, `BORE_OIL_VOL` to `oil_rate`, `AVG_WHP_P` to `whp` and
`AVG_DOWNHOLE_PRESSURE` to `bhp`. Add your own names with
`alias_map={"Fecha": "date"}`. Unparseable numbers become NaN rather than
failing the whole file; files without a well column take `well="name"`.

Tables already in memory (database queries, uploads) go through
`from_dataframe` with the same options.

## Well logs

`read_las` reads LAS 1.2, 2.0 and 3.0 files through
[lasio](https://github.com/kinverarity1/lasio) into a depth-indexed DataFrame.
Header items are stored in `df.attrs["well"]` and curve units in
`df.attrs["units"]`. It needs the `io` extra.
