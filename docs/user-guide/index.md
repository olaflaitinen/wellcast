# User guide

The user guide explains the engineering behind each module: the assumptions,
the equations, their valid ranges and the published sources. The
[API reference](../api/index.md) lists every function and argument.

## Conventions

* **Oilfield units** throughout: rates in bbl/d (gas in Mscf/d), pressures in
  psi, temperatures in degF, gas-oil ratios in scf/STB, choke sizes in 64ths
  of an inch. Convert other units on input with
  [`read_production(..., units=...)`](io.md) or [`convert`](units.md).
* **Time series** are pandas DataFrames indexed by a `DatetimeIndex` named
  `date`, with a `well` column, sorted by well and date: the
  [canonical schema](io.md#canonical-schema).
* **Flag, never delete.** Cleaning functions return boolean flags. Whether to
  drop, replace or keep a value stays the engineer's decision.
* **Correlations warn outside their data.** Every empirical correlation
  documents the range it was fitted on and issues a `RangeWarning` when
  extrapolated.
* **Errors are specific.** All exceptions derive from `WellcastError`:
  `DataValidationError` for bad input, `FitError` for failed fits,
  `UnitError` for unit problems and `MissingDependencyError` for missing
  extras.
