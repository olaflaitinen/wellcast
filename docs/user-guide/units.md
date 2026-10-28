# Units

`wellcast.units` converts between oilfield and SI units through a registry.
Each unit belongs to a physical quantity and maps affinely to a base unit:

$$
x_\text{base} = s \, x + o
$$

so a conversion between two units of the same quantity is

$$
x_\text{to} = \frac{s_\text{from} \, x + o_\text{from} - o_\text{to}}{s_\text{to}}.
$$

Only temperatures use an offset. Converting between quantities raises
`UnitError`.

```python
from wellcast.units import convert, list_units

convert(1000.0, "psi", "bar")  # 68.9476
convert(212.0, "degF", "degC")  # 100.0
convert(1.0, "MMscf/d", "sm3/d")  # 28316.85
convert(800.0, "scf/stb", "sm3/sm3")  # 142.49
list_units("pressure")
```

The container type is preserved: floats, NumPy arrays, pandas Series and
DataFrames all work.

| Quantity | Units |
|---|---|
| Pressure | psi, kPa, Pa, MPa, bar, atm |
| Temperature | degF, degC, K, degR |
| Volume rate | bbl/d, stb/d, m3/d, sm3/d, ksm3/d, scf/d, Mscf/d, MMscf/d |
| Volume | bbl, stb, Mbbl, MMbbl, m3, sm3, ft3, scf, Mscf, MMscf, Bscf |
| Gas-liquid ratio | scf/stb, Mscf/stb, sm3/sm3 |
| Length | ft, m, in, in/64 |
| Time | d, h, yr |
| Ratio | fraction, percent |

Common spellings are accepted (`BOPD`, `psia`, `Mcf/d`, `%`), and
`register_unit` adds your own.

!!! note "Standard conditions"
    Oilfield standard conditions (60 degF, 14.696 psia) and metric standard
    conditions (15 degC, 101.325 kPa) differ by about 0.5 percent in gas
    volume. Like most production reporting, wellcast treats `stb` and `sm3`
    (and `scf` and `sm3` of gas) as plain volumes and ignores the difference.

All pressures are absolute; gauge pressures are not converted implicitly.
