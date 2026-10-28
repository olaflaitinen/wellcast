# Inflow performance and chokes

An inflow performance relationship (IPR) gives the rate a well delivers at a
flowing bottomhole pressure $p_{wf}$ for an average reservoir pressure
$\bar{p}_r$.

## Models

| Function | Use for | Equation |
|---|---|---|
| `linear_ipr` | Undersaturated oil, single phase | $q = J(\bar{p}_r - p_{wf})$ |
| `vogel_ipr` | Saturated, solution-gas drive | $q/q_{max} = 1 - 0.2 x - 0.8 x^2$, $x = p_{wf}/\bar{p}_r$ |
| `fetkovich_ipr` | Back-pressure (multi-rate) tests | $q = C(\bar{p}_r^2 - p_{wf}^2)^n$ |
| `composite_ipr` | Undersaturated reservoir produced below $p_b$ | linear above $p_b$, Vogel below |

The composite IPR (Brown, 1984) joins a straight line above the bubble point
to a Vogel curve below it with matching slope:

$$
q = q_b + \frac{J p_b}{1.8}\left[1 - 0.2\frac{p_{wf}}{p_b} - 0.8\left(\frac{p_{wf}}{p_b}\right)^2\right],
\qquad q_b = J(\bar{p}_r - p_b).
$$

## From a well test

```python
from wellcast import ipr

qmax = ipr.vogel_qmax(rate=350.0, pwf=2000.0, reservoir_pressure=2500.0)  # 1067.1
ipr.vogel_ipr(1850.0, 2500.0, qmax)  # 441.7
j = ipr.composite_productivity_index(3000.0, 1500.0, 4000.0, 2000.0)
c, n = ipr.fit_fetkovich(rates, pwfs, reservoir_pressure=3600.0)
curve = ipr.ipr_curve("composite", 4000.0, bubble_point=2000.0, productivity_index=j)
```

The Vogel numbers above reproduce the classic textbook example
(Ahmed, Reservoir Engineering Handbook).

## Critical-flow chokes

For two-phase critical flow through a choke, the empirical correlations share
the form

$$
p_{wh} = \frac{A \, q_L \, R^{B}}{S^{C}}
$$

with $p_{wh}$ in psi, $q_L$ in STB/d, gas-liquid ratio $R$ in scf/STB and bean
size $S$ in 64ths of an inch:

| Correlation | A | B | C |
|---|---|---|---|
| Gilbert (1954) | 10.00 | 0.546 | 1.89 |
| Ros (1960) | 17.40 | 0.500 | 2.00 |
| Baxendell (1957) | 9.56 | 0.546 | 1.93 |
| Achong (1961) | 3.82 | 0.650 | 1.88 |

`choke_pressure` and its inverse `choke_rate` implement them;
`gilbert_choke`, `ros_choke` and `baxendell_choke` are shortcuts. The
equations hold for critical flow, roughly when the downstream pressure is
below 0.588 times the upstream pressure.

## References

* Vogel, J. V. (1968). Inflow performance relationships for solution-gas drive
  wells. JPT 20(1), SPE-1476-PA.
* Fetkovich, M. J. (1973). The isochronal testing of oil wells. SPE-4529-MS.
* Brown, K. E. (1984). The Technology of Artificial Lift Methods, Vol. 4.
* Gilbert, W. E. (1954). Flowing and gas-lift well performance. Drilling and
  Production Practice, API.
