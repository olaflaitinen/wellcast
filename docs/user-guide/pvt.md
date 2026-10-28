# PVT

`wellcast.pvt` implements the classic black-oil and gas correlations in
oilfield units (psia, degF, scf/STB, API gravity, gas gravity relative to air).
Inputs may be scalars or arrays.

## Oil

| Property | Function | Correlation |
|---|---|---|
| Bubble point | `bubble_point_standing`, `bubble_point_vasquez_beggs` | Standing (1947), Vasquez and Beggs (1980) |
| Solution GOR | `solution_gor_standing`, `solution_gor_vasquez_beggs` | idem |
| Oil FVF below $p_b$ | `oil_fvf_standing`, `oil_fvf_vasquez_beggs` | idem |
| Oil FVF above $p_b$ | `oil_fvf_undersaturated` | Vasquez and Beggs compressibility |
| Dead oil viscosity | `dead_oil_viscosity_beggs_robinson` | Beggs and Robinson (1975) |
| Live oil viscosity | `live_oil_viscosity_beggs_robinson` | Beggs and Robinson (1975) |
| Undersaturated viscosity | `undersaturated_oil_viscosity_vasquez_beggs` | Vasquez and Beggs (1980) |

Standing's bubble point is

$$
p_b = 18.2 \left[ \left(\frac{R_s}{\gamma_g}\right)^{0.83} 10^{\,0.00091 T - 0.0125\,\mathrm{API}} - 1.4 \right]
$$

and his saturated formation volume factor

$$
B_o = 0.9759 + 0.000120 \left[ R_s \sqrt{\gamma_g / \gamma_o} + 1.25 T \right]^{1.2}.
$$

The Beggs-Robinson viscosities are

$$
\mu_{od} = 10^{X} - 1, \quad X = 10^{3.0324 - 0.02023\,\mathrm{API}} \, T^{-1.163},
\qquad \mu_o = A \mu_{od}^B,
$$

with $A = 10.715 (R_s + 100)^{-0.515}$ and $B = 5.44 (R_s + 150)^{-0.338}$.

`black_oil_table` combines them into one table over a pressure range.

## Gas

`pseudo_critical_standing` gives the pseudo-critical properties of a natural
gas, and `z_factor_hall_yarborough` solves the Hall and Yarborough (1973)
equation of state, a fit to the Standing-Katz chart, by a damped Newton
method. `z_factor` combines both, and `gas_fvf` returns

$$
B_g = \frac{p_{sc}}{T_{sc}} \frac{z T}{p} \quad [\text{ft}^3/\text{scf}].
$$

The implementation reproduces the reference value $z = 0.9580$ at
$p_{pr} = 1.5$, $T_{pr} = 2.0$ and stays within 0.007 of the digitized
Standing-Katz chart for $1.2 \le T_{pr} \le 3$, $p_{pr} \le 5.5$.

## Valid ranges

| Correlation | Temperature (degF) | API | Gas gravity | Rs (scf/STB) | Pressure (psia) |
|---|---|---|---|---|---|
| Standing | 100 to 258 | 16.5 to 63.8 | 0.59 to 0.95 | 20 to 1425 | 130 to 7000 |
| Vasquez-Beggs | 70 to 295 | 15.3 to 59.5 | 0.511 to 1.351 | 0 to 2199 | 50 to 5250 |
| Beggs-Robinson | 70 to 295 | 16 to 58 | | 20 to 2070 | |
| Hall-Yarborough | $1.0 \le T_{pr} \le 3.0$ | | | | $p_{pr} \le 24$ |

Outside these ranges the functions still return values but issue a
`RangeWarning`.

## References

* Standing, M. B. (1947). A pressure-volume-temperature correlation for
  mixtures of California oils and gases. Drilling and Production Practice, API.
* Vasquez, M. and Beggs, H. D. (1980). Correlations for fluid physical
  property prediction. JPT 32(6), SPE-6719-PA.
* Beggs, H. D. and Robinson, J. R. (1975). Estimating the viscosity of crude
  oil systems. JPT 27(9), SPE-5434-PA.
* Hall, K. R. and Yarborough, L. (1973). A new equation of state for Z-factor
  calculations. Oil and Gas Journal 71(25).
