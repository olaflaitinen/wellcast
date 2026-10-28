# Cleaning and quality control

Field data contain shut-ins, reporting gaps, frozen gauges, transcription
errors and allocation artefacts. `wellcast.clean` finds them and returns
boolean flags aligned with the input. It never deletes data.

## Shut-ins

`detect_shut_ins` marks a row as shut in when its producing hours are zero, or
when every available rate is known and at or below a threshold. A missing
rate is not a zero rate: reporting gaps are not shut-ins.

## Outliers

| Function | Method | Use for |
|---|---|---|
| `hampel_filter` | Rolling median and MAD | Isolated spikes in noisy signals |
| `rolling_zscore_outliers` | Rolling mean and standard deviation | Smooth signals |
| `iqr_outliers` | Tukey's fences on the whole series | Quick global screening |
| `flag_outliers` | Any of the above, per well | Multi-well tables |

The Hampel identifier flags a value $x_i$ when

$$
|x_i - m_i| > t \cdot 1.4826 \cdot \mathrm{MAD}_i
$$

where $m_i$ and $\mathrm{MAD}_i$ are the median and median absolute deviation
of a centered window; 1.4826 makes the MAD a consistent estimate of the
standard deviation for Gaussian noise (Pearson, 2002).

Always group by well: a normal rate for one well can be an outlier for
another.

## Regular sampling and gaps

`resample_regular` puts every well on the same time grid, leaving missing
periods as NaN so gaps stay visible. `fill_gaps` fills only short gaps (by
default up to 3 rows), never extrapolates past the ends, and never mixes wells.

## QC report

`qc_report` returns a tidy table with one row per well and column:

* `missing` and `missing_pct`;
* `negative`: values below zero;
* `flatline`: values inside runs of at least 5 identical non-zero readings
  (a frozen gauge or copied-forward allocation). Zero runs are shut-ins and
  setpoint columns (`choke`, `on_stream_hours`) are constant by design, so
  neither counts;
* `impossible`: values outside physical limits such as a negative pressure,
  a water cut outside [0, 1] or more than 24 producing hours in a day.
