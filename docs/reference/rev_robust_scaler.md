# This function back-transform robust-scaled data to an output as data matrix

This function back-transform robust-scaled data to an output as data
matrix

## Usage

``` r
rev_robust_scaler(
  scaled.data,
  num.names,
  robust.lower,
  robust.upper,
  robust.median
)
```

## Arguments

- scaled.data:

  A matrix or array with scaled numeric data

- num.names:

  the names of numeric features

- robust.lower:

  A vector that contains the the lower quantile (25% by default) of each
  column

- robust.upper:

  A vector that contains the the upper quantile (75% by default) of each
  column

- robust.median:

  A vector that contains the the median (50%) of each column
