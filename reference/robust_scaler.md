# scale a dataset using robust scaler and return a scaled dataframe, the median, the lower quantile (25% by default) and the upper quantile (75% by default) of each column

scale a dataset using robust scaler and return a scaled dataframe, the
median, the lower quantile (25% by default) and the upper quantile (75%
by default) of each column

## Usage

``` r
robust_scaler(data, initial.imp = "sample", lower = 0.25, upper = 0.75)
```

## Arguments

- data:

  A data frame or tibble

- initial.imp:

  The method for initial imputation. Can be "mean", "median" or "sample"
  (default).

- lower:

  The lower quantile (25% by default)

- upper:

  The upper quantile (75% by default)
