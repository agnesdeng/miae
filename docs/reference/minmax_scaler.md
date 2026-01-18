# scale a dataset using minmax and return a scaled dataframe, the colmin and colmax of each column

scale a dataset using minmax and return a scaled dataframe, the colmin
and colmax of each column

## Usage

``` r
minmax_scaler(data, initial.imp = "sample")
```

## Arguments

- data:

  A data frame or tibble

- initial.imp:

  The method for initial imputation. Can be "mean", "median" or "sample"
  (default).
