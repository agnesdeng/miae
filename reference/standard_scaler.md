# scale a dataset using stardardize and return a scaled dataframe, the colmean and colsd of each column

scale a dataset using stardardize and return a scaled dataframe, the
colmean and colsd of each column

## Usage

``` r
standard_scaler(data, initial.imp = "sample")
```

## Arguments

- data:

  A data frame or tibble

- initial.imp:

  The method for initial imputation. Can be "mean", "median" or "sample"
  (default).
