# scale a dataset using decile and return a scaled dataframe, the 1st decile and the 9th decile of each column

scale a dataset using decile and return a scaled dataframe, the 1st
decile and the 9th decile of each column

## Usage

``` r
decile_scaler(data, initial.imp = "sample")
```

## Arguments

- data:

  A data frame or tibble

- initial.imp:

  The method for initial imputation. Can be "mean", "median" or "sample"
  (default).
