# This function back-transform decile-scaled data to an output as data matrix

This function back-transform decile-scaled data to an output as data
matrix

## Usage

``` r
rev_decile_scaler(scaled.data, num.names, decile1, decile9)
```

## Arguments

- scaled.data:

  A matrix or array with scaled numeric data

- num.names:

  the names of numeric features

- decile1:

  A vector that contains the the 1st decile (10th percentile) of each
  column

- decile9:

  A vector that contains the the 9th decile (90th percentile) of each
  column
