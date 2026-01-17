# This function back-transform minmax-scaled data to an output as data matrix

This function back-transform minmax-scaled data to an output as data
matrix

## Usage

``` r
rev_minmax_scaler(scaled.data, num.names, colmin, colmax)
```

## Arguments

- scaled.data:

  A matrix or array with scaled numeric data

- num.names:

  the names of numeric features

- colmin:

  A vector that contains the minimum of each column

- colmax:

  A vector that contains the maximum of each column
