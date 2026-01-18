# Impute new data with a saved `midae` or `mivae` imputation model

Impute new data with a saved `midae` or `mivae` imputation model

## Usage

``` r
impute_new(object, newdata, pmm.k = NULL, m = NULL, verbose = FALSE)
```

## Arguments

- object:

  A saved imputer object created by `midae(..., save.models = TRUE)` or
  `mivae(..., save.models = TRUE)`.

- newdata:

  A data frame, tibble or data.table. New data with missing values.

- pmm.k:

  The number of donors for predictive mean matching. If `NULL` (the
  default), the `pmm.k` value in the saved imputer object will be used.

- m:

  The number of imputed datasets. If `NULL` (the default), the `m` value
  in the saved imputer object will be used.

- verbose:

  A logical value indicating whether to print the progress.

## Value

A list of `m` imputed datasets for new data.
