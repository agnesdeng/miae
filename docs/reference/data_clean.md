# Data cleaning

Check some common errors of a raw dataset and return a suitable dataset
to be fed into the imputer. Note that this function is just a
preliminary check. It will not guarantee the output dataset is fully
cleaned.

## Usage

``` r
data_clean(rawdata, levels.tol = 0.2)
```

## Arguments

- rawdata:

  A data frame.

- levels.tol:

  Tolerant proportion of the number of levels to the number of
  observations in a multiclass variable. Default: 0.2

## Value

A preliminary cleaned dataset

## Examples

``` r
data(nhanes3)
rawdata <- nhanes3

rawdata[4, 4] <- NaN
rawdata[5, 5] <- Inf
rawdata[6, 6] <- -Inf

cleandata <- data_clean(rawdata = rawdata)
#> Warning: There exists at least one entry coded as NaN in the following numeric variable(s): head_circumference_cm.
#> It is now coverted to NA instead.
#> Warning: There exists at least one entry coded as Inf or -Inf in the following variable(s): recumbent_length_cm;weight_kg.
#> It is now coverted to NA instead.
```
