# plot the density of the impiuted values of a numeric variable using different dropout probabilities

plot the density of the impiuted values of a numeric variable using
different dropout probabilities

## Usage

``` r
plot_dropout(tune.results, var.name, xlim = NULL, ylim = NULL)
```

## Arguments

- tune.results:

  object returned by tune_dropout()

- var.name:

  the name of a numeric variable

- xlim:

  the left and right limit of the x-axis. Default: NULL.

- ylim:

  the lower and upper limit of the y-axis. Default: NULL.
