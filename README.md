
<!-- README.md is generated from README.Rmd. Please edit that file -->

# miae

<!-- badges: start -->
<!-- badges: end -->

**miae** is an R package for multiple imputation through autoencoders
built with **Torch**. It’s currently under development.

## 1. Installation

You can install the current development version of miae from
[GitHub](https://github.com/) with:

``` r
# install.packages("devtools")
#devtools::install_github("agnesdeng/miae")
```

``` r
library(miae)
```

## 2. Multiple imputation with denoising autoencoder with dropout

``` r
# load the penguins dataset
library(palmerpenguins)
# 5 variables have missing values
colSums(is.na(penguins))
#>           species            island    bill_length_mm     bill_depth_mm 
#>                 0                 0                 2                 2 
#> flipper_length_mm       body_mass_g               sex              year 
#>                 2                 2                11                 0


midae.imputed <- midae(data = penguins, m = 5, epochs = 10, batch.size = 50,
    input.dropout = 0.9, latent.dropout = 0.5, hidden.dropout = 1,
    optimizer = "adam", learning.rate = 0.001, weight.decay = 0,
    momentum = 0, encoder.structure = c(128, 64, 32), latent.dim = 8,
    decoder.structure = c(32, 64, 128), verbose = TRUE, print.every.n = 1,
    directory = tempdir())
#> Loss at epoch 1: 3.941177
#> Loss at epoch 2: 3.531636
#> Loss at epoch 3: 3.293763
#> Loss at epoch 4: 3.177864
#> Loss at epoch 5: 3.134452
#> Loss at epoch 6: 3.119597
#> Loss at epoch 7: 3.116458
#> Loss at epoch 8: 3.113550
#> Loss at epoch 9: 3.112760
#> Loss at epoch 10: 3.105702

# obtain the fifth imputed dataset
midae.imputed[[5]]
#> # A tibble: 344 × 8
#>    species island    bill_length_mm bill_depth_mm flipper_…¹ body_…² sex    year
#>    <fct>   <fct>              <dbl>         <dbl>      <dbl>   <dbl> <fct> <int>
#>  1 Adelie  Torgersen           39.1          18.7       181    3750  male   2007
#>  2 Adelie  Torgersen           39.5          17.4       186    3800  fema…  2007
#>  3 Adelie  Torgersen           40.3          18         195    3250  fema…  2007
#>  4 Adelie  Torgersen           44.1          17.3       202.   4217. male   2007
#>  5 Adelie  Torgersen           36.7          19.3       193    3450  fema…  2007
#>  6 Adelie  Torgersen           39.3          20.6       190    3650  male   2007
#>  7 Adelie  Torgersen           38.9          17.8       181    3625  fema…  2007
#>  8 Adelie  Torgersen           39.2          19.6       195    4675  male   2007
#>  9 Adelie  Torgersen           34.1          18.1       193    3475  male   2007
#> 10 Adelie  Torgersen           42            20.2       190    4250  male   2007
#> # … with 334 more rows, and abbreviated variable names ¹​flipper_length_mm,
#> #   ²​body_mass_g


# show the imputed values for missing entries in the
# variable 'bill_length_mm'
show_var(imputation.list = midae.imputed, var.name = "bill_length_mm",
    original.data = penguins)
#>          m1       m2       m3       m4       m5
#> 1: 43.11955 44.73244 43.15084 42.83997 44.10414
#> 2: 46.09493 45.39409 43.22618 43.72621 43.02097
```

## 3. Multiple imputation with variational autoencoder

``` r
# load the penguins dataset
library(palmerpenguins)
# 5 variables have missing values
colSums(is.na(penguins))
#>           species            island    bill_length_mm     bill_depth_mm 
#>                 0                 0                 2                 2 
#> flipper_length_mm       body_mass_g               sex              year 
#>                 2                 2                11                 0


mivae.imputed <- mivae(data = penguins, m = 5, epochs = 10, directory = tempdir())
#> Loss at epoch 1: 7.627935
#> Loss at epoch 2: 5.496252
#> Loss at epoch 3: 4.072000
#> Loss at epoch 4: 3.430029
#> Loss at epoch 5: 3.293771
#> Loss at epoch 6: 3.247174
#> Loss at epoch 7: 3.206221
#> Loss at epoch 8: 3.191160
#> Loss at epoch 9: 3.192088
#> Loss at epoch 10: 3.172476

# obtain the fifth imputed dataset
mivae.imputed[[5]]
#> # A tibble: 344 × 8
#>    species island    bill_length_mm bill_depth_mm flipper_…¹ body_…² sex    year
#>    <fct>   <fct>              <dbl>         <dbl>      <dbl>   <dbl> <fct> <int>
#>  1 Adelie  Torgersen           39.1          18.7       181    3750  male   2007
#>  2 Adelie  Torgersen           39.5          17.4       186    3800  fema…  2007
#>  3 Adelie  Torgersen           40.3          18         195    3250  fema…  2007
#>  4 Adelie  Torgersen           41.7          15.7       193.   3799. male   2007
#>  5 Adelie  Torgersen           36.7          19.3       193    3450  fema…  2007
#>  6 Adelie  Torgersen           39.3          20.6       190    3650  male   2007
#>  7 Adelie  Torgersen           38.9          17.8       181    3625  fema…  2007
#>  8 Adelie  Torgersen           39.2          19.6       195    4675  male   2007
#>  9 Adelie  Torgersen           34.1          18.1       193    3475  fema…  2007
#> 10 Adelie  Torgersen           42            20.2       190    4250  fema…  2007
#> # … with 334 more rows, and abbreviated variable names ¹​flipper_length_mm,
#> #   ²​body_mass_g


# show the imputed values for missing entries in the
# variable 'bill_length_mm'
show_var(imputation.list = mivae.imputed, var.name = "bill_length_mm",
    original.data = penguins)
#>          m1       m2       m3       m4       m5
#> 1: 45.27994 42.73557 45.86078 45.26621 41.65775
#> 2: 42.58150 44.45517 42.58916 43.75477 42.91792
```
