
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
# install.packages('devtools')
# devtools::install_github('agnesdeng/miae')
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


midae.imputed <- midae(data = penguins, m = 5, epochs = 10, latent.dim = 16,
    learning.rate = 0.001, batch.size = 50, encoder.structure = c(32,
        16), decoder.structure = c(16, 32), dropout.prob = 0.5)
#> Loss at epoch 1: 4.674150
#> Loss at epoch 2: 4.587650
#> Loss at epoch 3: 4.504400
#> Loss at epoch 4: 4.397995
#> Loss at epoch 5: 4.250195
#> Loss at epoch 6: 4.016459
#> Loss at epoch 7: 3.704035
#> Loss at epoch 8: 3.402138
#> Loss at epoch 9: 3.201382
#> Loss at epoch 10: 3.092280

# obtain the fifth imputed dataset
midae.imputed[[5]]
#> # A tibble: 344 × 8
#>    species island    bill_length_mm bill_depth_mm flipper_…¹ body_…² sex    year
#>    <fct>   <fct>              <dbl>         <dbl>      <dbl>   <dbl> <fct> <int>
#>  1 Adelie  Torgersen           39.1          18.7       181    3750  male   2007
#>  2 Adelie  Torgersen           39.5          17.4       186    3800  fema…  2007
#>  3 Adelie  Torgersen           40.3          18         195    3250  fema…  2007
#>  4 Adelie  Torgersen           46.9          17.4       199.   4682. male   2007
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
#> 1: 43.73368 51.48606 56.27728 61.38202 46.91410
#> 2: 37.00542 40.18033 45.13860 40.67043 42.55852
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


mivae.imputed <- mivae(data = penguins, m = 5, epochs = 10, latent.dim = 16,
    learning.rate = 0.001, batch.size = 50, encoder.structure = c(32,
        16), decoder.structure = c(16, 32))
#> Loss at epoch 1: 19.544793
#> Loss at epoch 2: 16.942577
#> Loss at epoch 3: 15.088655
#> Loss at epoch 4: 13.419140
#> Loss at epoch 5: 11.682644
#> Loss at epoch 6: 9.905313
#> Loss at epoch 7: 8.185184
#> Loss at epoch 8: 6.712330
#> Loss at epoch 9: 5.452672
#> Loss at epoch 10: 4.561251

# obtain the fifth imputed dataset
midae.imputed[[5]]
#> # A tibble: 344 × 8
#>    species island    bill_length_mm bill_depth_mm flipper_…¹ body_…² sex    year
#>    <fct>   <fct>              <dbl>         <dbl>      <dbl>   <dbl> <fct> <int>
#>  1 Adelie  Torgersen           39.1          18.7       181    3750  male   2007
#>  2 Adelie  Torgersen           39.5          17.4       186    3800  fema…  2007
#>  3 Adelie  Torgersen           40.3          18         195    3250  fema…  2007
#>  4 Adelie  Torgersen           46.9          17.4       199.   4682. male   2007
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
#> 1: 43.73368 51.48606 56.27728 61.38202 46.91410
#> 2: 37.00542 40.18033 45.13860 40.67043 42.55852
```
