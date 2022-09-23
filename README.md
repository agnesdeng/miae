
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
    path = file.path(tempdir(), "midaemodel.pt"))
#> Loss at epoch 1: 4.139691
#> Loss at epoch 2: 3.908631
#> Loss at epoch 3: 3.506792
#> Loss at epoch 4: 3.324880
#> Loss at epoch 5: 3.197487
#> Loss at epoch 6: 3.120290
#> Loss at epoch 7: 3.117152
#> Loss at epoch 8: 3.098643
#> Loss at epoch 9: 3.100188
#> Loss at epoch 10: 3.095499

# obtain the fifth imputed dataset
midae.imputed[[5]]
#> # A tibble: 344 × 8
#>    species island    bill_length_mm bill_depth_mm flipper_…¹ body_…² sex    year
#>    <fct>   <fct>              <dbl>         <dbl>      <dbl>   <dbl> <fct> <int>
#>  1 Adelie  Torgersen           39.1          18.7       181    3750  male   2007
#>  2 Adelie  Torgersen           39.5          17.4       186    3800  fema…  2007
#>  3 Adelie  Torgersen           40.3          18         195    3250  fema…  2007
#>  4 Adelie  Torgersen           42.8          16.9       198.   4107. male   2007
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
#>         m1       m2       m3       m4       m5
#> 1: 42.9338 42.46144 43.03259 42.86761 42.76480
#> 2: 43.3091 44.26307 42.70475 45.02276 46.88858
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


mivae.imputed <- mivae(data = penguins, m = 5, epochs = 10, path = file.path(tempdir(),
    "mivaemodel.pt"))
#> Loss at epoch 1: 7.255387
#> Loss at epoch 2: 5.133022
#> Loss at epoch 3: 3.770610
#> Loss at epoch 4: 3.420828
#> Loss at epoch 5: 3.269362
#> Loss at epoch 6: 3.238938
#> Loss at epoch 7: 3.209049
#> Loss at epoch 8: 3.180461
#> Loss at epoch 9: 3.181362
#> Loss at epoch 10: 3.168574

# obtain the fifth imputed dataset
mivae.imputed[[5]]
#> # A tibble: 344 × 8
#>    species island    bill_length_mm bill_depth_mm flipper_…¹ body_…² sex    year
#>    <fct>   <fct>              <dbl>         <dbl>      <dbl>   <dbl> <fct> <int>
#>  1 Adelie  Torgersen           39.1          18.7       181    3750  male   2007
#>  2 Adelie  Torgersen           39.5          17.4       186    3800  fema…  2007
#>  3 Adelie  Torgersen           40.3          18         195    3250  fema…  2007
#>  4 Adelie  Torgersen           42.5          16.6       195.   3871. male   2007
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
show_var(imputation.list = mivae.imputed, var.name = "bill_length_mm",
    original.data = penguins)
#>          m1       m2       m3       m4       m5
#> 1: 48.21146 42.55643 45.53735 46.89693 42.49806
#> 2: 42.21244 45.21044 41.50069 45.81596 43.29091
```

## 4. Impute new data using a saved imputation model

``` r
n <- nrow(penguins)
idx <- sample(1:n, size = round(0.7 * n), replace = FALSE)
train.data <- penguins[idx, ]
test.data <- penguins[-idx, ]



midae.data <- midae(data = train.data, m = 5, epochs = 5, path = "C:/Users/agnes/Desktop/torch/midaemodel.pt")
#> Loss at epoch 1: 4.425285
#> Loss at epoch 2: 4.270670
#> Loss at epoch 3: 4.114981
#> Loss at epoch 4: 3.918035
#> Loss at epoch 5: 3.662064
midae.newdata <- impute_new(path = "C:/Users/agnes/Desktop/torch/midaemodel.pt",
    newdata = test.data, m = 5)



mivae.data <- mivae(data = train.data, m = 5, epochs = 5, path = "C:/Users/agnes/Desktop/torch/mivaemodel.pt")
#> Loss at epoch 1: 6.495280
#> Loss at epoch 2: 5.205251
#> Loss at epoch 3: 4.352683
#> Loss at epoch 4: 3.775249
#> Loss at epoch 5: 3.441398
mivae.newdata <- impute_new(path = "C:/Users/agnes/Desktop/torch/mivaemodel.pt",
    newdata = test.data, m = 5)
```
