
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
#> Loss at epoch 1: training: 5.462930, validation: 5.431731
#> Loss at epoch 2: training: 5.224686, validation: 5.131269
#> Loss at epoch 3: training: 4.878889, validation: 4.769587
#> Loss at epoch 4: training: 4.563283, validation: 4.516335
#> Loss at epoch 5: training: 4.342751, validation: 4.267286
#> Loss at epoch 6: training: 4.138786, validation: 4.109632
#> Loss at epoch 7: training: 3.966700, validation: 3.988988
#> Loss at epoch 8: training: 3.888035, validation: 3.874518
#> Loss at epoch 9: training: 3.852712, validation: 3.800892
#> Loss at epoch 10: training: 3.808102, validation: 3.801624

# obtain the fifth imputed dataset
midae.imputed[[5]]
#> # A tibble: 344 × 8
#>    species island    bill_length_mm bill_depth_mm flipper_…¹ body_…² sex    year
#>    <fct>   <fct>              <dbl>         <dbl>      <dbl>   <dbl> <fct> <int>
#>  1 Adelie  Torgersen           39.1          18.7       181    3750  male   2007
#>  2 Adelie  Torgersen           39.5          17.4       186    3800  fema…  2007
#>  3 Adelie  Torgersen           40.3          18         195    3250  fema…  2007
#>  4 Adelie  Torgersen           42.9          17.1       196.   4124. male   2007
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
#> 1: 42.15255 42.28803 44.99340 43.25145 42.90605
#> 2: 44.36680 42.64654 45.28653 48.06393 46.08304
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
#> Loss at epoch 1: training: 5.526942, validation: 5.286708
#> Loss at epoch 2: training: 5.109785, validation: 4.902773
#> Loss at epoch 3: training: 4.666276, validation: 4.408109
#> Loss at epoch 4: training: 4.269063, validation: 4.203901
#> Loss at epoch 5: training: 4.115277, validation: 4.174640
#> Loss at epoch 6: training: 4.018902, validation: 4.004897
#> Loss at epoch 7: training: 3.910706, validation: 3.985325
#> Loss at epoch 8: training: 3.894901, validation: 3.881455
#> Loss at epoch 9: training: 3.854608, validation: 3.890917
#> Loss at epoch 10: training: 3.838459, validation: 3.875065

# obtain the fifth imputed dataset
mivae.imputed[[5]]
#> # A tibble: 344 × 8
#>    species island    bill_length_mm bill_depth_mm flipper_…¹ body_…² sex    year
#>    <fct>   <fct>              <dbl>         <dbl>      <dbl>   <dbl> <fct> <int>
#>  1 Adelie  Torgersen           39.1          18.7       181    3750  male   2007
#>  2 Adelie  Torgersen           39.5          17.4       186    3800  fema…  2007
#>  3 Adelie  Torgersen           40.3          18         195    3250  fema…  2007
#>  4 Adelie  Torgersen           44.4          17.2       202.   4110. fema…  2007
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
#> 1: 44.98584 41.62716 43.83523 41.67571 44.38759
#> 2: 44.90587 42.44198 47.68633 41.62919 42.75678
```

## 4. Impute new data using a saved imputation model

``` r
n <- nrow(penguins)
idx <- sample(1:n, size = round(0.7 * n), replace = FALSE)
train.data <- penguins[idx, ]
test.data <- penguins[-idx, ]

midae.data <- midae(data = train.data, m = 5, epochs = 5, path = "C:/Users/agnes/Desktop/torch/midaemodel.pt")
#> Loss at epoch 1: training: 5.385571, validation: 5.440279
#> Loss at epoch 2: training: 5.164929, validation: 5.194732
#> Loss at epoch 3: training: 4.924557, validation: 4.881342
#> Loss at epoch 4: training: 4.616096, validation: 4.475012
#> Loss at epoch 5: training: 4.250568, validation: 4.308607
midae.newdata <- impute_new(path = "C:/Users/agnes/Desktop/torch/midaemodel.pt",
    newdata = test.data, m = 5)


mivae.data <- mivae(data = train.data, m = 5, epochs = 5, path = "C:/Users/agnes/Desktop/torch/mivaemodel.pt")
#> Loss at epoch 1: training: 5.111503, validation: 4.853832
#> Loss at epoch 2: training: 4.702660, validation: 4.515272
#> Loss at epoch 3: training: 4.422507, validation: 4.235496
#> Loss at epoch 4: training: 4.155054, validation: 4.080909
#> Loss at epoch 5: training: 4.016924, validation: 3.998439
mivae.newdata <- impute_new(path = "C:/Users/agnes/Desktop/torch/mivaemodel.pt",
    newdata = test.data, m = 5)
```
