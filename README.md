
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


midae.imputed <- midae(data = penguins, m = 5, device = "cpu",
    pmm.type = "auto", pmm.k = 5, pmm.link = "prob", pmm.save.vars = NULL,
    epochs = 10, batch.size = 32, subsample = 0.7, shuffle = TRUE,
    input.dropout = 0.2, hidden.dropout = 0.5, optimizer = "adamW",
    learning.rate = 1e-04, weight.decay = 0.002, momentum = 0,
    eps = 1e-07, encoder.structure = c(128, 64, 32), decoder.structure = c(32,
        64, 128), act = "elu", init.weight = "xavier.normal",
    scaler = "minmax", loss.na.scale = FALSE, early_stopping_epochs = 10,
    verbose = TRUE, print.every.n = 1, save.model = FALSE, path = file.path(tempdir(),
        "midaemodel.pt"))
#> Loss at epoch 1: training: 5.994925, validation: 4.972960
#> Loss at epoch 2: training: 5.486036, validation: 6.118598
#> Loss at epoch 3: training: 5.841651, validation: 5.217031
#> Loss at epoch 4: training: 5.412953, validation: 5.340371
#> Loss at epoch 5: training: 5.304536, validation: 5.382771
#> Loss at epoch 6: training: 5.680770, validation: 5.121284
#> Loss at epoch 7: training: 5.206397, validation: 5.169374
#> Loss at epoch 8: training: 5.205464, validation: 4.857170
#> Loss at epoch 9: training: 5.026849, validation: 4.663248
#> Loss at epoch 10: training: 5.000378, validation: 4.804644

# obtain the fifth imputed dataset
midae.imputed[[5]]
#> # A tibble: 344 × 8
#>    species island    bill_length_mm bill_depth_mm flipper_…¹ body_…² sex    year
#>    <fct>   <fct>              <dbl>         <dbl>      <int>   <int> <fct> <int>
#>  1 Adelie  Torgersen           39.1          18.7        181    3750 male   2007
#>  2 Adelie  Torgersen           39.5          17.4        186    3800 fema…  2007
#>  3 Adelie  Torgersen           40.3          18          195    3250 fema…  2007
#>  4 Adelie  Torgersen           41.4          13.5         NA      NA male   2007
#>  5 Adelie  Torgersen           36.7          19.3        193    3450 fema…  2007
#>  6 Adelie  Torgersen           39.3          20.6        190    3650 male   2007
#>  7 Adelie  Torgersen           38.9          17.8        181    3625 fema…  2007
#>  8 Adelie  Torgersen           39.2          19.6        195    4675 male   2007
#>  9 Adelie  Torgersen           34.1          18.1        193    3475 fema…  2007
#> 10 Adelie  Torgersen           42            20.2        190    4250 male   2007
#> # … with 334 more rows, and abbreviated variable names ¹​flipper_length_mm,
#> #   ²​body_mass_g


# show the imputed values for missing entries in the
# variable 'bill_length_mm'
show_var(imputation.list = midae.imputed, var.name = "bill_length_mm",
    original.data = penguins)
#>      m1   m2   m3   m4   m5
#> 1: 41.1 50.8 36.3 33.1 41.4
#> 2: 34.4 48.1 36.2 44.0 34.6
```

## 3. Multiple imputation with variational autoencoder

``` r
mivae.imputed <- mivae(data = penguins, m = 5, beta = 1, pmm.type = "auto",
    pmm.k = 5, pmm.link = "prob", pmm.save.vars = NULL, epochs = 10,
    batch.size = 32, subsample = 0.7, shuffle = TRUE, input.dropout = 0.2,
    hidden.dropout = 0.5, optimizer = "adamW", learning.rate = 1e-04,
    weight.decay = 0.002, momentum = 0, eps = 1e-07, encoder.structure = c(128,
        64, 32), latent.dim = 4, decoder.structure = c(32, 64,
        128), act = "elu", init.weight = "xavier.normal", scaler = "minmax",
    loss.na.scale = FALSE, early_stopping_epochs = 10, verbose = TRUE,
    print.every.n = 1, save.model = FALSE, path = file.path(tempdir(),
        "mivaemodel.pt"))
#> Loss at epoch 1: training: 7.173252, validation: 7.833011
#> Loss at epoch 2: training: 6.875861, validation: 6.893571
#> Loss at epoch 3: training: 6.598606, validation: 7.147669
#> Loss at epoch 4: training: 6.484610, validation: 6.685006
#> Loss at epoch 5: training: 6.264470, validation: 6.330046
#> Loss at epoch 6: training: 6.478614, validation: 6.349185
#> Loss at epoch 7: training: 5.969728, validation: 6.308928
#> Loss at epoch 8: training: 5.972846, validation: 5.935225
#> Loss at epoch 9: training: 5.701628, validation: 5.801384
#> Loss at epoch 10: training: 5.632129, validation: 6.100064


# obtain the fifth imputed dataset
mivae.imputed[[5]]
#> # A tibble: 344 × 8
#>    species island    bill_length_mm bill_depth_mm flipper_…¹ body_…² sex    year
#>    <fct>   <fct>              <dbl>         <dbl>      <int>   <int> <fct> <int>
#>  1 Adelie  Torgersen           39.1          18.7        181    3750 male   2007
#>  2 Adelie  Torgersen           39.5          17.4        186    3800 fema…  2007
#>  3 Adelie  Torgersen           40.3          18          195    3250 fema…  2007
#>  4 Adelie  Torgersen           45.5          16.6         NA      NA fema…  2007
#>  5 Adelie  Torgersen           36.7          19.3        193    3450 fema…  2007
#>  6 Adelie  Torgersen           39.3          20.6        190    3650 male   2007
#>  7 Adelie  Torgersen           38.9          17.8        181    3625 fema…  2007
#>  8 Adelie  Torgersen           39.2          19.6        195    4675 male   2007
#>  9 Adelie  Torgersen           34.1          18.1        193    3475 fema…  2007
#> 10 Adelie  Torgersen           42            20.2        190    4250 fema…  2007
#> # … with 334 more rows, and abbreviated variable names ¹​flipper_length_mm,
#> #   ²​body_mass_g


# show the imputed values for missing entries in the
# variable 'bill_length_mm'
show_var(imputation.list = mivae.imputed, var.name = "bill_length_mm",
    original.data = penguins)
#>      m1   m2   m3   m4   m5
#> 1: 49.7 44.0 38.1 42.9 45.5
#> 2: 45.9 35.7 40.2 42.2 46.4
```

## 4. Impute new data using a saved imputation model

``` r
n <- nrow(penguins)
idx <- sample(1:n, size = round(0.7 * n), replace = FALSE)
train.data <- penguins[idx, ]
test.data <- penguins[-idx, ]


midae.obj <- midae(data = train.data, m = 5, epochs = 10, subsample = 1,
    early_stopping_epochs = 1, scaler = "minmax", save.model = TRUE,
    path = file.path(tempdir(), "midaemodel.pt"))
#> Loss at epoch 1: 5.259101
#> Loss at epoch 2: 5.380220
#> Loss at epoch 3: 4.955275
#> Loss at epoch 4: 5.024055
#> Loss at epoch 5: 5.226000
#> Loss at epoch 6: 4.944716
#> Loss at epoch 7: 4.783398
#> Loss at epoch 8: 4.626323
#> Loss at epoch 9: 4.611824
#> Loss at epoch 10: 4.804672
#> [1] "The DAE multiple imputation model is saved in  C:\\Users\\agnes\\AppData\\Local\\Temp\\RtmpI5FXK7/midaemodel.pt"


midae.newdata <- impute_new(object = midae.obj, newdata = test.data,
    m = 5)
```
