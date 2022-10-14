
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


midae.imputed <- midae(data = penguins, m = 5, epochs = 10, batch.size = 16,
    split.ratio = 0.7, shuffle = TRUE, input.dropout = 0.2, hidden.dropout = 0.5,
    optimizer = "adamW", learning.rate = 1e-04, weight.decay = 0.002,
    momentum = 0, encoder.structure = c(128, 64, 32), decoder.structure = c(32,
        64, 128), act = "elu", init.weight = "xavier.uniform",
    scaler = "minmax", verbose = TRUE, print.every.n = 1, save.model = FALSE,
    path = NULL)
#> Loss at epoch 1: training: 10.299205, validation: 10.888454
#> Loss at epoch 2: training: 10.275936, validation: 11.384246
#> Loss at epoch 3: training: 9.750004, validation: 10.271372
#> Loss at epoch 4: training: 9.907795, validation: 9.924134
#> Loss at epoch 5: training: 9.639254, validation: 9.603723
#> Loss at epoch 6: training: 9.457273, validation: 9.964234
#> Loss at epoch 7: training: 8.658179, validation: 8.908058
#> Loss at epoch 8: training: 8.535068, validation: 9.024557
#> Loss at epoch 9: training: 8.758941, validation: 8.026823
#> Loss at epoch 10: training: 8.371416, validation: 7.533647

# obtain the fifth imputed dataset
midae.imputed[[5]]
#> # A tibble: 344 × 8
#>    species island    bill_length_mm bill_depth_mm flipper_…¹ body_…² sex    year
#>    <fct>   <fct>              <dbl>         <dbl>      <dbl>   <dbl> <fct> <int>
#>  1 Adelie  Torgersen           39.1         18.7       181     3750  male   2007
#>  2 Adelie  Torgersen           39.5         17.4       186     3800  fema…  2007
#>  3 Adelie  Torgersen           40.3         18         195     3250  fema…  2007
#>  4 Adelie  Torgersen          -25.0          6.81       82.5   8748. fema…  2007
#>  5 Adelie  Torgersen           36.7         19.3       193     3450  fema…  2007
#>  6 Adelie  Torgersen           39.3         20.6       190     3650  male   2007
#>  7 Adelie  Torgersen           38.9         17.8       181     3625  fema…  2007
#>  8 Adelie  Torgersen           39.2         19.6       195     4675  male   2007
#>  9 Adelie  Torgersen           34.1         18.1       193     3475  fema…  2007
#> 10 Adelie  Torgersen           42           20.2       190     4250  male   2007
#> # … with 334 more rows, and abbreviated variable names ¹​flipper_length_mm,
#> #   ²​body_mass_g


# show the imputed values for missing entries in the
# variable 'bill_length_mm'
show_var(imputation.list = midae.imputed, var.name = "bill_length_mm",
    original.data = penguins)
#>          m1        m2        m3        m4        m5
#> 1: 12.83791 52.968483 57.652382 114.07214 -25.03034
#> 2: 32.66182 -1.516143 -5.914922  42.28546  68.24944
```

## 3. Multiple imputation with variational autoencoder

``` r

mivae.imputed <- mivae(data = penguins, m = 5, epochs = 10, batch.size = 16,
    split.ratio = 0.7, shuffle = TRUE, input.dropout = 0.2, hidden.dropout = 0.5,
    optimizer = "adamW", learning.rate = 1e-04, weight.decay = 0.002,
    momentum = 0, encoder.structure = c(128, 64, 32), latent.dim = 4,
    decoder.structure = c(32, 64, 128), act = "elu", init.weight = "xavier.uniform",
    scaler = "minmax", verbose = FALSE, print.every.n = 1, save.model = FALSE,
    path = NULL)


# obtain the fifth imputed dataset
mivae.imputed[[5]]
#> # A tibble: 344 × 8
#>    species island    bill_length_mm bill_depth_mm flipper_…¹ body_…² sex    year
#>    <fct>   <fct>              <dbl>         <dbl>      <dbl>   <dbl> <fct> <int>
#>  1 Adelie  Torgersen           39.1          18.7       181    3750  male   2007
#>  2 Adelie  Torgersen           39.5          17.4       186    3800  fema…  2007
#>  3 Adelie  Torgersen           40.3          18         195    3250  fema…  2007
#>  4 Adelie  Torgersen           65.2          16.9       200.   -432. male   2007
#>  5 Adelie  Torgersen           36.7          19.3       193    3450  fema…  2007
#>  6 Adelie  Torgersen           39.3          20.6       190    3650  male   2007
#>  7 Adelie  Torgersen           38.9          17.8       181    3625  fema…  2007
#>  8 Adelie  Torgersen           39.2          19.6       195    4675  male   2007
#>  9 Adelie  Torgersen           34.1          18.1       193    3475  fema…  2007
#> 10 Adelie  Torgersen           42            20.2       190    4250  male   2007
#> # … with 334 more rows, and abbreviated variable names ¹​flipper_length_mm,
#> #   ²​body_mass_g


# show the imputed values for missing entries in the
# variable 'bill_length_mm'
show_var(imputation.list = mivae.imputed, var.name = "bill_length_mm",
    original.data = penguins)
#>          m1       m2       m3       m4       m5
#> 1: 91.20238 55.95815 53.24149 47.53025 65.19030
#> 2: 20.21769 32.69418 24.08430 27.76072 24.78258
```

## 4. Impute new data using a saved imputation model

``` r
n <- nrow(penguins)
idx <- sample(1:n, size = round(0.7 * n), replace = FALSE)
train.data <- penguins[idx, ]
test.data <- penguins[-idx, ]



midae.imputed <- midae(data = train.data, m = 5, epochs = 10,
    scaler = "minmax", save.model = TRUE, path = file.path(tempdir(),
        "midaemodel.pt"))
#> Loss at epoch 1: training: 14.065244, validation: 14.268253
#> Loss at epoch 2: training: 13.267585, validation: 13.000554
#> Loss at epoch 3: training: 11.814671, validation: 12.848388
#> Loss at epoch 4: training: 13.013980, validation: 13.733624
#> Loss at epoch 5: training: 11.797876, validation: 11.002292
#> Loss at epoch 6: training: 11.717374, validation: 14.515843
#> Loss at epoch 7: training: 11.115485, validation: 11.930919
#> Loss at epoch 8: training: 10.961864, validation: 10.763505
#> Loss at epoch 9: training: 11.855260, validation: 10.489598
#> Loss at epoch 10: training: 10.729342, validation: 9.940429


midae.newdata <- impute_new(path = file.path(tempdir(), "midaemodel.pt"),
    newdata = test.data, scaler = "minmax", m = 5)
```
