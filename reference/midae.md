# Multiple imputation through denoising autoencoders with dropout

This function is used to generate multiply-imputed datasets using
denoising autoencoders with Monte-Carlo dropout, early stopping and
predictive mean matching (PMM).

## Usage

``` r
midae(
  data,
  m = 5,
  categorical.encoding = "onehot",
  device = "cpu",
  epochs = 100,
  batch.size = 512,
  subsample = 1,
  early.stopping.epochs = 1,
  dae.params = list(),
  pmm.type = NULL,
  pmm.k = 5,
  pmm.link = "prob",
  pmm.save.vars = NULL,
  loss.na.scale = FALSE,
  verbose = TRUE,
  print.every.n = 1,
  save.model = FALSE,
  path = NULL
)
```

## Arguments

- data:

  A `data.frame`, `tibble` or `data.table` with missing values.

- m:

  The number of imputed datasets. Default: 5.

- categorical.encoding:

  The method for representing multi-class categorical features. Can be
  either `"embeddings"` or `"onehot"` (default).

- device:

  Device to use. Either `"cpu"` (default) or `"cuda"` for GPU.

- epochs:

  The number of training epochs (iterations). Default: 100.

- batch.size:

  The size of samples in each batch. Default: 512.

- subsample:

  The subsample ratio of training data. Default: 1.

- early.stopping.epochs:

  An integer value `k`. The training of midae will stop if the
  validation performance has not improved for `k` epochs, only used when
  `subsample` is less than 1. Default: 1.

- dae.params:

  A list of parameters for denoising autoencoders. See the documentation
  for the function [`dae_default()`](dae_default.md) for details.

- pmm.type:

  The type of predictive mean matching (PMM). Possible values:

  - `NULL` (default): Imputations without PMM;

  - `0`: Imputations with PMM type 0;

  - `1`: Imputations with PMM type 1;

  - `2`: Imputations with PMM type 2;

  - `"auto"`: Imputations with PMM type 2 for numeric/integer variables;
    imputations without PMM for categorical variables.

- pmm.k:

  The number of donors for predictive mean matching. Default: 5

- pmm.link:

  The link for predictive mean matching in binary variables:

  - `"prob"` (default): use probabilities;

  - `"logit"`: use logit values.

- pmm.save.vars:

  The names of variables whose predicted values of observed entries will
  be saved. Only use for PMM.

- loss.na.scale:

  Whether to multiply the ratio of missing values in a feature to
  calculate the loss function. Default: `FALSE`.

- verbose:

  Whether or not to print training loss information. Default: `TRUE`.

- print.every.n:

  If `verbose` is set to `TRUE`, print out training loss for every n
  epochs. Default: 1.

- save.model:

  Whether or not to save the imputation model. Default: `FALSE`.

- path:

  The path where the final imputation model will be saved.

## Examples

``` r
withNA.df <- createNA(data = iris, p = 0.2)
imputed.data <- midae(
  data = withNA.df, m = 5, epochs = 5, batch.size = 32,
  path = file.path(tempdir(), "midaemodel.pt")
)
#> [1] "cpu"
#> [1] "Running midae()."
#> Loss at epoch 1: 7.498226
#> Loss at epoch 2: 6.597657
#> Loss at epoch 3: 6.072527
#> Loss at epoch 4: 5.444873
#> Loss at epoch 5: 5.092110
```
