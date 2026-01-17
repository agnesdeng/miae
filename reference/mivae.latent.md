# multiple imputation through variational autoencoders (latent) haven't change default setting yet

multiple imputation through variational autoencoders (latent) haven't
change default setting yet

## Usage

``` r
mivae.latent(
  data,
  m = 5,
  categorical.encoding = "embeddings",
  device = "cpu",
  epochs = 5,
  batch.size = 32,
  subsample = 1,
  early.stopping.epochs = 1,
  vae.params = list(),
  pmm.params = list(),
  loss.na.scale = FALSE,
  verbose = TRUE,
  print.every.n = 1,
  save.model = FALSE,
  path = NULL
)
```

## Arguments

- data:

  A data frame, tibble or data table with missing values.

- m:

  The number of imputed datasets.

- categorical.encoding:

  The method for representing multi-class categorical features. Can be
  either "embeddings" or "onehot" (default).

- device:

  Device to use. Either "cpu" (default) or "cuda" for GPU.

- epochs:

  The number of training epochs (iterations). Default: 100.

- batch.size:

  The size of samples in each batch. Default: 512.

- subsample:

  The subsample ratio of training data. Default: 1.

- early.stopping.epochs:

  An integer value `k`. Mivae training will stop if the validation
  performance has not improved for `k` epochs, only used when
  `subsample`\<1. Default: 1.

- vae.params:

  A list of parameters for variational autoencoders. See
  [`vae_default`](vae_default.md) for details.

- pmm.params:

  A list of parameters for predictive mean matching. See
  `vae_pmm_default` for details.

- loss.na.scale:

  Whether to multiply the ratio of missing values in a feature to
  calculate the loss function. Default: FALSE.

- verbose:

  Whether or not to print training loss information. Default: TRUE.

- print.every.n:

  If verbose is set to TRUE, print out training loss for every n epochs.
  Default: 1.

- save.model:

  Whether or not to save the imputation model. Default: FALSE.

- path:

  The path where the final imputation model will be saved.
