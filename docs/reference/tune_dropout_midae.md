# Tune dropout rate for midae

Tune dropout rate for midae

## Usage

``` r
tune_dropout_midae(
  data,
  dropout.grid = list(input.dropout = c(0, 0.25, 0.5), hidden.dropout = c(0, 0.25, 0.5)),
  m = 5,
  categorical.encoding = "embeddings",
  device = "cpu",
  epochs = 5,
  batch.size = 32,
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

  A dataset on which the midae model will be trained.

- dropout.grid:

  A list containing two vectors: `input.dropout` and `hidden.dropout`,
  each specifying the dropout rates to be tested for the input layer and
  hidden layers, respectively.

- m:

  The number of imputations to perform.

- categorical.encoding:

  The method used for encoding categorical variables. Defaults to
  "embeddings".

- device:

  The computing device to use, either "cpu" or "cuda" for GPU.

- epochs:

  The number of training epochs for each model.

- batch.size:

  The size of the batches used in training.

- subsample:

  The proportion of the data to be used in training. Defaults to 1,
  meaning the full dataset is used.

- early.stopping.epochs:

  The number of epochs with no improvement after which training will be
  stopped.

- dae.params:

  A list of parameters for the denoising autoencoder.

- loss.na.scale:

  Boolean flag indicating whether to scale the loss function based on NA
  values. Defaults to FALSE.

- verbose:

  Boolean flag to control the verbosity of the function's output.

- print.every.n:

  Specifies how often (in epochs) to print the training progress. Only
  relevant if verbose is TRUE.

- save.model:

  Boolean flag indicating whether to save the trained model.

- path:

  File path where the model should be saved if save.model is TRUE. If
  NULL and save.model is TRUE, the model is saved in the current
  directory.

- pmm.params:

  A list of parameters for predictive mean matching.

## Value

A list containing the tuned parameters and their corresponding
performance metrics.

## Examples

``` r
1+1
#> [1] 2
```
