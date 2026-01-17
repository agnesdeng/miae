# Auxiliary function for vae.params

Auxiliary function for setting up the default vae-related
hyperparameters for mivae.

## Usage

``` r
vae_default(
  shuffle = TRUE,
  drop.last = FALSE,
  beta = 1,
  input.dropout = 0,
  hidden.dropout = 0,
  optimizer = "adamW",
  learning.rate = 0.001,
  weight.decay = 0.01,
  momentum = 0,
  dampening = 0,
  eps = 1e-08,
  rho = 0.9,
  alpha = 0.99,
  learning.rate.decay = 0,
  encoder.structure = c(256, 128, 64),
  latent.dim = 8,
  decoder.structure = c(64, 128, 256),
  act = "elu",
  init.weight = "he.normal.elu",
  scaler = "standard",
  initial.imp = "sample",
  lower = 0.25,
  upper = 0.75
)
```

## Arguments

- shuffle:

  Whether or not to shuffle training data. Default: `TRUE`.

- drop.last:

  Whether or not to drop the last batch. Default: `FALSE`.

- beta:

  A regularized parameter. Default: 1.

- input.dropout:

  The dropout probability of the input layer. Default: 0.

- hidden.dropout:

  The dropout probability of the hidden layers. Default: 0.

- optimizer:

  The name of the optimizer. Options are : `"adamW"` (default),
  `"adam"`, `"adadelta"`, `"adagrad"`, `"rmsprop"`, or `"sgd"`.

- learning.rate:

  The learning rate. Default: 0.001.

- weight.decay:

  Weight decay (L2 penalty). Default: 0.01.

- momentum:

  Parameter for the `"sgd"` optimizer (default: 0). It is used for
  accelerating SGD in the relevant direction and dampens oscillations.

- dampening:

  Dampening for momentum (default: 0) used for the `"sgd"` optimizer.

- eps:

  A small positive value (default: 1e-08) used to prevent division by
  zero for optimizers `"adamW"`, `"adam"`, `"adadelta"`,`"adagrad"` and
  `"rmsprop"`.

- rho:

  Parameter for the `"adadelta"` optimizer (default: 0.9). A coefficient
  used for computing a running average of squared gradients.

- alpha:

  Smoothing constant (default: 0.99) for the `"rmsprop"` optimizer.

- learning.rate.decay:

  Learning rate decay (default: 0) for the `"adagrad"` optimizer.

- encoder.structure:

  A vector indicating the structure of encoder. Default: c(256, 128, 64)

- latent.dim:

  Size of the latent layer. Default: 8.

- decoder.structure:

  A vector indicating the structure of decoder. Default: c(64, 128, 256)

- act:

  The name of activation function. Can be: `"relu"`, `"elu"` (default),
  `"leaky.relu"`, `"tanh"`, `"sigmoid"` and `"identity"`.

- init.weight:

  The distribution for weight initialization. Can be `"he.normal"`,
  `"he.uniform"`, `"xavier.uniform"`, `"xavier.normal"`,
  `"he.normal.dropout"`, `"he.normal.elu"` (default),
  `"he.normal.elu.dropout"`, `"he.normal.selu"` or
  `"he.normal.leaky.relu"`.

- scaler:

  The name of the scaler used for transforming numeric features. Can be
  `"standard"` (default), `"minmax"` , `"decile"`, `"robust"` or
  `"none"`.

- initial.imp:

  The method for initial imputation. Can be `"mean"`, `"median"` or
  `"sample"` (default).

- lower:

  The lower quantile (0.25 by default) for `scaler = "robust"`.

- upper:

  The upper quantile (0.75 by default) for `scaler = "robust"`.
