% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/midae.R
\name{dae_default}
\alias{dae_default}
\title{Auxiliary function for dae.params}
\usage{
dae_default(
  shuffle = TRUE,
  drop.last = FALSE,
  input.dropout = 0.2,
  hidden.dropout = 0.5,
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
  init.weight = "he.normal.elu.dropout",
  scaler = "standard",
  initial.imp = "sample",
  lower = 0.25,
  upper = 0.75
)
}
\arguments{
\item{shuffle}{Whether or not to shuffle training data. Default: \code{TRUE}.}

\item{drop.last}{Whether or not to drop the last batch. Default: \code{FALSE}.}

\item{input.dropout}{The dropout probability of the input layer. Default: 0.2.}

\item{hidden.dropout}{The dropout probability of the hidden layers. Default: 0.5.}

\item{optimizer}{The name of the optimizer. Options are : \code{"adamW"} (default), \code{"adam"}, \code{"adadelta"}, \code{"adagrad"}, \code{"rmsprop"}, or \code{"sgd"}.}

\item{learning.rate}{The learning rate. Default: 0.001.}

\item{weight.decay}{Weight decay (L2 penalty). Default: 0.01.}

\item{momentum}{Parameter for the \code{"sgd"} optimizer (default: 0). It is used for accelerating SGD in the relevant direction and dampens oscillations.}

\item{dampening}{Dampening for momentum (default: 0) used for the \code{"sgd"} optimizer.}

\item{eps}{A small positive value (default: 1e-08) used to prevent division by zero for optimizers \code{"adamW"}, \code{"adam"}, \code{"adadelta"},\code{"adagrad"} and \code{"rmsprop"}.}

\item{rho}{Parameter for the \code{"adadelta"} optimizer (default: 0.9). A coefficient used for computing a running average of squared gradients.}

\item{alpha}{Smoothing constant (default: 0.99) for the \code{"rmsprop"} optimizer.}

\item{learning.rate.decay}{Learning rate decay (default: 0) for the \code{"adagrad"} optimizer.}

\item{encoder.structure}{A vector indicating the structure of encoder. Default: c(256, 128, 64)}

\item{latent.dim}{Size of the latent layer. Default: 8.}

\item{decoder.structure}{A vector indicating the structure of decoder. Default: c(64, 128, 256)}

\item{act}{The name of activation function. Can be: \code{"relu"}, \code{"elu"} (default), \code{"leaky.relu"}, \code{"tanh"}, \code{"sigmoid"} and \code{"identity"}.}

\item{init.weight}{The distribution for weight initialization. Can be \code{"he.normal"}, \code{"he.uniform"}, \code{"xavier.uniform"}, \code{"xavier.normal"}, \code{"he.normal.dropout"}, \code{"he.normal.elu"}, \code{"he.normal.elu.dropout"} (default), \code{"he.normal.selu"} or \code{"he.normal.leaky.relu"}.}

\item{scaler}{The name of the scaler used for transforming numeric features. Can be \code{"standard"} (default), \code{"minmax"} , \code{"decile"}, \code{"robust"} or \code{"none"}.}

\item{initial.imp}{The method for initial imputation. Can be \code{"mean"}, \code{"median"} or \code{"sample"} (default).}

\item{lower}{The lower quantile (0.25 by default) for \code{scaler = "robust"}.}

\item{upper}{The upper quantile (0.75 by default) for \code{scaler = "robust"}.}
}
\description{
Auxiliary function for setting up the default dae-related hyperparameters for midae.
}
