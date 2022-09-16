#' Modules for autoencoders
#' @docType  module
#' @description Set up a autoencoder imputer object with specified hyperparameters and then obtain multiple imputed datasets
#' @format  NULL
#' @importFrom torch nn_module nn_sequential nn_linear nn_relu nn_sigmoid
#' @export
#build an autoencoders
ae <- nn_module(
  "autoencoder",
  initialize = function(n.features, latent.dim,encoder.structure,decoder.structure) {



    self$encoder <- nn_sequential(
      nn_linear(n.features, 128),
      nn_relu(),
      nn_linear(128, 64),
      nn_relu(),
      nn_linear(64, 32),
      nn_relu(),
      nn_linear(32, latent.dim)
    )

    self$decoder <- nn_sequential(
      nn_linear(latent.dim, 32),
      nn_relu(),
      nn_linear(32, 64),
      nn_relu(),
      nn_linear(64, 128),
      nn_relu(),
      nn_linear(128, n.features)
    )


  },

  forward = function(x) {
    x<-self$encoder(x)
    x<-self$decoder(x)
    x
  }
)
