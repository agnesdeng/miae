#' Modules for denoising autoencoders
#' @docType  module
#' @description Set up an denoising autoencoder imputer object with specified hyperparameters and then obtain multiple imputed datasets
#' @format  NULL
#' @importFrom torch nn_module nn_sequential nn_linear nn_relu nn_sigmoid
#' @export
#build an autoencoders
dae <- nn_module(
  "dae",
  initialize = function(n.features, latent.dim, dropout.prob) {

    self$dropout.prob <- dropout.prob

    self$encoder <- nn_sequential(

      nn_linear(n.features, 128),
      #nn_dropout(p = dropout.prob,inplace = FALSE),
      nn_relu(),
      # nn_dropout(p = dropout.prob,inplace = FALSE),
      nn_linear(128, 64),
      #nn_dropout(p = dropout.prob,inplace = FALSE),
      nn_relu(),
      #nn_dropout(p = dropout.prob,inplace = FALSE),
      nn_linear(64, 32),
      # nn_dropout(p = dropout.prob,inplace = FALSE),
      nn_relu(),
      #nn_dropout(p = dropout.prob,inplace = FALSE),
      nn_linear(32, latent.dim)
      # nn_dropout(p = dropout.prob,inplace = FALSE)
      #nn_relu()
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
    x<-nnf_dropout(input=x,p = self$dropout.prob,inplace = FALSE)
    x<-self$decoder(x)
    x
  }
)
