#' Modules for variational autoencoders
#' @docType  module
#' @description Set up a autoencoder imputer object with specified hyperparameters and then obtain multiple imputed datasets
#' @format  NULL
#' @importFrom torch nn_module nn_sequential nn_linear nn_relu nn_sigmoid
#' @export

vae <- nn_module(
  "vae",
  initialize = function(n.features, latent.dim) {

    self$encoder <- nn_sequential(
      nn_linear(n.features, 128),
      nn_relu(),
      nn_linear(128, 64),
      nn_relu(),
      nn_linear(64, 32),
      nn_relu()
    )

    self$latent.dim <- latent.dim
    self$mean<-nn_linear(32,latent.dim)
    self$log_var<-nn_linear(32,latent.dim)



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
    y<-self$encoder(x)
    mu<-self$mean(y)
    log.var<-self$log_var(y)
    z<-mu+torch_exp(log.var$mul(0.5))*torch_randn(c(dim(x)[1],self$latent.dim))

    reconstr_x<-self$decoder(z)
    list("reconstrx"=reconstr_x,"mu"=mu,"log.var"=log.var)
  }
)
