#' Modules for denoising autoencoders
#' @description Set up an denoising autoencoder imputer object with specified hyperparameters and then obtain multiple imputed datasets
#' @format  NULL
#' @importFrom torch nn_module nn_sequential nn_linear nn_relu nn_sigmoid
#' @export
# build an autoencoders
dae <- nn_module(
  "dae",
  initialize = function(n.features, input.dropout, latent.dropout, hidden.dropout, encoder.structure,latent.dim, decoder.structure) {

     self$input.dropout <- input.dropout
     self$latent.dropout <- latent.dropout
     self$hidden.dropout <- hidden.dropout

    # encoder
    encoder.list <- list()
    for (i in seq_along(encoder.structure)) {
      # for(size in encoder.structure){
      if (i == 1) {
        encoder.list[[2 * i - 1]] <- nn_linear(n.features, encoder.structure[i])
      } else {
        encoder.list[[2 * i - 1]] <- nn_linear(encoder.structure[i - 1], encoder.structure[i])
      }

      encoder.list[[2 * i]] <- nn_relu()
    }

    encoder.list[[2 * length(encoder.structure) + 1]] <- nn_linear(encoder.structure[i], latent.dim)

    self$encoder <- nn_module_list(modules = encoder.list)


    # decoder
    decoder.list <- list()
    for (i in seq_along(decoder.structure)) {
      # for(size in decoder.structure){
      if (i == 1) {
        decoder.list[[2 * i - 1]] <- nn_linear(latent.dim, decoder.structure[i])
      } else {
        decoder.list[[2 * i - 1]] <- nn_linear(decoder.structure[i - 1], decoder.structure[i])
      }


        decoder.list[[2 * i]] <- nn_relu()



    }

    decoder.list[[2 * length(decoder.structure)+1]] <- nn_linear(decoder.structure[i], n.features)

    self$decoder <- nn_module_list(modules = decoder.list)
  },

  forward = function(x) {

    x<-nnf_dropout(input = x, p = self$input.dropout, inplace = FALSE)

    for (i in 1:length(self$encoder)) {
      x <- self$encoder[[i]](x)
    }

    x <- nnf_dropout(input = x, p = self$latent.dropout, inplace = FALSE)

    for (i in 1:length(self$decoder)) {
      x <- self$decoder[[i]](x)
    }

    x
  }
)
