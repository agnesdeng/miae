#' Modules for denoising autoencoders
#' @description Set up an denoising autoencoder imputer object with specified hyperparameters and then obtain multiple imputed datasets
#' @format  NULL
#' @importFrom torch nn_module nn_sequential nn_linear nn_relu nn_sigmoid
#' @export
# build an autoencoders
dae1 <- nn_module(
  "dae",
  initialize = function(n.features, input.dropout, latent.dropout, hidden.dropout, encoder.structure,latent.dim, decoder.structure, act) {

    #self$input.dropout <- input.dropout
    #self$latent.dropout <- latent.dropout
    #self$hidden.dropout <- hidden.dropout

    self$input_dropout<-nn_dropout(p = input.dropout, inplace = FALSE)
    self$latent_dropout<-nn_dropout(p = latent.dropout, inplace = FALSE)
    self$hidden_dropout<-nn_dropout(p = hidden.dropout, inplace = FALSE)

    # encoder
    encoder.list <- list()
    for (i in seq_along(encoder.structure)) {
      # for(size in encoder.structure){
      if (i == 1) {
        encoder.list[[i]] <- nn_linear(n.features, encoder.structure[i])
        #nn_init_xavier_uniform_(encoder.list[[2 * i - 1]]$weight, gain = 1/sqrt(6))

      } else {
        encoder.list[[2 * i - 1]] <- nn_linear(encoder.structure[i - 1], encoder.structure[i])
        # nn_init_xavier_uniform_(encoder.list[[2 * i - 1]]$weight, gain = 1/sqrt(6))
      }

      if(act=="relu"){
        encoder.list[[2 * i]] <- nn_relu()
      }else if(act=="elu"){
        encoder.list[[2 * i]] <- nn_elu()
      }

    }

    encoder.list[[2 * length(encoder.structure) + 1]] <- nn_linear(encoder.structure[i], latent.dim)
    #nn_init_xavier_uniform_(encoder.list[[2 * length(encoder.structure) + 1]])

    self$encoder <- nn_module_list(modules = encoder.list)


    # decoder
    decoder.list <- list()
    for (i in seq_along(decoder.structure)) {
      # for(size in decoder.structure){
      if (i == 1) {
        decoder.list[[2 * i - 1]] <- nn_linear(latent.dim, decoder.structure[i])
        #nn_init_xavier_uniform_(decoder.list[[2 * i - 1]]$weight, gain = 1/sqrt(6))
      } else {
        decoder.list[[2 * i - 1]] <- nn_linear(decoder.structure[i - 1], decoder.structure[i])
        #nn_init_xavier_uniform_(decoder.list[[2 * i - 1]]$weight, gain = 1/sqrt(6))
      }

      if(act=="relu"){
        decoder.list[[2 * i]] <- nn_relu()
      }else if (act=="elu"){
        decoder.list[[2 * i]] <- nn_elu()
      }



    }

    decoder.list[[2 * length(decoder.structure)+1]] <- nn_linear(decoder.structure[i], n.features)
    #nn_init_xavier_uniform_(decoder.list[[2 * length(decoder.structure)+1]])
    self$decoder <- nn_module_list(modules = decoder.list)
  },

  forward = function(x) {

    x<-self$input_dropout(input = x)

    for (i in 1:length(self$encoder)) {

      if(i %% 2 ==1){
        x <-self$hidden_dropout(input = x)
      }
      x <- self$encoder[[i]](x)

    }

    x <- self$latent_dropout(input = x)

    for (i in 1:length(self$decoder)) {
      if(i %% 2 ==1){
        x <-self$hidden_dropout(input = x)
      }
      x <- self$decoder[[i]](x)
    }

    x
  }
)
