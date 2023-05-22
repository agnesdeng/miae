#' Modules for denoising autoencoders
#' @description Set up an denoising autoencoder imputer object with specified hyperparameters and then obtain multiple imputed datasets
#' @format  NULL
#' @importFrom torch nn_module nn_sequential nn_linear nn_relu nn_sigmoid
#' @export
# build an autoencoders
dae <- nn_module(
  "dae",
  initialize = function(n.features, input.dropout, hidden.dropout, encoder.structure, latent.dim, decoder.structure, act) {

     self$input.dropout <- input.dropout

     self$hidden.dropout <- hidden.dropout

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
      }else if(act=="identity"){
        encoder.list[[2 * i]] <- nn_identity()
      }else if(act=="tanh"){
        encoder.list[[2 * i]] <- nn_tanh()
      }else if(act=="sigmoid"){
        encoder.list[[2 * i]] <- nn_sigmoid()
      }else if(act=="leaky.relu"){
        encoder.list[[2 * i]] <- nn_leaky_relu(negative_slope = 0.01)
      }else{
        stop("This activation function is not supported yet")
      }



    }

    num.elayers <- length(encoder.structure)

    encoder.list[[2 * num.elayers + 1]] <- nn_linear(encoder.structure[num.elayers], latent.dim)

    #encoder.list[[2 * length(encoder.structure) + 1]] <- nn_linear(encoder.structure[i], latent.dim)
    #nn_init_xavier_uniform_(encoder.list[[2 * length(encoder.structure) + 1]])

    self$encoder <- nn_module_list(modules = encoder.list)


    # decoder
    decoder.list <- list()
    for (i in seq_along(decoder.structure)) {
      # for(size in decoder.structure){
      if (i == 1) {
        decoder.list[[i]] <- nn_linear(latent.dim, decoder.structure[i])
        #nn_init_xavier_uniform_(decoder.list[[2 * i - 1]]$weight, gain = 1/sqrt(6))
      } else {
        decoder.list[[2 * i - 1]] <- nn_linear(decoder.structure[i - 1], decoder.structure[i])
        #nn_init_xavier_uniform_(decoder.list[[2 * i - 1]]$weight, gain = 1/sqrt(6))
      }

      if(act=="relu"){
        decoder.list[[2 * i]] <- nn_relu()
      }else if (act=="elu"){
        decoder.list[[2 * i]] <- nn_elu()
      }else if(act=="identity"){
        decoder.list[[2 * i]] <- nn_identity()
      }else if(act=="tanh"){
        decoder.list[[2 * i]] <- nn_tanh()
      }else if(act=="sigmoid"){
        decoder.list[[2 * i]] <- nn_sigmoid()
      }else if(act=="leaky.relu"){
        decoder.list[[2 * i]] <- nn_leaky_relu(negative_slope = 0.01)
      }else{
        stop("This activation function is not supported yet")
      }



    }

    num.dlayers <- length(decoder.structure)
    decoder.list[[2 * num.dlayers+1]] <- nn_linear(decoder.structure[num.dlayers], n.features)
    #nn_init_xavier_uniform_(decoder.list[[2 * length(decoder.structure)+1]])
    self$decoder <- nn_module_list(modules = decoder.list)
  },

  forward = function(x) {

    x<-nnf_dropout(input = x, p = self$input.dropout, inplace = FALSE)

    for (i in 1:length(self$encoder)) {

      x <- self$encoder[[i]](x)

      if(i %% 2 ==0){
        x <-nnf_dropout(input = x, p = self$hidden.dropout,inplace= FALSE)
      }


    }

   # x <- nnf_dropout(input = x, p = self$latent.dropout, inplace = FALSE)

    for (i in 1:length(self$decoder)) {

      x <- self$decoder[[i]](x)

      if(i %% 2 ==0){
        x <-nnf_dropout(input = x, p = self$hidden.dropout,inplace= FALSE)
      }

    }

    x
  }
)



#' Modules for denoising autoencoders (without latent dim)
#' @description Set up an denoising autoencoder imputer object with specified hyperparameters and then obtain multiple imputed datasets
#' @format  NULL
#' @importFrom torch nn_module nn_sequential nn_linear nn_relu nn_sigmoid
#' @export
# build an autoencoders
dae0 <- nn_module(
  "dae",
  initialize = function(n.features, input.dropout, hidden.dropout, encoder.structure, decoder.structure, act) {

    self$input.dropout <- input.dropout

    self$hidden.dropout <- hidden.dropout

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
      }else if(act=="identity"){
        encoder.list[[2 * i]] <- nn_identity()
      }else if(act=="tanh"){
        encoder.list[[2 * i]] <- nn_tanh()
      }else if(act=="sigmoid"){
        encoder.list[[2 * i]] <- nn_sigmoid()
      }else if(act=="leaky.relu"){
        encoder.list[[2 * i]] <- nn_leaky_relu(negative_slope = 0.01)
      }else{
        stop("This activation function is not supported yet")
      }



    }

    #encoder.list[[2 * length(encoder.structure) + 1]] <- nn_linear(encoder.structure[i], latent.dim)
    #nn_init_xavier_uniform_(encoder.list[[2 * length(encoder.structure) + 1]])

    self$encoder <- nn_module_list(modules = encoder.list)


    # decoder
    decoder.list <- list()
    for (i in seq_along(decoder.structure)) {
      # for(size in decoder.structure){
      if (i == 1) {
        decoder.list[[i]] <- nn_linear(encoder.structure[length(encoder.structure)], decoder.structure[i])
        #nn_init_xavier_uniform_(decoder.list[[2 * i - 1]]$weight, gain = 1/sqrt(6))
      } else {
        decoder.list[[2 * i - 1]] <- nn_linear(decoder.structure[i - 1], decoder.structure[i])
        #nn_init_xavier_uniform_(decoder.list[[2 * i - 1]]$weight, gain = 1/sqrt(6))
      }

      if(act=="relu"){
        decoder.list[[2 * i]] <- nn_relu()
      }else if (act=="elu"){
        decoder.list[[2 * i]] <- nn_elu()
      }else if(act=="identity"){
        decoder.list[[2 * i]] <- nn_identity()
      }else if(act=="tanh"){
        decoder.list[[2 * i]] <- nn_tanh()
      }else if(act=="sigmoid"){
        decoder.list[[2 * i]] <- nn_sigmoid()
      }else if(act=="leaky.relu"){
        decoder.list[[2 * i]] <- nn_leaky_relu(negative_slope = 0.01)
      }else{
        stop("This activation function is not supported yet")
      }



    }

    decoder.list[[2 * length(decoder.structure)+1]] <- nn_linear(decoder.structure[i], n.features)
    #nn_init_xavier_uniform_(decoder.list[[2 * length(decoder.structure)+1]])
    self$decoder <- nn_module_list(modules = decoder.list)
  },

  forward = function(x) {

    x<-nnf_dropout(input = x, p = self$input.dropout, inplace = FALSE)

    for (i in 1:length(self$encoder)) {

      x <- self$encoder[[i]](x)

      if(i %% 2 ==0){
        x <-nnf_dropout(input = x, p = self$hidden.dropout,inplace= FALSE)
      }


    }

    # x <- nnf_dropout(input = x, p = self$latent.dropout, inplace = FALSE)

    for (i in 1:length(self$decoder)) {

      x <- self$decoder[[i]](x)

      if(i %% 2 ==0){
        x <-nnf_dropout(input = x, p = self$hidden.dropout,inplace= FALSE)
      }

    }

    x
  }
)
