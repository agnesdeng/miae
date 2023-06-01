#' Modules for variational autoencoders
#' @description Set up a autoencoder imputer object with specified hyperparameters and then obtain multiple imputed datasets
#' @format  NULL
#' @importFrom torch nn_module nn_sequential nn_linear nn_relu nn_sigmoid
#' @export
vae <- nn_module(
  "vae",
  initialize = function(n.features, input.dropout, hidden.dropout, encoder.structure, latent.dim, decoder.structure, act) {
    self$input.dropout <- input.dropout
    # self$latent.dropout <- latent.dropout
    self$hidden.dropout <- hidden.dropout

    # encoder
    encoder.list <- list()
    for (i in seq_along(encoder.structure)) {
      # for(size in encoder.structure){
      if (i == 1) {
        encoder.list[[i]] <- nn_linear(n.features, encoder.structure[i])
      } else {
        encoder.list[[2 * i - 1]] <- nn_linear(encoder.structure[i - 1], encoder.structure[i])
      }

      if (act == "relu") {
        encoder.list[[2 * i]] <- nn_relu()
      } else if (act == "elu") {
        encoder.list[[2 * i]] <- nn_elu()
      } else if (act == "identity") {
        encoder.list[[2 * i]] <- nn_identity()
      } else if (act == "tanh") {
        encoder.list[[2 * i]] <- nn_tanh()
      } else if (act == "sigmoid") {
        encoder.list[[2 * i]] <- nn_sigmoid()
      } else if (act == "leaky.relu") {
        encoder.list[[2 * i]] <- nn_leaky_relu(negative_slope = 0.01)
      }else{
        stop("This activation function is not supported yet")
      }
    }

    # encoder.list[[2*length(encoder.structure)+1]]<- nn_linear(encoder.structure[i],latent.dim)

    self$encoder <- nn_module_list(modules = encoder.list)



    self$latent.dim <- latent.dim
    self$mean <- nn_linear(encoder.structure[length(encoder.structure)], latent.dim)
    self$log_var <- nn_linear(encoder.structure[length(encoder.structure)], latent.dim)

    # decoder
    decoder.list <- list()
    for (i in seq_along(decoder.structure)) {
      # for(size in decoder.structure){
      if (i == 1) {
        decoder.list[[i]] <- nn_linear(latent.dim, decoder.structure[i])
      } else {
        decoder.list[[2 * i - 1]] <- nn_linear(decoder.structure[i - 1], decoder.structure[i])
      }

      if (act == "relu") {
        decoder.list[[2 * i]] <- nn_relu()
      } else if (act == "elu") {
        decoder.list[[2 * i]] <- nn_elu()
      } else if (act == "identity") {
        decoder.list[[2 * i]] <- nn_identity()
      } else if (act == "tanh") {
        decoder.list[[2 * i]] <- nn_tanh()
      } else if (act == "sigmoid") {
        decoder.list[[2 * i]] <- nn_sigmoid()
      } else if (act == "leaky.relu") {
        decoder.list[[2 * i]] <- nn_leaky_relu(negative_slope = 0.01)
      }else{
        stop("This activation function is not supported yet")
      }
    }

    decoder.list[[2 * length(decoder.structure) + 1]] <- nn_linear(decoder.structure[i], n.features)

    self$decoder <- nn_module_list(modules = decoder.list)
  },
  forward = function(x) {
    y <- nnf_dropout(input = x, p = self$input.dropout, inplace = FALSE)

    for (i in 1:length(self$encoder)) {
      y <- self$encoder[[i]](y)

      if (i %% 2 == 0) {
        y <- nnf_dropout(input = y, p = self$hidden.dropout, inplace = FALSE)
      }
    }

    mu <- self$mean(y)
    log.var <- self$log_var(y)
    # z <- mu + torch_exp(log.var$mul(0.5)) * torch_randn(c(dim(x)[1], self$latent.dim))

    ### reparameterization
    std <- torch_exp(log.var$mul(0.5))
    eps <- torch_randn_like(std)
    z <- mu + eps * std
    ###


    for (i in 1:length(self$decoder)) {
      z <- self$decoder[[i]](z)

      if (i %% 2 == 0) {
        z <- nnf_dropout(input = z, p = self$hidden.dropout, inplace = FALSE)
      }
    }
    list("reconstrx" = z, "mu" = mu, "log.var" = log.var)



    # y<-self$encoder(x)
    # mu<-self$mean(y)
    # log.var<-self$log_var(y)
    # z<-mu+torch_exp(log.var$mul(0.5))*torch_randn(c(dim(x)[1],self$latent.dim))

    # reconstr_x<-self$decoder(z)
    # list("reconstrx"=reconstr_x,"mu"=mu,"log.var"=log.var)
  }
)
