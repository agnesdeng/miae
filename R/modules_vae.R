#Modules for variational autoencoders
#Set up a autoencoder imputer object with specified hyperparameters and then obtain multiple imputed datasets
vae <- nn_module(
  "vae",
  initialize = function(categorical.encoding, n.others, cardinalities, embedding.dim,
                        input.dropout, hidden.dropout, encoder.structure, latent.dim, decoder.structure, act) {

    self$categorical.encoding <- categorical.encoding

    if(!is.null(embedding.dim)){
      if(categorical.encoding == "embeddings"){
        self$embedder <- embedding_module(cardinalities, embedding.dim)
        n.features<-n.others+sum(embedding.dim)
      }else if(categorical.encoding == "onehot"){
        n.features<-n.others+sum(cardinalities)
      }else{
        stop(cat('categorical.encoding can only be either "embeddings" or "onehot".\n'))
      }
    }else{
      n.features<-n.others
    }



    n.outputs<-n.others+sum(cardinalities)

    self$input.dropout <- input.dropout
    self$hidden.dropout <- hidden.dropout
    self$embedding.dim <- embedding.dim

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
      }else if(act=="selu"){
        encoder.list[[2 * i]] <- nn_selu()
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
      }else if(act=="selu"){
        decoder.list[[2 * i]] <- nn_selu()
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

    num.dlayers <- length(decoder.structure)

    decoder.list[[2 * length(decoder.structure) + 1]] <- nn_linear(decoder.structure[num.dlayers], n.outputs)

    self$decoder <- nn_module_list(modules = decoder.list)

  },

  forward = function(num.tensor, logi.tensor, bin.tensor, cat.tensor) {

    if(!is.null(self$embedding.dim)){
      #multi-class
      if(self$categorical.encoding=="embeddings"){
        if(is.null(cat.tensor)){
          embedded.input<-cat.tensor
        }else{
          embedded.input<-self$embedder(cat.tensor)
        }
        L<-list(num.tensor, logi.tensor, bin.tensor,embedded.input)
        L<-Filter(Negate(is.null),L)
        x<-torch_cat(L,dim=2)
      }else{
        L<-list(num.tensor, logi.tensor, bin.tensor,cat.tensor)
        L<-Filter(Negate(is.null),L)
        x<-torch_cat(L,dim=2)
      }

    }else{
      #only numeric or binary
      L<-list(num.tensor, logi.tensor, bin.tensor)
      L<-Filter(Negate(is.null),L)
      x<-torch_cat(L,dim=2)


    }




    x <- nnf_dropout(input = x, p = self$input.dropout, inplace = FALSE)

    for (i in 1:length(self$encoder)) {
      x <- self$encoder[[i]](x)

      if (i %% 2 == 0) {
        x <- nnf_dropout(input = x, p = self$hidden.dropout, inplace = FALSE)
      }
    }

    mu <- self$mean(x)
    log.var <- self$log_var(x)
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
