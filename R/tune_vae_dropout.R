#'tuning dropout for midae
#' @param data A data frame, tibble or data table with missing values.
#' @param dropout.grid A list of tuning values for input.dropout and hidden.dropout
#' @param m The number of imputed datasets.
#' @param epochs The number of training epochs (iterations).
#' @param batch.size The size of samples in each batch.
#' @param split.ratio The ratio of training data. Default: 0.7.
#' @param shuffle Whether or not to shuffle training data. Default: TRUE
#' @param input.dropout The dropout probability of the input layer.
#' @param hidden.dropout The dropout probability of the hidden layers.
#' @param optimizer The name of the optimizer. Options are : "adamW" (default), "adam" and "sgd".
#' @param learning.rate The learning rate. The default value is 0.001.
#' @param weight.decay Weight decay (L2 penalty). The default value is 0.
#' @param momentum Parameter for "sgd" optimizer. It is used for accelerating SGD in the relevant direction and dampens oscillations.
#' @param encoder.structure A vector indicating the structure of encoder. Default: c(128,64,32)
#' @param latent.dim The size of latent layer. The default value is 8.
#' @param decoder.structure A vector indicating the structure of decoder. Default: c(32,64,128)
#' @param act The name of activation function. Can be: "relu", "elu", "leaky.relu", "tanh", "sigmoid" and "identity".
#' @param init.weight Techniques for weights initialization. Can be "xavier.uniform" or "kaiming.uniform".
#' @param scaler The name of scaler for transforming numeric features. Can be "standard", "minmax" or "none".
#' @param verbose Whether or not to print training loss information. Default: TRUE.
#' @param print.every.n If verbose is set to TRUE, print out training loss for every n epochs.
#' @param save.model Whether or not to save the imputation model. Default: FALSE.
#' @param path The path where the final imputation model will be saved.
#' @importFrom torch dataloader nn_mse_loss nn_bce_with_logits_loss nn_cross_entropy_loss optim_adam optim_sgd torch_save torch_load torch_argmax dataloader_make_iter dataloader_next
#' @importFrom torchopt optim_adamw
#' @export
tune_vae_dropout<-function(data, dropout.grid = list(input.dropout=c(0.2,0.5,0.7),hidden.dropout=c(0.3, 0.6, 0.8)),
                           m=5, epochs = 5, batch.size = 32,
                           split.ratio = 0.7, shuffle = TRUE,
                           optimizer = "adamW", learning.rate = 0.0001, weight.decay = 0, momentum = 0,
                           encoder.structure = c(128, 64, 32), latent.dim = 8, decoder.structure = c(32, 64, 128),
                           act = "elu", init.weight="xavier.normal", scaler= "minmax",
                           verbose = TRUE, print.every.n = 1, save.model = FALSE, path = NULL){




    if(save.model & is.null(path)){
      stop("Please specify a path to save the imputation model.")
    }


    optim.name<-optimizer

    pre.obj <- preprocess(data, scaler = scaler)

    torch.data <- torch_dataset(data, scaler = scaler)


    n.features <- torch.data$.ncol()

    n.samples <- torch.data$.length()


    ###
    train.idx <- sample(1:n.samples, size = floor(split.ratio*n.samples), replace = FALSE)
    valid.idx <- setdiff(1:n.samples, train.idx)

    train.ds <- torch_dataset_idx(data=data,idx=train.idx, scaler = scaler)
    valid.ds <- torch_dataset_idx(data=data,idx=valid.idx, scaler = scaler)

    train.dl<- dataloader(dataset = train.ds,batch_size = batch.size, shuffle = shuffle)
    valid.dl<- dataloader(dataset = valid.ds,batch_size = batch.size, shuffle = FALSE)

    train.size <- length(train.dl)
    valid.size <- length(valid.dl)



    # define the loss function for different variables
    num_loss <- torch::nn_mse_loss(reduction = "mean")
    bin_loss <- torch::nn_bce_with_logits_loss(reduction = "mean")
    multi_loss <- torch::nn_cross_entropy_loss(reduction = "mean")


    # epochs: number of iterations
    model.params<-expand.grid(dropout.grid)
    n.models<-nrow(model.params)


    Model.list<-vector("list", length=n.models)

    #pre-allocate
    #imputed.missing<-replicate(n.models, list())

    na.loc <- pre.obj$na.loc
    na.vars <- pre.obj$ordered.names[colSums(na.loc) != 0]
    num.navars<-length(na.vars)


    imputed.missing<-replicate(num.navars, list())
    names(imputed.missing)<-na.vars

    for (var in na.vars) {
      imputed.missing[[var]]<- vector("list", length = n.models)
      n.na<-sum(na.loc[,var])

      for(i in seq_len(n.models)){
        imputed.missing[[var]][[i]]<-data.frame(matrix(NA,nrow=n.na,ncol=m))

        colnames(imputed.missing[[var]][[i]])<-paste0("m",1:m)
      }

    }


    #runs models
    for(i in seq_len(n.models)){


      model <-vae(n.features = n.features, input.dropout = model.params$input.dropout[i], hidden.dropout = model.params$hidden.dropout[i],
                  encoder.structure = encoder.structure,latent.dim = latent.dim, decoder.structure = encoder.structure, act = act)

      if(init.weight=="xavier.normal"){
        model$apply(init_xavier_normal)
      }else if(init.weight=="xavier.uniform"){
        model$apply(init_xavier_uniform)
      }else if(init.weight=="xavier.midas"){
        model$apply(init_xavier_midas)
      }







      # choose optimizer & learning rate
      if(optim.name=="adam"){
        optimizer <- torch::optim_adam(model$parameters, lr = learning.rate, weight_decay = weight.decay)
      }else if(optim.name=="sgd"){
        optimizer <- torch::optim_sgd(model$parameters, lr = learning.rate, momentum = momentum, weight_decay = weight.decay)
      }else if(optim.name=="adamW"){
        optimizer <- torchopt::optim_adamw(model$parameters, lr = learning.rate, weight_decay = weight.decay)
      }






      for (epoch in seq_len(epochs)) {

        model$train()

        train.loss <- 0



        coro::loop(for (b in train.dl) { # loop over all batches in each epoch



          Out <- model(b$data)

          # numeric
          num.cost <- vector("list", length = length(pre.obj$num))
          names(num.cost) <- pre.obj$num

          for (var in pre.obj$num){
            obs.idx<-which(pre.obj$na.loc[as.array(b$index),var]!=TRUE)
            num.cost[[var]] <- num_loss(input = Out$reconstrx[obs.idx, pre.obj$num.idx[[var]]], target = b$data[obs.idx, pre.obj$num.idx[[var]]])
          }

          total.num.cost <- do.call(sum, num.cost)

          # binary
          bin.cost <- vector("list", length = length(pre.obj$bin))
          names(bin.cost) <- pre.obj$bin

          for (var in pre.obj$bin) {
            obs.idx<-which(pre.obj$na.loc[as.array(b$index),var]!=TRUE)
            bin.cost[[var]] <- bin_loss(input = Out$reconstrx[obs.idx, pre.obj$bin.idx[[var]]], target = b$data[obs.idx, pre.obj$bin.idx[[var]]])
          }

          total.bin.cost <- do.call(sum, bin.cost)

          # multiclass
          multi.cost <- vector("list", length = length(pre.obj$multi))
          names(multi.cost) <- pre.obj$multi

          for (var in pre.obj$multi) {
            obs.idx<-which(pre.obj$na.loc[as.array(b$index),var]!=TRUE)
            multi.cost[[var]] <- multi_loss(input = Out$reconstrx[obs.idx, pre.obj$multi.idx[[var]]], target = torch::torch_argmax(b$data[obs.idx, pre.obj$multi.idx[[var]]], dim = 2))
          }
          total.multi.cost <- do.call(sum, multi.cost)

          # Total cost
          cost <- sum(total.num.cost, total.bin.cost, total.multi.cost)


          # KL loss
          mu <- Out$mu
          log.var <- Out$log.var

          #
          # kl.div =  1 + log.var - mu$pow(2) - log.var$exp()
          # kl.div.sum = - 0.5 *kl.div$sum()

          kl.divergence <- torch_mean(-0.5 * torch_sum(1 + log.var - mu$pow(2) - log.var$exp()))

          total.cost <- cost + kl.divergence


          #zero out the gradients
          optimizer$zero_grad()

          total.cost$backward()

          #update params
          optimizer$step()


          train.loss <- train.loss + total.cost$item()

          if (save.model & epoch == epochs) {
            Model.list[[i]]<-torch::torch_save(model, path = path)
          }
        })


        model$eval()
        valid.loss<-0

        #validation loss
        coro::loop(for (b in valid.dl){
          Out <- model(b$data)

          # numeric
          num.cost <- vector("list", length = length(pre.obj$num))
          names(num.cost) <- pre.obj$num

          for (var in pre.obj$num){
            obs.idx<-which(pre.obj$na.loc[as.array(b$index),var]!=TRUE)
            num.cost[[var]] <- num_loss(input = Out$reconstrx[obs.idx, pre.obj$num.idx[[var]]], target = b$data[obs.idx, pre.obj$num.idx[[var]]])
          }

          total.num.cost <- do.call(sum, num.cost)

          # binary
          bin.cost <- vector("list", length = length(pre.obj$bin))
          names(bin.cost) <- pre.obj$bin

          for (var in pre.obj$bin) {
            obs.idx<-which(pre.obj$na.loc[as.array(b$index),var]!=TRUE)
            bin.cost[[var]] <- bin_loss(input = Out$reconstrx[obs.idx, pre.obj$bin.idx[[var]]], target = b$data[obs.idx, pre.obj$bin.idx[[var]]])
          }

          total.bin.cost <- do.call(sum, bin.cost)

          # multiclass
          multi.cost <- vector("list", length = length(pre.obj$multi))
          names(multi.cost) <- pre.obj$multi

          for (var in pre.obj$multi) {
            obs.idx<-which(pre.obj$na.loc[as.array(b$index),var]!=TRUE)
            multi.cost[[var]] <- multi_loss(input = Out$reconstrx[obs.idx, pre.obj$multi.idx[[var]]], target = torch::torch_argmax(b$data[obs.idx, pre.obj$multi.idx[[var]]], dim = 2))
          }
          total.multi.cost <- do.call(sum, multi.cost)

          # Total cost
          cost <- sum(total.num.cost, total.bin.cost, total.multi.cost)

          # KL loss
          mu <- Out$mu
          log.var <- Out$log.var

          #
          # kl.div =  1 + log.var - mu$pow(2) - log.var$exp()
          # kl.div.sum = - 0.5 *kl.div$sum()

          kl.divergence <- torch_mean(-0.5 * torch_sum(1 + log.var - mu$pow(2) - log.var$exp()))

          total.cost <- cost + kl.divergence



          valid.loss <- valid.loss + total.cost$item()



          valid.loss <- valid.loss + cost$item()





        })



        if(verbose & (epoch ==1 | epoch %% print.every.n == 0)){
          cat(sprintf("Loss at epoch %d: training: %3f, validation: %3f\n", epoch, train.loss / train.size, valid.loss / valid.size))
        }

        #model <- torch::torch_load(path = path)

      }
      # The whole dataset
      eval_dl <- torch::dataloader(dataset = torch.data, batch_size = n.samples, shuffle = FALSE)


      wholebatch <- eval_dl %>%
        torch::dataloader_make_iter() %>%
        torch::dataloader_next()



      # imputed data
      Model.list[[i]]<-model




      for (j in seq_len(m)) {
        output.list <- model(wholebatch$data)
        imp.data <- postprocess(output.data = output.list$reconstrx, pre.obj = pre.obj, scaler = scaler)
        for (var in na.vars) {
          imputed.missing[[var]][[i]][,j]<- imp.data[[var]][na.loc[, var]]
        }
      }


      for(var in na.vars){
        K<-nrow(imputed.missing[[var]][[i]])
        imputed.missing[[var]][[i]]<-imputed.missing[[var]][[i]] %>%
          tibble::add_column(input.dropout=rep(model.params$input.dropout[i],K),
                             hidden.dropout=rep(model.params$hidden.dropout[i],K)
          )
      }




    }#end of n models


    observed.data<-vector("list",length=num.navars)
    names(observed.data)<-na.vars

    for(var in na.vars){
      observed.data[[var]]<-data[[var]][!is.na(data[[var]])]
    }

    if(save.model){
      return(list("imputed.missing"=imputed.missing,"observed.data"=observed.data,"Model.list"=Model.list))
    }else{
      return(list("imputed.missing"=imputed.missing,"observed.data"=observed.data))
    }





















}
