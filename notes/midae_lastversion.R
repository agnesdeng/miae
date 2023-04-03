#' multiple imputation through denoising autoencoders with dropout (show train and valid loss)
#' @param data A data frame, tibble or data table with missing values.
#' @param m The number of imputed datasets.
#' @param epochs The number of training epochs (iterations).
#' @param batch.size The size of samples in each batch. Default: 32.
#' @param sampling The sampling ratio of training data. Default: 1.
#' @param shuffle Whether or not to shuffle training data. Default: TRUE
#' @param input.dropout The dropout probability of the input layer.
#' @param hidden.dropout The dropout probability of the hidden layers.
#' @param optimizer The name of the optimizer. Options are : "adamW" (default), "adam" and "sgd".
#' @param learning.rate The learning rate. The default value is 0.001.
#' @param weight.decay Weight decay (L2 penalty). The default value is 0.
#' @param momentum Parameter for "sgd" optimizer. It is used for accelerating SGD in the relevant direction and dampens oscillations.
#' @param encoder.structure A vector indicating the structure of encoder. Default: c(128,64,32)
#' @param eps A small positive value used to prevent division by zero for the "adamW" optimizer. Default: 1e-07.
#' @param decoder.structure A vector indicating the structure of decoder. Default: c(32,64,128)
#' @param act The name of activation function. Can be: "relu", "elu", "leaky.relu", "tanh", "sigmoid" and "identity".
#' @param init.weight Techniques for weights initialization. Can be "xavier.uniform" or "kaiming.uniform".
#' @param scaler The name of scaler for transforming numeric features. Can be "standard", "minmax" ,"decile" or "none".
#' @param verbose Whether or not to print training loss information. Default: TRUE.
#' @param print.every.n If verbose is set to TRUE, print out training loss for every n epochs.
#' @param save.model Whether or not to save the imputation model. Default: FALSE.
#' @param path The path where the final imputation model will be saved.
#' @importFrom torch dataloader nn_mse_loss nn_bce_with_logits_loss nn_cross_entropy_loss optim_adam optim_sgd torch_save torch_load torch_argmax dataloader_make_iter dataloader_next
#' @importFrom torchopt optim_adamw
#' @export
#' @examples
#' withNA.df <- createNA(data = iris,p = 0.2)
#' imputed.data <- midae(data = withNA.df, m = 5, epochs = 5, path = file.path(tempdir(),"midaemodel.pt")
midae <- function(data, m = 5, epochs = 5, batch.size = 32,
                  sampling = 1, shuffle = TRUE,
                  input.dropout = 0.2, hidden.dropout = 0.5,
                  optimizer = "adamW", learning.rate = 0.0001, weight.decay = 0.002, momentum = 0, eps = 1e-07,
                  encoder.structure = c(128, 64, 32), decoder.structure = c(32, 64, 128),
                  act = "elu", init.weight="xavier.normal", scaler= "none",
                  verbose = TRUE, print.every.n = 1, save.model = FALSE, path = NULL) {

  if(save.model & is.null(path)){
    stop("Please specify a path to save the imputation model.")
  }


  pre.obj <- preprocess(data, scaler = scaler)

  torch.data <- torch_dataset(data, scaler = scaler)


  n.features <- torch.data$.ncol()

  n.samples <- torch.data$.length()


  ###
  train.idx <- sample(1:n.samples, size = floor(sampling*n.samples), replace = FALSE)
  valid.idx <- setdiff(1:n.samples, train.idx)

  train.ds <- torch_dataset_idx(data=data,idx=train.idx, scaler = scaler)
  valid.ds <- torch_dataset_idx(data=data,idx=valid.idx, scaler = scaler)

  train.dl<- dataloader(dataset = train.ds,batch_size = batch.size, shuffle = shuffle)
  valid.dl<- dataloader(dataset = valid.ds,batch_size = batch.size, shuffle = FALSE)

  train.size <- length(train.dl)
  valid.size <- length(valid.dl)
  ###

  model <- dae(n.features = n.features, input.dropout = input.dropout, hidden.dropout = hidden.dropout, encoder.structure = encoder.structure, decoder.structure = encoder.structure, act = act)



  if(init.weight=="xavier.normal"){
    model$apply(init_xavier_normal)
  }else if(init.weight=="xavier.uniform"){
    model$apply(init_xavier_uniform)
  }else if(init.weight=="xavier.midas"){
    model$apply(init_xavier_midas)
  }



  # define the loss function for different variables
  num_loss <- torch::nn_mse_loss(reduction = "mean")
  bin_loss <- torch::nn_bce_with_logits_loss(reduction = "mean")
  multi_loss <- torch::nn_cross_entropy_loss(reduction = "mean")



  # choose optimizer & learning rate
  if(optimizer=="adam"){
    optimizer <- torch::optim_adam(model$parameters, lr = learning.rate, weight_decay = weight.decay)
  }else if(optimizer=="sgd"){
    optimizer <- torch::optim_sgd(model$parameters, lr = learning.rate, momentum = momentum, weight_decay = weight.decay)
  }else if(optimizer=="adamW"){
    #torch default eps = 1e-08, tensorfolow default eps =1e-07
    optimizer <- torchopt::optim_adamw(model$parameters, lr = learning.rate, weight_decay = weight.decay,  eps = eps)
  }




  # epochs: number of iterations

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
      num.cost[[var]] <- num_loss(input = Out[obs.idx, pre.obj$num.idx[[var]]], target = b$data[obs.idx, pre.obj$num.idx[[var]]])
      }

      total.num.cost <- do.call(sum, num.cost)

      # binary
      bin.cost <- vector("list", length = length(pre.obj$bin))
      names(bin.cost) <- pre.obj$bin

      for (var in pre.obj$bin) {
        obs.idx<-which(pre.obj$na.loc[as.array(b$index),var]!=TRUE)
        bin.cost[[var]] <- bin_loss(input = Out[obs.idx, pre.obj$bin.idx[[var]]], target = b$data[obs.idx, pre.obj$bin.idx[[var]]])
      }

      total.bin.cost <- do.call(sum, bin.cost)

      # multiclass
      multi.cost <- vector("list", length = length(pre.obj$multi))
      names(multi.cost) <- pre.obj$multi

      for (var in pre.obj$multi) {
        obs.idx<-which(pre.obj$na.loc[as.array(b$index),var]!=TRUE)
        multi.cost[[var]] <- multi_loss(input = Out[obs.idx, pre.obj$multi.idx[[var]]], target = torch::torch_argmax(b$data[obs.idx, pre.obj$multi.idx[[var]]], dim = 2))
      }
      total.multi.cost <- do.call(sum, multi.cost)

      # Total cost
      cost <- sum(total.num.cost, total.bin.cost, total.multi.cost)

      #zero out the gradients
      optimizer$zero_grad()

      cost$backward()

      #update params
      optimizer$step()


      batch.loss <- cost$item()
      train.loss <- train.loss + batch.loss

      if (save.model & epoch == epochs) {
        torch::torch_save(model, path = path)
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
        num.cost[[var]] <- num_loss(input = Out[obs.idx, pre.obj$num.idx[[var]]], target = b$data[obs.idx, pre.obj$num.idx[[var]]])
      }

      total.num.cost <- do.call(sum, num.cost)

      # binary
      bin.cost <- vector("list", length = length(pre.obj$bin))
      names(bin.cost) <- pre.obj$bin

      for (var in pre.obj$bin) {
        obs.idx<-which(pre.obj$na.loc[as.array(b$index),var]!=TRUE)
        bin.cost[[var]] <- bin_loss(input = Out[obs.idx, pre.obj$bin.idx[[var]]], target = b$data[obs.idx, pre.obj$bin.idx[[var]]])
      }

      total.bin.cost <- do.call(sum, bin.cost)

      # multiclass
      multi.cost <- vector("list", length = length(pre.obj$multi))
      names(multi.cost) <- pre.obj$multi

      for (var in pre.obj$multi) {
        obs.idx<-which(pre.obj$na.loc[as.array(b$index),var]!=TRUE)
        multi.cost[[var]] <- multi_loss(input = Out[obs.idx, pre.obj$multi.idx[[var]]], target = torch::torch_argmax(b$data[obs.idx, pre.obj$multi.idx[[var]]], dim = 2))
      }
      total.multi.cost <- do.call(sum, multi.cost)

      # Total cost
      cost <- sum(total.num.cost, total.bin.cost, total.multi.cost)


      batch.loss <- cost$item()
      valid.loss <- valid.loss + batch.loss





    })



    if(verbose & (epoch ==1 | epoch %% print.every.n == 0)){
      cat(sprintf("Loss at epoch %d: training: %3f, validation: %3f\n", epoch, train.loss / train.size, valid.loss / valid.size))
    }


    }



  #model <- torch::torch_load(path = path)
  model$eval()

  # The whole dataset
  eval_dl <- torch::dataloader(dataset = torch.data, batch_size = n.samples, shuffle = FALSE)


  wholebatch <- eval_dl %>%
    torch::dataloader_make_iter() %>%
    torch::dataloader_next()

  # imputed data
  imputed.data <- vector("list", length = m)
  na.loc <- pre.obj$na.loc

  for (i in seq_len(m)) {
    output.data <- model(wholebatch$data)
    imp.data <- postprocess(output.data = output.data, pre.obj = pre.obj, scaler = scaler)
    na.vars <- pre.obj$ordered.names[colSums(na.loc) != 0]

    for (var in na.vars) {

        data[[var]][na.loc[, var]] <- imp.data[[var]][na.loc[, var]]

    }

    imputed.data[[i]] <- data
  }
  imputed.data
}


#' multiple imputation through denoising autoencoders with dropout (all data as training data)
#' @param data A data frame, tibble or data table with missing values.
#' @param m The number of imputed datasets.
#' @param epochs The number of training epochs (iterations).
#' @param batch.size The size of samples in each batch.
#' @param shuffle Whether or not to shuffle training data. Default: TRUE
#' @param input.dropout The dropout probability of the input layer.
#' @param hidden.dropout The dropout probability of the hidden layers.
#' @param optimizer The name of the optimizer. Options are : "adam" (default), and "sgd".
#' @param learning.rate The learning rate. The default value is 0.001.
#' @param weight.decay Weight decay (L2 penalty). The default value is 0.
#' @param momentum Parameter for "sgd" optimizer. It is used for accelerating SGD in the relevant direction and dampens oscillations.
#' @param encoder.structure A vector indicating the structure of encoder. Default: c(128,64,32)
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
#' @examples
#' withNA.df <- createNA(data = iris,p = 0.2)
#' imputed.data <- midae(data = withNA.df, m = 5, epochs = 5, path = file.path(tempdir(),"midaemodel.pt")
midae0 <- function(data, m = 5, epochs = 5, batch.size = 16,
                   shuffle = TRUE,
                  input.dropout = 0.2, hidden.dropout = 0.5,
                  optimizer = "adamW", learning.rate = 0.0001, weight.decay = 0.002, momentum = 0,
                  encoder.structure = c(128, 64, 32), decoder.structure = c(32, 64, 128),
                  act = "elu", init.weight="xavier.normal", scaler="none",
                  verbose = TRUE, print.every.n = 1, save.model = FALSE, path = NULL) {

  if(save.model & is.null(path)){
    stop("Please specify a path to save the imputation model.")
  }


  pre.obj <- preprocess(data,scaler = scaler)

  torch.data <- torch_dataset(data,scaler = scaler)


  n.features <- torch.data$.ncol()

  n.samples <- torch.data$.length()

  dl <- torch::dataloader(dataset = torch.data, batch_size = batch.size, shuffle = shuffle)
  ###
  #train.idx <- sample(1:n.samples, size = floor(0.7*n.samples), replace = FALSE)
  #valid.idx <- setdiff(1:n.samples, train.idx)

  #train.ds <- torch_dataset_idx(data,train.idx)
  #valid.ds <- torch_dataset_idx(data,valid.idx)

  #train.dl<- dataloader(dataset = train.ds,batch_size = batch.size, shuffle = TRUE)
 # valid.dl<- dataloader(dataset = valid.ds,batch_size = batch.size, shuffle = FALSE)

 # train.size <- length(train.ds)
  #valid.size <- length(valid.ds)
  ###

  #dl <- torch::dataloader(dataset = torch.data, batch_size = batch.size, shuffle = TRUE)
  model <- dae(n.features = n.features, input.dropout = input.dropout, hidden.dropout = hidden.dropout, encoder.structure = encoder.structure, decoder.structure = encoder.structure, act = act)



  if(init.weight=="xavier.normal"){
    model$apply(init_xavier_normal)
  }else if(init.weight=="xavier.uniform"){
    model$apply(init_xavier_uniform)
  }else if(init.weight=="xavier.midas"){
    model$apply(init_xavier_midas)
  }



  # define the loss function for different variables
  num_loss <- torch::nn_mse_loss(reduction = "mean")
  bin_loss <- torch::nn_bce_with_logits_loss(reduction = "mean")
  multi_loss <- torch::nn_cross_entropy_loss(reduction = "mean")


  # choose optimizer & learning rate
  if(optimizer=="adam"){
    optimizer <- torch::optim_adam(model$parameters, lr = learning.rate, weight_decay = weight.decay)
  }else if(optimizer=="sgd"){
    optimizer <- torch::optim_sgd(model$parameters, lr = learning.rate, momentum = momentum, weight_decay = weight.decay)
  }else if(optimizer=="adamW"){
    #torch default eps = 1e-08, tensorfolow default eps =1e-07
    optimizer <- torchopt::optim_adamw(model$parameters, lr = learning.rate, weight_decay = weight.decay,  eps = eps)
  }



  # epochs: number of iterations

  for (epoch in seq_len(epochs)) {

    model$train()

    train.loss <- 0



    coro::loop(for (b in dl) { # loop over all batches in each epoch

       #zero out the gradients
      optimizer$zero_grad()

      Out <- model(b$data)

      # numeric
      num.cost <- vector("list", length = length(pre.obj$num))
      names(num.cost) <- pre.obj$num

      for (var in pre.obj$num){
        obs.idx<-which(pre.obj$na.loc[as.array(b$index),var]!=TRUE)

        if(length(obs.idx)==0){
          cat(paste(paste(var,"obs.idx:"),obs.idx))
        }

        num.cost[[var]] <- torch_sqrt(num_loss(input = Out[obs.idx, pre.obj$num.idx[[var]],NULL], target = b$data[obs.idx, pre.obj$num.idx[[var]],NULL]))
      }

      total.num.cost <- do.call(sum, num.cost)

      # binary
      bin.cost <- vector("list", length = length(pre.obj$bin))
      names(bin.cost) <- pre.obj$bin

      for (var in pre.obj$bin) {
        obs.idx<-which(pre.obj$na.loc[as.array(b$index),var]!=TRUE)
        bin.cost[[var]] <- bin_loss(input = Out[obs.idx, pre.obj$bin.idx[[var]]], target = b$data[obs.idx, pre.obj$bin.idx[[var]]])
      }

      total.bin.cost <- do.call(sum, bin.cost)

      # multiclass
      multi.cost <- vector("list", length = length(pre.obj$multi))
      names(multi.cost) <- pre.obj$multi

      for (var in pre.obj$multi) {
        obs.idx<-which(pre.obj$na.loc[as.array(b$index),var]!=TRUE)
        multi.cost[[var]] <- multi_loss(input = Out[obs.idx, pre.obj$multi.idx[[var]]], target = torch::torch_argmax(b$data[obs.idx, pre.obj$multi.idx[[var]]], dim = 2))
      }
      total.multi.cost <- do.call(sum, multi.cost)

      # Total cost
      cost <- sum(total.num.cost, total.bin.cost, total.multi.cost)



      cost$backward()

      #update params
      optimizer$step()



      train.loss <- train.loss + cost$item()

      if (save.model & epoch == epochs) {
        torch::torch_save(model, path = path)
      }
    })





    if(verbose & (epoch ==1 | epoch %% print.every.n == 0)){
      cat(sprintf("Loss at epoch %d: %1f\n", epoch, train.loss / length(dl)))
    }


  }



  #model <- torch::torch_load(path = path)
  model$eval()


  # The whole dataset
  eval_dl <- torch::dataloader(dataset = torch.data, batch_size = n.samples, shuffle = FALSE)


  wholebatch <- eval_dl %>%
    torch::dataloader_make_iter() %>%
    torch::dataloader_next()

  # imputed data
  imputed.data <- vector("list", length = m)
  na.loc <- pre.obj$na.loc

  for (i in seq_len(m)) {
    output.data <- model(wholebatch$data)
    imp.data <- postprocess(output.data = output.data, pre.obj = pre.obj,scaler = scaler)
    na.vars <- pre.obj$ordered.names[colSums(na.loc) != 0]

    for (var in na.vars) {

      data[[var]][na.loc[, var]] <- imp.data[[var]][na.loc[, var]]

    }

    imputed.data[[i]] <- data
  }
  imputed.data
}

