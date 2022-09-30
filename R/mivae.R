#' multiple imputation through variational autoencoders with dropout
#' @param data A data frame, tibble or data table with missing values.
#' @param m The number of imputed datasets.
#' @param epochs The number of training epochs (iterations).
#' @param batch.size The size of samples in each batch.
#' @param input.dropout The dropout probability of the input layer.
#' @param latent.dropout The dropout probability of the latent layer.
#' @param hidden.dropout The dropout probability of the hidden layers.
#' @param optimizer The name of the optimizer. Options are : "adam" (default), and "sgd".
#' @param learning.rate The learning rate. The default value is 0.001.
#' @param weight.decay Weight decay (L2 penalty). The default value is 0.
#' @param momentum Parameter for "sgd" optimizer. It is used for accelerating SGD in the relevant direction and dampens oscillations.
#' @param encoder.structure A vector indicating the structure of encoder. Default: c(128,64,32)
#' @param latent.dim The size of latent layer. The default value is 8.
#' @param decoder.structure A vector indicating the structure of decoder. Default: c(32,64,128)
#' @param verbose Whether or not to print training loss information. Default: TRUE.
#' @param print.every.n If verbose is set to TRUE, print out training loss for every n epochs.
#' @param save.model Whether or not to save the imputation model. Default: FALSE.
#' @param path The path where the final imputation model will be saved.
#' @importFrom torch dataloader torch_mean nn_mse_loss nn_bce_with_logits_loss nn_cross_entropy_loss optim_adam optim_sgd torch_save torch_load torch_argmax dataloader_make_iter dataloader_next
#' @export
#' @examples
#' withNA.df <- createNA(data = iris,p = 0.2)
#' imputed.data <- mivae(data = withNA.df, m = 5, epochs = 5, path = file.path(tempdir(),"mivaemodel.pt")
mivae <- function(data, m = 5, epochs = 10, batch.size = 50,
                  input.dropout = 0, latent.dropout = 0, hidden.dropout = 0,
                  optimizer = "adam", learning.rate = 0.001, weight.decay = 0, momentum = 0,
                  encoder.structure = c(128, 64, 32), latent.dim = 8, decoder.structure = c(32, 64, 128),
                  verbose = TRUE, print.every.n = 1, save.model = FALSE, path = NULL) {


  if(save.model & is.null(path)){
    stop("Please specify a path to save the imputation model.")
  }

  pre.obj <- preprocess(data)

  torch.data <- torch_dataset(data)


  n.features <- torch.data$.ncol()
  n.samples <- torch.data$.length()


  ###
  train.idx <- sample(1:n.samples, size = floor(0.7*n.samples), replace = FALSE)
  valid.idx <- setdiff(1:n.samples, train.idx)

  train.ds <- torch_dataset_idx(data,train.idx)
  valid.ds <- torch_dataset_idx(data,valid.idx)

  train.dl<- dataloader(dataset = train.ds,batch_size = batch.size, shuffle = TRUE)
  valid.dl<- dataloader(dataset = valid.ds,batch_size = batch.size, shuffle = FALSE)

  train.size <- length(train.ds)
  valid.size <- length(valid.ds)

  #dl <- torch::dataloader(dataset = torch.data, batch_size = batch.size, shuffle = TRUE)
  model <-  vae(n.features = n.features, latent.dim = latent.dim, input.dropout = input.dropout, latent.dropout = latent.dropout, hidden.dropout = hidden.dropout, encoder.structure = encoder.structure, decoder.structure = encoder.structure)


  # define the loss function for different variables
  num_loss <- torch::nn_mse_loss(reduction = "sum")
  bin_loss <- torch::nn_bce_with_logits_loss(reduction = "sum")
  multi_loss <- torch::nn_cross_entropy_loss(reduction = "sum")


  # choose optimizer & learning rate
  if(optimizer=="adam"){
    optimizer <- torch::optim_adam(model$parameters, lr = learning.rate, weight_decay = weight.decay)
  }else if(optimizer=="sgd"){
    optimizer <- torch::optim_sgd(model$parameters, lr = learning.rate, momentum = momentum, weight_decay = weight.decay)
  }


  # epochs: number of iterations

  for (epoch in seq_len(epochs)) {

    model$train()

    train.loss <- 0

    coro::loop(for (b in train.dl) {  # loop over all batches in each epoch

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

      # Total cost (reconstruction loss)
      cost <- sum(total.num.cost, total.bin.cost, total.multi.cost)

      # KL loss
      mu <- Out$mu
      log.var <- Out$log.var

      #
      # kl.div =  1 + log.var - mu$pow(2) - log.var$exp()
      # kl.div.sum = - 0.5 *kl.div$sum()

      kl.divergence <- torch_mean(-0.5 * torch_sum(1 + log.var - mu$pow(2) - log.var$exp()))

      total.cost <- cost + kl.divergence

      #
      optimizer$zero_grad()
      total.cost$backward()
      optimizer$step()


      batch.loss <- total.cost$item()
      train.loss <- train.loss + batch.loss

      if (save.model & epoch == epochs) {
        torch::torch_save(model, path = path)
      }
    })

    model$eval()
    valid.loss<-0

    coro::loop(for (b in valid.dl) {  # loop over all batches in each epoch

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

      # Total cost (reconstruction loss)
      cost <- sum(total.num.cost, total.bin.cost, total.multi.cost)

      # KL loss
      mu <- Out$mu
      log.var <- Out$log.var

      #
      # kl.div =  1 + log.var - mu$pow(2) - log.var$exp()
      # kl.div.sum = - 0.5 *kl.div$sum()

      kl.divergence <- torch_mean(-0.5 * torch_sum(1 + log.var - mu$pow(2) - log.var$exp()))

      total.cost <- cost + kl.divergence


      batch.loss <- total.cost$item()
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
    output.list <- model(wholebatch$data)
    imp.data <- postprocess(output.data = output.list$reconstrx, pre.obj = pre.obj)
    na.vars <- pre.obj$ordered.names[colSums(na.loc) != 0]

    for (var in na.vars) {

        data[[var]][na.loc[, var]] <- imp.data[[var]][na.loc[, var]]

    }

    imputed.data[[i]] <- data
  }
  imputed.data
}



#' multiple imputation through variational autoencoders with dropout
#' @param data A data frame, tibble or data table with missing values.
#' @param m The number of imputed datasets.
#' @param epochs The number of training epochs (iterations).
#' @param batch.size The size of samples in each batch.
#' @param input.dropout The dropout probability of the input layer.
#' @param latent.dropout The dropout probability of the latent layer.
#' @param hidden.dropout The dropout probability of the hidden layers.
#' @param optimizer The name of the optimizer. Options are : "adam" (default), and "sgd".
#' @param learning.rate The learning rate. The default value is 0.001.
#' @param weight.decay Weight decay (L2 penalty). The default value is 0.
#' @param momentum Parameter for "sgd" optimizer. It is used for accelerating SGD in the relevant direction and dampens oscillations.
#' @param encoder.structure A vector indicating the structure of encoder. Default: c(128,64,32)
#' @param latent.dim The size of latent layer. The default value is 8.
#' @param decoder.structure A vector indicating the structure of decoder. Default: c(32,64,128)
#' @param verbose Whether or not to print training loss information. Default: TRUE.
#' @param print.every.n If verbose is set to TRUE, print out training loss for every n epochs.
#' @param path The path where the final imputation model will be saved.
#' @importFrom torch dataloader torch_mean nn_mse_loss nn_bce_with_logits_loss nn_cross_entropy_loss optim_adam optim_sgd torch_save torch_load torch_argmax dataloader_make_iter dataloader_next
#' @export
#' @examples
#' withNA.df <- createNA(data = iris,p = 0.2)
#' imputed.data <- mivae(data = withNA.df, m = 5, epochs = 5, path = file.path(tempdir(),"mivaemodel.pt")
mivae0 <- function(data, m = 5, epochs = 10, batch.size = 50,
                  input.dropout = 0, latent.dropout = 0, hidden.dropout = 0,
                  optimizer = "adam", learning.rate = 0.001, weight.decay = 0, momentum = 0,
                  encoder.structure = c(128, 64, 32), latent.dim = 8, decoder.structure = c(32, 64, 128),
                  verbose = TRUE, print.every.n = 1, path = NULL) {


  if(is.null(path)){
    stop("Please specify a path to save the imputation model.")
  }

  pre.obj <- preprocess(data)

  torch.data <- torch_dataset(data)


  n.features <- torch.data$.ncol()
  n.samples <- torch.data$.length()



  dl <- torch::dataloader(dataset = torch.data, batch_size = batch.size, shuffle = TRUE)


  model <-  vae(n.features = n.features, latent.dim = latent.dim, input.dropout = input.dropout, latent.dropout = latent.dropout, hidden.dropout = hidden.dropout, encoder.structure = encoder.structure, decoder.structure = encoder.structure)


  # define the loss function for different variables
  num_loss <- torch::nn_mse_loss(reduction = "mean")
  bin_loss <- torch::nn_bce_with_logits_loss()
  multi_loss <- torch::nn_cross_entropy_loss()


  # choose optimizer & learning rate
  if(optimizer=="adam"){
    optimizer <- torch::optim_adam(model$parameters, lr = learning.rate, weight_decay = weight.decay)
  }else if(optimizer=="sgd"){
    optimizer <- torch::optim_sgd(model$parameters, lr = learning.rate, momentum = momentum, weight_decay = weight.decay)
  }



  # epochs: number of iterations

  for (epoch in seq_len(epochs)) {

    model$train()

    epoch.loss <- 0

    coro::loop(for (b in dl) { # loop over all minibatches for one epoch

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

      # Total cost (reconstruction loss)
      cost <- sum(total.num.cost, total.bin.cost, total.multi.cost)

      # KL loss
      mu <- Out$mu
      log.var <- Out$log.var

      #
      # kl.div =  1 + log.var - mu$pow(2) - log.var$exp()
      # kl.div.sum = - 0.5 *kl.div$sum()

      kl.divergence <- torch_mean(-0.5 * torch_sum(1 + log.var - mu$pow(2) - log.var$exp()))

      total.cost <- cost + kl.divergence

      #
      optimizer$zero_grad()
      total.cost$backward()
      optimizer$step()


      batch.loss <- total.cost$item()
      epoch.loss <- epoch.loss + batch.loss

      if (epoch == epochs) {
        torch::torch_save(model, path = path)
      }
    })

    if(verbose & (epoch ==1 | epoch %% print.every.n == 0)){
      cat(sprintf("Loss at epoch %d: %1f\n", epoch, epoch.loss / length(dl)))
    }
  }


  model <- torch::torch_load(path = path)

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
    output.list <- model(wholebatch$data)
    imp.data <- postprocess(output.data = output.list$reconstrx, pre.obj = pre.obj)
    na.vars <- pre.obj$ordered.names[colSums(na.loc) != 0]

    for (var in na.vars) {

      data[[var]][na.loc[, var]] <- imp.data[[var]][na.loc[, var]]

    }

    imputed.data[[i]] <- data
  }
  imputed.data
}
