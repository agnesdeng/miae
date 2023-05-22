# A function use to obtain yhatobs using the whole dataset (use for pmm.type=1)
yhatobs_pmm1 <- function(data, na.loc, na.vars, extra.vars, pmm.link,
                         epochs = 5, batch.size = 32,
                         shuffle = TRUE,
                         optimizer = "adamW", learning.rate = 0.0001, weight.decay = 0.002, momentum = 0, eps = 1e-07,
                         encoder.structure = c(128, 64, 32), latent.dim = 16, decoder.structure = c(32, 64, 128),
                         act = "elu", init.weight = "xavier.normal", scaler = "none",
                         loss.na.scale = FALSE,
                         verbose = TRUE, print.every.n = 1) {
  # param xgb.params NULL if XGBmodels was fed in
  # return a list of yhatobs values for specified variables
  # check whether xgb.params contains sample related hyperparameters, need to coerce to 1 as we want to obtain yhatobs using the whole dataset



  # For PMM1, we need to use the whole dataset and no dropout to obtain donors

  subsample <- 1
  input.dropout <- 0
  hidden.dropout <- 0


  pre.obj <- preprocess(data, scaler = scaler)

  torch.data <- torch_dataset(data, scaler = scaler)


  n.features <- torch.data$.ncol()

  n.samples <- torch.data$.length()

  # use all available data
  train.dl <- torch::dataloader(dataset = torch.data, batch_size = batch.size, shuffle = shuffle)
  model <- dae(n.features = n.features, input.dropout = input.dropout, hidden.dropout = hidden.dropout, encoder.structure = encoder.structure, latent.dim = latent.dim, decoder.structure = encoder.structure, act = act)


  ## vae??


  if (init.weight == "xavier.normal") {
    model$apply(init_xavier_normal)
  } else if (init.weight == "xavier.uniform") {
    model$apply(init_xavier_uniform)
  } else if (init.weight == "xavier.midas") {
    model$apply(init_xavier_midas)
  }



  # define the loss function for different variables
  num_loss <- torch::nn_mse_loss(reduction = "mean")
  bin_loss <- torch::nn_bce_with_logits_loss(reduction = "mean")
  multi_loss <- torch::nn_cross_entropy_loss(reduction = "mean")



  # choose optimizer & learning rate
  if (optimizer == "adam") {
    optimizer <- torch::optim_adam(model$parameters, lr = learning.rate, weight_decay = weight.decay)
  } else if (optimizer == "sgd") {
    optimizer <- torch::optim_sgd(model$parameters, lr = learning.rate, momentum = momentum, weight_decay = weight.decay)
  } else if (optimizer == "adamW") {
    # torch default eps = 1e-08, tensorfolow default eps =1e-07
    optimizer <- torchopt::optim_adamw(model$parameters, lr = learning.rate, weight_decay = weight.decay, eps = eps)
  }


  # epochs: number of iterations

  for (epoch in seq_len(epochs)) {
    model$train()

    train.loss <- 0


    coro::loop(for (b in train.dl) { # loop over all batches in each epoch

      Out <- model(b$data)

      # numeric
      if(length(pre.obj$num)>0){
        num.cost <- vector("list", length = length(pre.obj$num))
        names(num.cost) <- pre.obj$num

        for (var in pre.obj$num) {
          obs.idx <- which(pre.obj$na.loc[as.array(b$index), var] != TRUE)
          num.cost[[var]] <- num_loss(input = Out[obs.idx, pre.obj$num.idx[[var]]], target = b$data[obs.idx, pre.obj$num.idx[[var]]])
        }



        if (loss.na.scale) {
          if (length(pre.obj$num) > 1) {
            na.ratios <- colMeans(pre.obj$na.loc[, pre.obj$num])
            num.cost <- mapply(`*`, num.cost, na.ratios)
            total.num.cost <- do.call(sum, num.cost)
          } else {
            na.ratio <- mean(pre.obj$na.loc[, pre.obj$num])
            num.cost <- torch_mul(num.cost[[1]], na.ratio)
            total.num.cost <- num.cost
          }
        } else {
          total.num.cost <- do.call(sum, num.cost)
        }
      }else{
        total.num.cost <- 0
      }

      # binary
      if(length(pre.obj$bin)>0){
        bin.cost <- vector("list", length = length(pre.obj$bin))
        names(bin.cost) <- pre.obj$bin

        for (var in pre.obj$bin) {
          obs.idx <- which(pre.obj$na.loc[as.array(b$index), var] != TRUE)
          bin.cost[[var]] <- bin_loss(input = Out[obs.idx, pre.obj$bin.idx[[var]]], target = b$data[obs.idx, pre.obj$bin.idx[[var]]])
        }

        if (loss.na.scale) {
          if (length(pre.obj$bin) > 1) {
            na.ratios <- colMeans(pre.obj$na.loc[, pre.obj$bin])
            bin.cost <- mapply(`*`, bin.cost, na.ratios)
            total.bin.cost <- do.call(sum, bin.cost)
          } else {
            na.ratio <- mean(pre.obj$na.loc[, pre.obj$bin])
            bin.cost <- torch_mul(bin.cost[[1]], na.ratio)
            total.bin.cost <- bin.cost
          }
        } else {
          total.bin.cost <- do.call(sum, bin.cost)
        }
      }else{
        total.bin.cost <- 0
      }



      # multiclass
      if(length(pre.obj$multi)>0){
        multi.cost <- vector("list", length = length(pre.obj$multi))
        names(multi.cost) <- pre.obj$multi

        for (var in pre.obj$multi) {
          obs.idx <- which(pre.obj$na.loc[as.array(b$index), var] != TRUE)
          multi.cost[[var]] <- multi_loss(input = Out[obs.idx, pre.obj$multi.idx[[var]]], target = torch::torch_argmax(b$data[obs.idx, pre.obj$multi.idx[[var]]], dim = 2))
        }

        if (loss.na.scale) {
          if (length(pre.obj$multi) > 1) {
            na.ratios <- colMeans(pre.obj$na.loc[, pre.obj$multi])
            multi.cost <- mapply(`*`, multi.cost, na.ratios)
            total.multi.cost <- do.call(sum, multi.cost)
          } else {
            na.ratio <- mean(pre.obj$na.loc[, pre.obj$multi])
            multi.cost <- torch_mul(multi.cost[[1]], na.ratio)
            total.multi.cost <- multi.cost
          }
        } else {
          total.multi.cost <- do.call(sum, multi.cost)
        }
      }else{
        total.multi.cost <- 0
      }

      # Total cost
      cost <- sum(total.num.cost, total.bin.cost, total.multi.cost)

      # zero out the gradients
      optimizer$zero_grad()

      cost$backward()

      # update params
      optimizer$step()


      batch.loss <- cost$item()
      train.loss <- train.loss + batch.loss
    })
  }


  model$eval()

  # The whole dataset
  eval_dl <- torch::dataloader(dataset = torch.data, batch_size = n.samples, shuffle = FALSE)


  wholebatch <- eval_dl %>%
    torch::dataloader_make_iter() %>%
    torch::dataloader_next()

  # imputed data

  output.data <- model(wholebatch$data)
  imp.data <- postprocess(output.data = output.data, pre.obj = pre.obj, scaler = scaler)
  #

  save.p <- length(na.vars) + length(extra.vars)
  yhatobs.list <- vector("list", save.p)
  names(yhatobs.list) <- c(na.vars, extra.vars)


  for(var in na.vars){
    if(var %in% pre.obj$num){
      yhatobs.list[[var]] <- imp.data[[var]][!na.loc[, var]]

    }else if(var %in% pre.obj$bin){
      var.idx<-pre.obj$bin.idx[[var]]
      if(pmm.link=="logit"){
        yhatobs.list[[var]]<-as_array(output.data[!na.loc[, var],var.idx])
      }else if (pmm.link=="prob"){
        transform_fn <- nn_sigmoid()
        yhatobs.list[[var]]<-as_array(transform_fn(output.data[!na.loc[, var],var.idx]))
      }else{
        stop("pmm.link has to be either `logit` or `prob`")
      }

    }else if(var %in% pre.obj$multi){
      var.idx<-pre.obj$multi.idx[[var]]
      #probability for each class of a multiclass variable
      yhatobs.list[[var]]<-as_array(output.data[!na.loc[, var],var.idx])
    }
  }





  if (!is.null(extra.vars)) {
    for (var in extra.vars) {

      if(var %in% pre.obj$num){
        yhatobs.list[[var]] <- imp.data[[var]]
      }else if(var %in% pre.obj$bin){
        var.idx<-pre.obj$bin.idx[[var]]
        if(pmm.link=="logit"){
          yhatobs.list[[var]]<-as_array(output.data[,var.idx])
        }else if (pmm.link=="prob"){
          transform_fn <- nn_sigmoid()
          yhatobs.list[[var]]<-as_array(transform_fn(output.data[,var.idx]))
        }else{
          stop("pmm.link has to be either `logit` or `prob`")
        }

      }else if(var %in% pre.obj$multi){
        var.idx<-pre.obj$multi.idx[[var]]
        #probability for each class of a multiclass variable
        yhatobs.list[[var]]<-as_array(output.data[,var.idx])

      }
    }
  }







  return(yhatobs.list)
}
