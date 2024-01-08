# A function use to obtain yhatobs using the whole dataset (use for pmm.type=1) (Note: currently this function only support midae, as mivae doesn't need pmm. Will include it later.)
yhatobs_pmm1 <- function(module = "dae", data, categorical.encoding, device,
                         na.loc, na.vars, extra.vars, pmm.link,
                         epochs = 5, batch.size = 32,
                         shuffle = TRUE, drop.last,
                         optimizer = "adamW", learning.rate = 0.001, weight.decay = 0.01, momentum = 0, dampening = 0, eps = 1e-08, rho = 0.9, alpha = 0.99, learning.rate.decay = 0,
                         encoder.structure = c(256, 128, 64), latent.dim = 8, decoder.structure = c(64, 128, 256),
                         act = "elu", init.weight = "he.normal.elu.dropout", scaler = "standard", initial.imp = "sample", lower = 0.25, upper = 0.75,
                         loss.na.scale = FALSE,
                         verbose = TRUE, print.every.n = 1) {
  # param xgb.params NULL if XGBmodels was fed in
  # return a list of yhatobs values for specified variables
  # check whether xgb.params contains sample related hyperparameters, need to coerce to 1 as we want to obtain yhatobs using the whole dataset



  # For PMM1, we need to use the whole dataset and no dropout to obtain donors

  subsample <- 1
  input.dropout <- 0
  hidden.dropout <- 0


  pre.obj <- preprocess(data, scaler = scaler, lower = lower, upper = upper, categorical.encoding = categorical.encoding, initial.imp = initial.imp)
  cardinalities <- pre.obj$cardinalities
  embedding.dim <- pre.obj$embedding.dim

  # n.num+n.logi+n.bin
  origin.names <- colnames(data)
  n.others <- length(origin.names) - length(cardinalities)


  # torch.data <- torch_dataset(data, scaler = scaler)
  data.tensor <- torch_dataset(data, scaler = scaler, lower = lower, upper = upper, categorical.encoding = categorical.encoding, initial.imp = initial.imp)

  # n.features <- torch.data$.ncol()

  n.samples <- nrow(data)

  # use all available data
  # train.dl <- torch::dataloader(dataset = torch.data, batch_size = batch.size, shuffle = shuffle)

  train.samples <- n.samples
  train.idx <- 1:n.samples
  # train.idx<-sample(1:n.samples, size = n.samples, replace = FALSE)
  train.batches <- batch_set(n.samples = train.samples, batch.size = batch.size, drop.last = drop.last)
  train.batch.set <- train.batches$batch.set
  train.num.batches <- train.batches$num.batches
  # data.tensor<-torch_dataset(data, scaler = scaler, device = device)
  train.original.data <- data.tensor

  # model <- dae(n.features = n.features, input.dropout = input.dropout, hidden.dropout = hidden.dropout, encoder.structure = encoder.structure, latent.dim = latent.dim, decoder.structure = encoder.structure, act = act)


  if (module == "dae") {
    # mivae doesn't need pmm
    model <- dae(
      categorical.encoding = categorical.encoding, n.others = n.others, cardinalities = cardinalities, embedding.dim = embedding.dim,
      input.dropout = input.dropout, hidden.dropout = hidden.dropout, encoder.structure = encoder.structure, latent.dim = latent.dim, decoder.structure = decoder.structure, act = act
    )$to(device = device)
  } else {
    model <- vae(
      categorical.encoding = categorical.encoding, n.others = n.others, cardinalities = cardinalities, embedding.dim = embedding.dim,
      input.dropout = input.dropout, hidden.dropout = hidden.dropout, encoder.structure = encoder.structure, latent.dim = latent.dim, decoder.structure = decoder.structure, act = act
    )
  }
  model <- model$to(device = device)



  if (init.weight == "he.normal") {
    model$apply(init_he_normal)
  } else if (init.weight == "he.normal.dropout") {
    model$apply(init_he_normal_dropout)
  } else if (init.weight == "he.uniform") {
    model$apply(init_he_uniform)
  } else if (init.weight == "he.normal.elu") {
    model$apply(init_he_normal_elu)
  } else if (init.weight == "he.normal.elu.dropout") {
    model$apply(init_he_normal_elu_dropout)
  } else if (init.weight == "he.normal.selu") {
    model$apply(init_he_normal_selu)
  } else if (init.weight == "he.normal.leaky.relu") {
    model$apply(init_he_normal_leaky.relu)
  } else if (init.weight == "xavier.normal") {
    model$apply(init_xavier_normal)
  } else if (init.weight == "xavier.uniform") {
    model$apply(init_xavier_uniform)
  } else if (init.weight == "xavier.midas") {
    model$apply(init_xavier_midas)
  } else {
    stop("This weight initialization is not supported yet")
  }



  # define the loss function for different variables
  num_loss <- nn_mse_loss(reduction = "mean")
  logi_loss <- nn_bce_with_logits_loss(reduction = "mean")
  bin_loss <- nn_bce_with_logits_loss(reduction = "mean")
  multi_loss <- nn_cross_entropy_loss(reduction = "mean")



  # choose optimizer & learning rate
  if (optimizer == "adamW") {
    optimizer <- torchopt::optim_adamw(model$parameters, lr = learning.rate, eps = eps, weight_decay = weight.decay)
  } else if (optimizer == "sgd") {
    optimizer <- optim_sgd(model$parameters, lr = learning.rate, momentum = momentum, dampening = dampening, weight_decay = weight.decay)
  } else if (optimizer == "adam") { # torch default eps = 1e-08, tensorfolow default eps =1e-07
    optimizer <- optim_adam(model$parameters, lr = learning.rate, eps = eps, weight_decay = weight.decay)
  } else if (optimizer == "adadelta") {
    optimizer <- optim_adadelta(model$parameters, lr = learning.rate, rho = rho, eps = eps, weight_decay = weight.decay)
  } else if (optimizer == "adagrad") {
    optimizer <- optim_adagrad(model$parameters, lr = learning.rate, lr_decay = learning.rate.decay, eps = eps, weight_decay = weight.decay)
  } else if (optimizer == "rmsprop") {
    optimizer <- optim_rmsprop(model$parameters, lr = learning.rate, alpha = alpha, eps = eps, weight_decay = weight.decay, momentum = momentum)
  }


  # epochs: number of iterations
  if (verbose) {
    print("Running midae for obtaining yhatobs when pmm.type = 1. Note that in this run, subsample = 1, and no dropout applied.")
  }




  for (epoch in seq_len(epochs)) {
    model$train()

    train.loss <- 0

    # rearrange all the data in each epoch
    if (shuffle) {
      permute <- torch_randperm(train.samples) + 1L
    } else {
      permute <- torch_tensor(1:train.samples)
    }

    train.data <- train.original.data[permute]


    for (i in 1:train.num.batches) {
      b <- list()
      b.index <- train.batch.set[[i]]

      b$data <- lapply(train.data, function(x) x[b.index])
      # index in original full data
      b$index <- train.idx[as.array(permute)[b.index]]

      num.tensor <- move_to_device(tensor = b$data$num.tensor, device = device)
      logi.tensor <- move_to_device(tensor = b$data$logi.tensor, device = device)
      bin.tensor <- move_to_device(tensor = b$data$bin.tensor, device = device)
      multi.tensor <- move_to_device(tensor = b$data$multi.tensor, device = device)
      onehot.tensor <- move_to_device(tensor = b$data$onehot.tensor, device = device)

      if (categorical.encoding == "embeddings") {
        Out <- model(num.tensor = num.tensor, logi.tensor = logi.tensor, bin.tensor = bin.tensor, cat.tensor = multi.tensor)
      } else if (categorical.encoding == "onehot") {
        Out <- model(num.tensor = num.tensor, logi.tensor = logi.tensor, bin.tensor = bin.tensor, cat.tensor = onehot.tensor)
      } else {
        stop(cat('categorical.encoding can only be either "embeddings" or "onehot".\n'))
      }


      if (module == "vae") {
        Out <- Out$reconstrx
      }

      # numeric
      if (length(pre.obj$num) > 0) {
        num.cost <- vector("list", length = length(pre.obj$num))
        names(num.cost) <- pre.obj$num

        for (idx in seq_along(pre.obj$num.idx)) {
          var <- pre.obj$num[idx]
          obs.idx <- which(pre.obj$na.loc[as.array(b$index), var] != TRUE)
          num.cost[[var]] <- num_loss(input = Out[obs.idx, pre.obj$num.idx[[var]]], target = num.tensor[obs.idx, idx])
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
      } else {
        total.num.cost <- torch_zeros(1)
      }

      # logical
      if (length(pre.obj$logi) > 0) {
        logi.cost <- vector("list", length = length(pre.obj$logi))
        names(logi.cost) <- pre.obj$logi

        for (idx in seq_along(pre.obj$logi)) {
          var <- pre.obj$logi[idx]
          obs.idx <- which(pre.obj$na.loc[as.array(b$index), var] != TRUE)
          logi.cost[[var]] <- logi_loss(input = Out[obs.idx, pre.obj$logi.idx[[var]]], target = logi.tensor[obs.idx, idx])
        }


        if (loss.na.scale) {
          if (length(pre.obj$logi) > 1) {
            na.ratios <- colMeans(pre.obj$na.loc[, pre.obj$logi])
            logi.cost <- mapply(`*`, logi.cost, na.ratios)
            total.logi.cost <- do.call(sum, logi.cost)
          } else {
            na.ratio <- mean(pre.obj$na.loc[, pre.obj$logi])
            logi.cost <- torch_mul(logi.cost[[1]], na.ratio)
            total.logi.cost <- logi.cost
          }
        } else {
          total.logi.cost <- do.call(sum, logi.cost)
        }
      } else {
        total.logi.cost <- torch_zeros(1)
      }






      # binary
      if (length(pre.obj$bin) > 0) {
        bin.cost <- vector("list", length = length(pre.obj$bin))
        names(bin.cost) <- pre.obj$bin


        for (idx in seq_along(pre.obj$bin)) {
          var <- pre.obj$bin[idx]
          obs.idx <- which(pre.obj$na.loc[as.array(b$index), var] != TRUE)
          bin.cost[[var]] <- bin_loss(input = Out[obs.idx, pre.obj$bin.idx[[var]]], target = bin.tensor[obs.idx, idx])
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
      } else {
        total.bin.cost <- torch_zeros(1)
      }




      # multiclass
      if (length(pre.obj$multi) > 0) {
        multi.cost <- vector("list", length = length(pre.obj$multi))
        names(multi.cost) <- pre.obj$multi



        for (idx in seq_along(pre.obj$multi)) {
          var <- pre.obj$multi[idx]
          obs.idx <- which(pre.obj$na.loc[as.array(b$index), var] != TRUE)
          # which(pre.obj$na.loc[as.array(b$index), var] == TRUE)
          multi.cost[[var]] <- multi_loss(input = Out[obs.idx, pre.obj$multi.idx[[var]]], target = multi.tensor[obs.idx, idx])
        }


        if (loss.na.scale) {
          if (length(pre.obj$multi) > 1) {
            na.ratios <- colMeans(pre.obj$na.loc[, pre.obj$multi])
            # if a column is fully observed, the contribute loss is zero. ..may not be ideal
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
      } else {
        total.multi.cost <- torch_zeros(1)
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
    }


    # each epoch

    if (verbose & (epoch == 1 | epoch %% print.every.n == 0)) {
      cat(sprintf("Loss at epoch %d: %1f\n", epoch, train.loss / train.num.batches))
    }
  }

  model$eval()



  # The whole dataset
  # eval_dl <- torch::dataloader(dataset = torch.data, batch_size = n.samples, shuffle = FALSE)


  # wholebatch <- eval_dl %>%
  # torch::dataloader_make_iter() %>%
  # torch::dataloader_next()

  # imputed data

  # output.data <- model(wholebatch$data)
  # output.data <- model(data.tensor$num.tensor$to(device = device),data.tensor$logi.tensor$to(device = device),data.tensor$bin.tensor$to(device = device),data.tensor$multi.tensor$to(device = device))



  num.tensor <- move_to_device(tensor = data.tensor$num.tensor, device = device)
  logi.tensor <- move_to_device(tensor = data.tensor$logi.tensor, device = device)
  bin.tensor <- move_to_device(tensor = data.tensor$bin.tensor, device = device)
  multi.tensor <- move_to_device(tensor = data.tensor$multi.tensor, device = device)
  onehot.tensor <- move_to_device(tensor = data.tensor$onehot.tensor, device = device)

  if (categorical.encoding == "embeddings") {
    Out <- model(num.tensor = num.tensor, logi.tensor = logi.tensor, bin.tensor = bin.tensor, cat.tensor = multi.tensor)
  } else if (categorical.encoding == "onehot") {
    Out <- model(num.tensor = num.tensor, logi.tensor = logi.tensor, bin.tensor = bin.tensor, cat.tensor = onehot.tensor)
  } else {
    stop(cat('categorical.encoding can only be either "embeddings" or "onehot".\n'))
  }

  if (module == "vae") {
    Out <- Out$reconstrx
  }

  output.data <- Out$to(device = "cpu")

  imp.data <- postprocess(output.data = output.data, pre.obj = pre.obj, scaler = scaler)
  #

  save.p <- length(na.vars) + length(extra.vars)
  yhatobs.list <- vector("list", save.p)
  names(yhatobs.list) <- c(na.vars, extra.vars)


  for (var in na.vars) {
    if (var %in% pre.obj$num) {
      yhatobs.list[[var]] <- imp.data[[var]][!na.loc[, var]]
    } else if (var %in% pre.obj$bin) {
      var.idx <- pre.obj$bin.idx[[var]]
      if (pmm.link == "logit") {
        yhatobs.list[[var]] <- as_array(output.data[!na.loc[, var], var.idx])
      } else if (pmm.link == "prob") {
        transform_fn <- nn_sigmoid()
        yhatobs.list[[var]] <- as_array(transform_fn(output.data[!na.loc[, var], var.idx]))
      } else {
        stop("pmm.link has to be either `logit` or `prob`")
      }
    } else if (var %in% pre.obj$multi) {
      var.idx <- pre.obj$multi.idx[[var]]
      # probability for each class of a multiclass variable
      yhatobs.list[[var]] <- as_array(output.data[!na.loc[, var], var.idx])
    }
  }





  if (!is.null(extra.vars)) {
    for (var in extra.vars) {
      if (var %in% pre.obj$num) {
        yhatobs.list[[var]] <- imp.data[[var]]
      } else if (var %in% pre.obj$bin) {
        var.idx <- pre.obj$bin.idx[[var]]
        if (pmm.link == "logit") {
          yhatobs.list[[var]] <- as_array(output.data[, var.idx])
        } else if (pmm.link == "prob") {
          transform_fn <- nn_sigmoid()
          yhatobs.list[[var]] <- as_array(transform_fn(output.data[, var.idx]))
        } else {
          stop("pmm.link has to be either `logit` or `prob`")
        }
      } else if (var %in% pre.obj$multi) {
        var.idx <- pre.obj$multi.idx[[var]]
        # probability for each class of a multiclass variable
        yhatobs.list[[var]] <- as_array(output.data[, var.idx])
      }
    }
  }


  return(yhatobs.list)
}
