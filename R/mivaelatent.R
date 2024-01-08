#' multiple imputation through variational autoencoders (latent) haven't change default setting yet
#' @param data A data frame, tibble or data table with missing values.
#' @param m The number of imputed datasets.
#' @param categorical.encoding The method for representing multi-class categorical features. Can be either "embeddings" or "onehot" (default).
#' @param device Device to use. Either "cpu" (default) or "cuda" for GPU.
#' @param epochs The number of training epochs (iterations). Default: 100.
#' @param batch.size The size of samples in each batch. Default: 512.
#' @param subsample The subsample ratio of training data. Default: 1.
#' @param early.stopping.epochs An integer value \code{k}. Mivae training will stop if the validation performance has not improved for \code{k} epochs, only used when \code{subsample}<1. Default: 1.
#' @param vae.params A list of parameters for variational autoencoders. See \code{\link[=vae_default]{vae_default}} for details.
#' @param pmm.params A list of parameters for predictive mean matching. See \code{\link[=vae_pmm_default]{vae_pmm_default}} for details.
#' @param loss.na.scale Whether to multiply the ratio of missing values in  a feature to calculate the loss function. Default: FALSE.
#' @param verbose Whether or not to print training loss information. Default: TRUE.
#' @param print.every.n If verbose is set to TRUE, print out training loss for every n epochs. Default: 1.
#' @param save.model Whether or not to save the imputation model. Default: FALSE.
#' @param path The path where the final imputation model will be saved.
mivae.latent <- function(data, m = 5, categorical.encoding = "embeddings", device = "cpu",
                  epochs = 5, batch.size = 32,
                  subsample = 1,
                  early.stopping.epochs = 1,
                  vae.params=list(),
                  pmm.params=list(),
                  loss.na.scale = FALSE,
                  verbose = TRUE, print.every.n = 1,
                  save.model = FALSE, path = NULL) {

  device <- torch_device(device)

  vae.params <- do.call("vae_default", vae.params)
  pmm.params <- do.call("vae_pmm_default", pmm.params)


  shuffle <- vae.params$shuffle
  drop.last<- vae.params$drop.last
  beta <- vae.params$beta
  input.dropout <- vae.params$input.dropout
  hidden.dropout <- vae.params$hidden.dropout

  optimizer <- vae.params$optimizer
  learning.rate <- vae.params$learning.rate
  weight.decay <- vae.params$weight.decay
  momentum <- vae.params$momentum
  eps <- vae.params$eps
  dampening <- vae.params$dampening
  rho <- vae.params$rho
  alpha <- vae.params$alpha
  learning.rate.decay <- vae.params$learning.rate.decay


  encoder.structure <- vae.params$encoder.structure
  latent.dim <- vae.params$latent.dim
  decoder.structure<- vae.params$decoder.structure
  act <- vae.params$act
  init.weight <- vae.params$init.weight
  scaler <- vae.params$scaler
  lower<-vae.params$lower
  upper<-vae.params$upper
  initial.imp<-vae.params$initial.imp



  pmm.type <- pmm.params$pmm.type
  pmm.k <- pmm.params$pmm.k
  pmm.link <- pmm.params$pmm.link
  pmm.save.vars <- pmm.params$pmm.save.vars


  if (subsample == 1 & early.stopping.epochs > 1) {
    stop("To use early stopping based on validation error, please set subsample < 1.")
  }


  # save.model & is.null(path)
  if (is.null(path)) {
    stop("Please specify a path to save the imputation model.")
  }

  # check pmm.save.vars #included in colnames of data
  origin.names <- colnames(data)

  if (!all(pmm.save.vars %in% origin.names)) {
    stop("Some variables specified in `pmm.save.vars` do not exist in the dataset. Please check again.")
  }


  pre.obj <- preprocess(data, scaler = scaler, lower=lower,upper=upper, categorical.encoding = categorical.encoding, initial.imp = initial.imp)


  cardinalities<-pre.obj$cardinalities
  embedding.dim<-pre.obj$embedding.dim

  #n.num+n.logi+n.bin
  n.others <- length(origin.names)-length(cardinalities)


  data.tensor<-torch_dataset(data, scaler = scaler, lower=lower,upper=upper, categorical.encoding = categorical.encoding, initial.imp = initial.imp)

  n.samples <- nrow(data)



  # check pmm
  sort.result <- sortNA(data)
  sorted.dt <- sort.result$sorted.dt
  sorted.types <- feature_type(sorted.dt)
  sorted.naSums <- colSums(is.na(sorted.dt))
  check_pmm(pmm.type = pmm.type, subsample = subsample, input.dropout = input.dropout, hidden.dropout = hidden.dropout, Nrow = n.samples, sorted.naSums, sorted.types, pmm.k)


  # imputed data
  imputed.data <- vector("list", length = m)
  imputed.mu <- vector("list", length = m)
  imputed.log.var <- vector("list", length = m)

  na.loc <- pre.obj$na.loc
  na.vars <- pre.obj$ordered.names[colSums(na.loc) != 0]


  if (is.null(pmm.save.vars) | setequal(pmm.save.vars, na.vars)) {
    extra.vars <- NULL
  } else {
    # check for other cases
    if (!all(na.vars %in% pmm.save.vars)) {
      # not all pmm.save.vars is in the na.vars
      stop("Some variables has missing values in the training data, but 'pmm.save.vars' does not contains all of them. Please re-specify `save.vars`.")
    } else {
      # pmm.save.vars contains all na.vars with some other variables
      extra.vars <- setdiff(pmm.save.vars, na.vars)
    }
  }


  # yobs.list

  if (is.null(pmm.type)) {
    yobs.list <- NULL
  } else {
    save.p <- length(na.vars) + length(extra.vars)
    yobs.list <- vector("list", save.p)
    names(yobs.list) <- c(na.vars, extra.vars)

    for (var in na.vars) {
      yobs.list[[var]] <- data[[var]][!na.loc[, var]]
    }

    if (!is.null(extra.vars)) {
      for (var in extra.vars) {
        yobs.list[[var]] <- data[[var]]
      }
    }
  }



  # yhatobs.list #need to amend yhatobs_pmm1 to includes vae
  if (isTRUE(pmm.type == 1)) {
    yhatobs.list <- yhatobs_pmm1(module="vae",
                                 data = data, categorical.encoding = categorical.encoding, device = device, na.loc = na.loc, na.vars = na.vars, extra.vars = extra.vars, pmm.link = pmm.link,
                                 epochs = epochs, batch.size = batch.size, drop.last = drop.last, shuffle = shuffle,
                                 optimizer = optimizer, learning.rate = learning.rate, weight.decay = weight.decay, momentum = momentum, eps = eps,
                                 encoder.structure = encoder.structure, latent.dim = latent.dim, decoder.structure = decoder.structure,
                                 act = act, init.weight = init.weight, scaler = scaler,initial.imp = initial.imp,lower=lower, upper=upper,
                                 loss.na.scale = loss.na.scale,
                                 verbose = verbose, print.every.n = print.every.n
    )
  } else if (is.null(pmm.type)) {
    yhatobs.list <- NULL
  } else {
    yhatobs.list <- vector("list", m)
    yhatobs.list.each <- vector("list", save.p)
    names(yhatobs.list.each) <- c(na.vars, extra.vars)
    for (i in seq_along(yhatobs.list)) {
      yhatobs.list[[i]] <- yhatobs.list.each
    }
  }


  if (subsample == 1) {

    train.samples <- n.samples
    train.idx <- 1:n.samples
    train.batches<-batch_set(n.samples = train.samples, batch.size = batch.size, drop.last = drop.last)
    train.batch.set<-train.batches$batch.set
    train.num.batches<-train.batches$num.batches
    train.original.data<-data.tensor




  } else {

    train.idx <- sample(1:n.samples, size = floor(subsample * n.samples), replace = FALSE)
    valid.idx <- setdiff(1:n.samples, train.idx)

    train.samples <- length(train.idx)
    valid.samples <- length(valid.idx)



    train.original.data<-torch_dataset_idx(data, idx=train.idx, scaler = scaler, lower=lower,upper=upper, categorical.encoding = categorical.encoding, initial.imp = initial.imp)
    valid.original.data<-torch_dataset_idx(data, idx=valid.idx, scaler = scaler, lower=lower,upper=upper, categorical.encoding = categorical.encoding, initial.imp = initial.imp)

    train.batches<-batch_set(n.samples = train.samples, batch.size = batch.size, drop.last = drop.last)
    train.batch.set<-train.batches$batch.set
    train.num.batches<-train.batches$num.batches

    valid.batches<-batch_set(n.samples = valid.samples, batch.size = batch.size, drop.last = drop.last)
    valid.batch.set<-valid.batches$batch.set
    valid.num.batches<-valid.batches$num.batches
  }


  # mivae model -------------------------------------------------------------


  model <- vae(categorical.encoding = categorical.encoding, n.others = n.others, cardinalities = cardinalities, embedding.dim = embedding.dim,
               input.dropout = input.dropout, hidden.dropout = hidden.dropout, encoder.structure = encoder.structure, latent.dim = latent.dim, decoder.structure = decoder.structure, act = act)

  model <-model$to(device=device)


  if (init.weight == "he.normal") {
    model$apply(init_he_normal)
  }else if (init.weight == "he.uniform") {
    model$apply(init_he_uniform)
  }else if (init.weight == "he.normal.elu") {
    model$apply(init_he_normal_elu)
  }else if (init.weight == "he.normal.selu") {
    model$apply(init_he_normal_selu)
  }else if (init.weight == "he.normal.leaky.relu") {
    model$apply(init_he_normal_leaky.relu)
  }else if (init.weight == "xavier.normal") {
    model$apply(init_xavier_normal)
  } else if (init.weight == "xavier.uniform") {
    model$apply(init_xavier_uniform)
  } else if (init.weight == "xavier.midas") {
    model$apply(init_xavier_midas)
  }else{
    stop("This weight initialization is not supported yet")
  }





  # mivae loss -------------------------------------------------------------------


  # define the loss function
  num_loss <- nn_mse_loss(reduction = "mean")
  logi_loss <- nn_bce_with_logits_loss(reduction = "mean")
  bin_loss <- nn_bce_with_logits_loss(reduction = "mean")
  multi_loss <- nn_cross_entropy_loss(reduction = "mean")




  # choose optimizer & learning rate
  if (optimizer == "adamW") {
    optimizer <- optim_adamw(model$parameters, lr = learning.rate, eps = eps, weight_decay = weight.decay)
  } else if (optimizer == "sgd") {
    optimizer <- optim_sgd(model$parameters, lr = learning.rate, momentum = momentum, dampening = dampening, weight_decay = weight.decay)
  } else if (optimizer == "adam") {# torch default eps = 1e-08, tensorfolow default eps =1e-07
    optimizer <- optim_adam(model$parameters, lr = learning.rate, eps = eps, weight_decay = weight.decay)
  } else if (optimizer == "adadelta") {
    optimizer <- optim_adadelta(model$parameters, lr = learning.rate, rho = rho, eps = eps, weight_decay = weight.decay)
  } else if (optimizer == "adagrad") {
    optimizer <- optim_adagrad(model$parameters, lr = learning.rate, lr_decay = learning.rate.decay, eps = eps, weight_decay = weight.decay)
  } else if (optimizer == "rmsprop") {
    optimizer <- optim_rmsprop(model$parameters, lr = learning.rate, alpha = alpha, eps = eps, weight_decay = weight.decay, momentum = momentum)
  }


  # epochs: number of iterations
  if(verbose){
    print("Running mivae().")
  }


  # epochs: number of iterations
  best.loss <- Inf
  num.nondecresing.epochs <- 0

  for (epoch in seq_len(epochs)) {
    model$train()

    train.loss <- 0


    #rearrange all the data in each epoch
    if(shuffle){
      permute<-torch_randperm(train.samples)+1L

    }else{
      permute<-torch_tensor(1:train.samples)
    }

    train.data<-train.original.data[permute]

    for(i in 1:train.num.batches){

      b<-list()
      b.index<-train.batch.set[[i]]
      b$data<-lapply(train.data, function(x) x[b.index])
      b$index<-train.idx[as.array(permute)[b.index]]

      num.tensor<-move_to_device(tensor=b$data$num.tensor, device=device)
      logi.tensor<-move_to_device(tensor=b$data$logi.tensor, device=device)
      bin.tensor<-move_to_device(tensor=b$data$bin.tensor, device=device)
      multi.tensor<-move_to_device(tensor=b$data$multi.tensor, device=device)
      onehot.tensor<-move_to_device(tensor=b$data$onehot.tensor, device=device)



      if(categorical.encoding=="embeddings"){
        Out <- model(num.tensor=num.tensor,logi.tensor=logi.tensor,bin.tensor=bin.tensor, cat.tensor=multi.tensor)
      }else if(categorical.encoding=="onehot"){
        Out <- model(num.tensor=num.tensor,logi.tensor=logi.tensor,bin.tensor=bin.tensor, cat.tensor=onehot.tensor)
      }else{
        stop(cat('categorical.encoding can only be either "embeddings" or "onehot".\n'))
      }


      if (length(pre.obj$num) > 0) {
        num.cost <- vector("list", length = length(pre.obj$num))
        names(num.cost) <- pre.obj$num

        for (idx in seq_along(pre.obj$num.idx)) {
          var<-pre.obj$num[idx]
          obs.idx <- which(pre.obj$na.loc[as.array(b$index), var] != TRUE)
          num.cost[[var]] <- num_loss(input = Out$reconstrx[obs.idx, pre.obj$num.idx[[var]]], target = num.tensor[obs.idx, idx])
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
          var<-pre.obj$logi[idx]
          obs.idx <- which(pre.obj$na.loc[as.array(b$index), var] != TRUE)
          logi.cost[[var]] <- logi_loss(input =  Out$reconstrx[obs.idx, pre.obj$logi.idx[[var]]], target = logi.tensor[obs.idx, idx])
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
          var<-pre.obj$bin[idx]
          obs.idx <- which(pre.obj$na.loc[as.array(b$index), var] != TRUE)
          bin.cost[[var]] <- bin_loss(input = Out$reconstrx[obs.idx, pre.obj$bin.idx[[var]]], target = bin.tensor[obs.idx, idx])
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
          var<-pre.obj$multi[idx]
          obs.idx <- which(pre.obj$na.loc[as.array(b$index), var] != TRUE)
          multi.cost[[var]] <- multi_loss(input = Out$reconstrx[obs.idx, pre.obj$multi.idx[[var]]], target = multi.tensor[obs.idx, idx])
        }


        if (loss.na.scale) {
          if (length(pre.obj$multi) > 1) {
            na.ratios <- colMeans(pre.obj$na.loc[, pre.obj$multi])
            #if a column is fully observed, the contribute loss is zero. ..may not be ideal
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


      # Total cost (reconstruction loss)
      cost <- sum(total.num.cost, total.bin.cost, total.multi.cost)


      # KL ----------------------------------------------------------------------
      mu <- Out$mu
      log.var <- Out$log.var

      # KL divergence
      # the mean of a batch for the sum of kld the latent dimensions
      kl.divergence <- torch_mean(-0.5 * torch_sum(1 + log.var - mu^2 - log.var$exp(), dim = 2))

      total.cost <- cost + beta * kl.divergence




      # zero out the gradients
      optimizer$zero_grad()
      total.cost$backward()
      # update params
      optimizer$step()


      batch.loss <- total.cost$item()
      train.loss <- train.loss + batch.loss

    }


    ### if subsample<1, show validation error

    if (subsample < 1) {
      model$eval()

      valid.loss <- 0

      #rearrange all the data in each epoch
      if(shuffle){
        permute<-torch_randperm(valid.samples)+1L
      }else{
        permute<-torch_tensor(1:valid.samples)
      }


      valid.data<-valid.original.data[permute]

      # validation loss
      for(i in 1:valid.num.batches){
        b<-list()
        b.index<-valid.batch.set[[i]]

        b$data<-lapply(valid.data, function(x) x[b.index])

        #index in the original full dataset
        b$index<-valid.idx[as.array(permute)[b.index]]



        num.tensor<-move_to_device(tensor=b$data$num.tensor, device=device)
        logi.tensor<-move_to_device(tensor=b$data$logi.tensor, device=device)
        bin.tensor<-move_to_device(tensor=b$data$bin.tensor, device=device)
        multi.tensor<-move_to_device(tensor=b$data$multi.tensor, device=device)
        onehot.tensor<-move_to_device(tensor=b$data$onehot.tensor, device=device)

        if(categorical.encoding=="embeddings"){
          Out <- model(num.tensor=num.tensor,logi.tensor=logi.tensor,bin.tensor=bin.tensor, cat.tensor=multi.tensor)
        }else if(categorical.encoding=="onehot"){
          Out <- model(num.tensor=num.tensor,logi.tensor=logi.tensor,bin.tensor=bin.tensor, cat.tensor=onehot.tensor)
        }else{
          stop(cat('categorical.encoding can only be either "embeddings" or "onehot".\n'))
        }


        # numeric
        if (length(pre.obj$num) > 0) {
          num.cost <- vector("list", length = length(pre.obj$num))
          names(num.cost) <- pre.obj$num

          for (idx in seq_along(pre.obj$num.idx)) {
            var<-pre.obj$num[idx]
            obs.idx <- which(pre.obj$na.loc[as.array(b$index), var] != TRUE)
            num.cost[[var]] <- num_loss(input = Out$reconstrx[obs.idx, pre.obj$num.idx[[var]]], target = num.tensor[obs.idx, idx])
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
            var<-pre.obj$logi[idx]
            obs.idx <- which(pre.obj$na.loc[as.array(b$index), var] != TRUE)
            logi.cost[[var]] <- logi_loss(input = Out$reconstrx[obs.idx, pre.obj$logi.idx[[var]]], target = logi.tensor[obs.idx, idx])
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
            var<-pre.obj$bin[idx]
            obs.idx <- which(pre.obj$na.loc[as.array(b$index), var] != TRUE)
            bin.cost[[var]] <- bin_loss(input = Out$reconstrx[obs.idx, pre.obj$bin.idx[[var]]], target = bin.tensor[obs.idx, idx])
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
            var<-pre.obj$multi[idx]
            obs.idx <- which(pre.obj$na.loc[as.array(b$index), var] != TRUE)
            multi.cost[[var]] <- multi_loss(input = Out$reconstrx[obs.idx, pre.obj$multi.idx[[var]]], target = multi.tensor[obs.idx, idx])
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
        } else {
          total.multi.cost <- torch_zeros(1)
        }




        # Total cost
        cost <- sum(total.num.cost, total.bin.cost, total.multi.cost)

        # KL divergence
        mu <- Out$mu
        log.var <- Out$log.var
        # the mean of a batch for the sum of kld the latent dimensions
        kl.divergence <- torch_mean(-0.5 * torch_sum(1 + log.var - mu^2 - log.var$exp(), dim = 2))

        total.cost <- cost + beta * kl.divergence




        # zero out the gradients
        optimizer$zero_grad()
        total.cost$backward()
        # update params
        optimizer$step()


        batch.loss <- total.cost$item()
        valid.loss <- valid.loss + batch.loss

      }
    }



    # each epoch
    if (subsample == 1) {
      if (verbose & (epoch == 1 | epoch %% print.every.n == 0)) {
        cat(sprintf("Loss at epoch %d: %1f\n", epoch, train.loss / train.num.batches))
      }
      if (save.model & epoch == epochs) {
        torch_save(model, path = path)
      }
    } else if (subsample < 1) {
      valid.epoch.loss <- valid.loss / valid.num.batches
      if (verbose & (epoch == 1 | epoch %% print.every.n == 0)) {
        cat(sprintf("Loss at epoch %d: training: %3f, validation: %3f\n", epoch, train.loss / train.num.batches, valid.epoch.loss))
      }
      if (early.stopping.epochs > 1) {
        if (valid.epoch.loss < best.loss) {
          best.loss <- valid.epoch.loss
          best.epoch <- epoch
          num.nondecresing.epochs <- 0
          torch_save(model, path = path)
        } else {
          num.nondecresing.epochs <- num.nondecresing.epochs + 1
          if (num.nondecresing.epochs >= early.stopping.epochs) {
            cat(sprintf("Best loss at epoch %d: %1f\n", best.epoch, best.loss))
            break
          }
        }
      }
    }
  }

  # model <- torch::torch_load(path = path)
  if (subsample < 1 & early.stopping.epochs > 1) {
    model <- torch_load(path = path)
  }

  model<-model$to(device=device)

  model$eval()




  for (i in seq_len(m)) {


    num.tensor<-move_to_device(tensor=data.tensor$num.tensor, device=device)
    logi.tensor<-move_to_device(tensor=data.tensor$logi.tensor, device=device)
    bin.tensor<-move_to_device(tensor=data.tensor$bin.tensor, device=device)
    multi.tensor<-move_to_device(tensor=data.tensor$multi.tensor, device=device)
    onehot.tensor<-move_to_device(tensor=data.tensor$onehot.tensor, device=device)

    if(categorical.encoding=="embeddings"){
      Out <- model(num.tensor=num.tensor,logi.tensor=logi.tensor,bin.tensor=bin.tensor, cat.tensor=multi.tensor)
    }else if(categorical.encoding=="onehot"){
      Out <- model(num.tensor=num.tensor,logi.tensor=logi.tensor,bin.tensor=bin.tensor, cat.tensor=onehot.tensor)
    }else{
      stop(cat('categorical.encoding can only be either "embeddings" or "onehot".\n'))
    }

    output.data<-Out$reconstrx$to(device = "cpu")

    #additional
    output.mu<-Out$mu$to(device = "cpu")
    output.log.var<-Out$log.var$to(device = "cpu")

    imp.data <- postprocess(output.data = output.data, pre.obj = pre.obj, scaler = scaler)



    if (isFALSE(save.model)) {
      # don't need to save pmm values
      if (is.null(pmm.type)) {
        for (var in na.vars) {
          data[[var]][na.loc[, var]] <- imp.data[[var]][na.loc[, var]]
        }
      } else if (pmm.type == 0 | pmm.type == 2) {
        for (var in na.vars) {
          if (var %in% pre.obj$num) {
            # numeric or binary? check binary
            yhatobs <- imp.data[[var]][!na.loc[, var]]
            yhatmis <- imp.data[[var]][na.loc[, var]]
            data[[var]][na.loc[, var]] <- pmm(yhatobs = yhatobs, yhatmis = yhatmis, yobs = yobs.list[[var]], k = pmm.k)
          } else if (var %in% pre.obj$logi) {
            # binary
            var.idx <- pre.obj$logi.idx[[var]]

            if (pmm.link == "logit") {
              yhatobs <- as_array(output.data[!na.loc[, var], var.idx])
              yhatmis <- as_array(output.data[na.loc[, var], var.idx])
            } else if (pmm.link == "prob") {
              transform_fn <- nn_sigmoid()
              yhatobs <- as_array(transform_fn(output.data[!na.loc[, var], var.idx]))
              yhatmis <- as_array(transform_fn(output.data[na.loc[, var], var.idx]))
            } else {
              stop("pmm.link has to be either `logit` or `prob`")
            }

            data[[var]][na.loc[, var]]<-pmm(yhatobs = yhatobs, yhatmis = yhatmis, yobs = yobs.list[[var]], k = pmm.k)

          } else if (var %in% pre.obj$bin) {
            # binary
            var.idx <- pre.obj$bin.idx[[var]]

            if (pmm.link == "logit") {
              yhatobs <- as_array(output.data[!na.loc[, var], var.idx])
              yhatmis <- as_array(output.data[na.loc[, var], var.idx])
            } else if (pmm.link == "prob") {
              transform_fn <- nn_sigmoid()
              yhatobs <- as_array(transform_fn(output.data[!na.loc[, var], var.idx]))
              yhatmis <- as_array(transform_fn(output.data[na.loc[, var], var.idx]))
            } else {
              stop("pmm.link has to be either `logit` or `prob`")
            }

            data[[var]][na.loc[, var]]<-pmm(yhatobs = yhatobs, yhatmis = yhatmis, yobs = yobs.list[[var]], k = pmm.k)

            #level.idx <- pmm.multiclass(yhatobs = yhatobs, yhatmis = yhatmis, yobs = yobs.list[[var]], k = pmm.k)
            #data[[var]][na.loc[, var]] <- levels(data[[var]])[level.idx]
          } else if (var %in% pre.obj$multi) {
            # multiclass
            var.idx <- pre.obj$multi.idx[[var]]

            # probability for each class of a multiclass variable
            yhatobs <- as_array(output.data[!na.loc[, var], var.idx])
            yhatmis <- as_array(output.data[na.loc[, var], var.idx])

            level.idx <- pmm.multiclass(yhatobs = yhatobs, yhatmis = yhatmis, yobs = yobs.list[[var]], k = pmm.k)
            data[[var]][na.loc[, var]] <- levels(data[[var]])[level.idx]
          }
        }
      } else if (pmm.type == "auto") {
        for (var in na.vars) {
          yhatobs <- imp.data[[var]][!na.loc[, var]]
          yhatmis <- imp.data[[var]][na.loc[, var]]

          if (var %in% pre.obj$num) {
            # numeric
            data[[var]][na.loc[, var]] <- pmm(yhatobs = yhatobs, yhatmis = yhatmis, yobs = yobs.list[[var]], k = pmm.k)
          } else {
            # binary or multiclass: no pmm
            data[[var]][na.loc[, var]] <- yhatmis
          }
        }
      } else if (pmm.type == 1) {
        for (var in na.vars) {
          yhatobs <- yhatobs.list[[var]]
          if (var %in% pre.obj$num) {
            yhatmis <- imp.data[[var]][na.loc[, var]]
            data[[var]][na.loc[, var]] <- pmm(yhatobs = yhatobs, yhatmis = yhatmis, yobs = yobs.list[[var]], k = pmm.k)
          }else if (var %in% pre.obj$logi) {
            # binary
            var.idx <- pre.obj$logi.idx[[var]]
            if (pmm.link == "logit") {
              yhatmis <- as_array(output.data[na.loc[, var], var.idx])
            } else if (pmm.link == "prob") {
              transform_fn <- nn_sigmoid()
              yhatmis <- as_array(transform_fn(output.data[na.loc[, var], var.idx]))
            } else {
              stop("pmm.link has to be either `logit` or `prob`")
            }

            data[[var]][na.loc[, var]] <-pmm(yhatobs = yhatobs, yhatmis = yhatmis, yobs = yobs.list[[var]], k = pmm.k)

          } else if (var %in% pre.obj$bin) {
            # binary
            var.idx <- pre.obj$bin.idx[[var]]
            if (pmm.link == "logit") {
              yhatmis <- as_array(output.data[na.loc[, var], var.idx])
            } else if (pmm.link == "prob") {
              transform_fn <- nn_sigmoid()
              yhatmis <- as_array(transform_fn(output.data[na.loc[, var], var.idx]))
            } else {
              stop("pmm.link has to be either `logit` or `prob`")
            }

            data[[var]][na.loc[, var]] <-pmm(yhatobs = yhatobs, yhatmis = yhatmis, yobs = yobs.list[[var]], k = pmm.k)

            #level.idx <- pmm.multiclass(yhatobs = yhatobs, yhatmis = yhatmis, yobs = yobs.list[[var]], k = pmm.k)
            #data[[var]][na.loc[, var]] <- levels(data[[var]])[level.idx]

          } else if (var %in% pre.obj$multi) {
            # multiclass
            var.idx <- pre.obj$multi.idx[[var]]
            yhatmis <- as_array(output.data[na.loc[, var], var.idx])
            level.idx <- pmm.multiclass(yhatobs = yhatobs, yhatmis = yhatmis, yobs = yobs.list[[var]], k = pmm.k)
            data[[var]][na.loc[, var]] <- levels(data[[var]])[level.idx]
          }
        }
      }
    } else {
      # need to save pmm values
      if (is.null(pmm.type)) {
        for (var in na.vars) {
          data[[var]][na.loc[, var]] <- imp.data[[var]][na.loc[, var]]
        }
      } else if (pmm.type == 0 | pmm.type == 2) {
        for (var in na.vars) {
          if (var %in% pre.obj$num) {
            # numeric or binary? check binary
            yhatobs <- imp.data[[var]][!na.loc[, var]]
            yhatmis <- imp.data[[var]][na.loc[, var]]
            data[[var]][na.loc[, var]] <- pmm(yhatobs = yhatobs, yhatmis = yhatmis, yobs = yobs.list[[var]], k = pmm.k)
          } else if (var %in% pre.obj$logi) {
            # binary
            var.idx <- pre.obj$logi.idx[[var]]

            if (pmm.link == "logit") {
              yhatobs <- as_array(output.data[!na.loc[, var], var.idx])
              yhatmis <- as_array(output.data[na.loc[, var], var.idx])
            } else if (pmm.link == "prob") {
              transform_fn <- nn_sigmoid()
              yhatobs <- as_array(transform_fn(output.data[!na.loc[, var], var.idx]))
              yhatmis <- as_array(transform_fn(output.data[na.loc[, var], var.idx]))
            } else {
              stop("pmm.link has to be either `logit` or `prob`")
            }

            data[[var]][na.loc[, var]] <- pmm(yhatobs = yhatobs, yhatmis = yhatmis, yobs = yobs.list[[var]], k = pmm.k)
          } else if (var %in% pre.obj$bin) {
            # binary
            var.idx <- pre.obj$bin.idx[[var]]

            if (pmm.link == "logit") {
              yhatobs <- as_array(output.data[!na.loc[, var], var.idx])
              yhatmis <- as_array(output.data[na.loc[, var], var.idx])
            } else if (pmm.link == "prob") {
              transform_fn <- nn_sigmoid()
              yhatobs <- as_array(transform_fn(output.data[!na.loc[, var], var.idx]))
              yhatmis <- as_array(transform_fn(output.data[na.loc[, var], var.idx]))
            } else {
              stop("pmm.link has to be either `logit` or `prob`")
            }

            data[[var]][na.loc[, var]] <- pmm(yhatobs = yhatobs, yhatmis = yhatmis, yobs = yobs.list[[var]], k = pmm.k)
          } else if (var %in% pre.obj$multi) {
            # multiclass
            var.idx <- pre.obj$multi.idx[[var]]

            # probability for each class of a multiclass variable
            yhatobs <- as_array(output.data[!na.loc[, var], var.idx])
            yhatmis <- as_array(output.data[na.loc[, var], var.idx])

            level.idx <- pmm.multiclass(yhatobs = yhatobs, yhatmis = yhatmis, yobs = yobs.list[[var]], k = pmm.k)
            data[[var]][na.loc[, var]] <- levels(data[[var]])[level.idx]
          }
          # save yhatobs
          yhatobs.list[[i]][[var]] <- yhatobs
        }
      } else if (pmm.type == "auto") {
        for (var in na.vars) {
          yhatobs <- imp.data[[var]][!na.loc[, var]]
          yhatmis <- imp.data[[var]][na.loc[, var]]

          if (var %in% pre.obj$num) {
            # numeric
            data[[var]][na.loc[, var]] <- pmm(yhatobs = yhatobs, yhatmis = yhatmis, yobs = yobs.list[[var]], k = pmm.k)
            # save yhatobs
            yhatobs.list[[i]][[var]] <- yhatobs
          } else {
            # binary or multiclass: no pmm
            data[[var]][na.loc[, var]] <- yhatmis
          }
        }
      } else if (pmm.type == 1) {
        for (var in na.vars) {
          yhatobs <- yhatobs.list[[var]]
          if (var %in% pre.obj$num) {
            yhatmis <- imp.data[[var]][na.loc[, var]]
            data[[var]][na.loc[, var]] <- pmm(yhatobs = yhatobs, yhatmis = yhatmis, yobs = yobs.list[[var]], k = pmm.k)
          } else if (var %in% pre.obj$bin) {
            # binary
            var.idx <- pre.obj$bin.idx[[var]]
            if (pmm.link == "logit") {
              yhatmis <- as_array(output.data[na.loc[, var], var.idx])
            } else if (pmm.link == "prob") {
              transform_fn <- nn_sigmoid()
              yhatmis <- as_array(transform_fn(output.data[na.loc[, var], var.idx]))
            } else {
              stop("pmm.link has to be either `logit` or `prob`")
            }

            data[[var]][na.loc[, var]] <-pmm(yhatobs = yhatobs, yhatmis = yhatmis, yobs = yobs.list[[var]], k = pmm.k)

          } else if (var %in% pre.obj$multi) {
            # multiclass
            var.idx <- pre.obj$multi.idx[[var]]
            yhatmis <- as_array(output.data[na.loc[, var], var.idx])
            level.idx <- pmm.multiclass(yhatobs = yhatobs, yhatmis = yhatmis, yobs = yobs.list[[var]], k = pmm.k)
            data[[var]][na.loc[, var]] <- levels(data[[var]])[level.idx]
          }
        }
      }


      # save extra.vars in yhatobs.list
      if (!is.null(extra.vars)) {
        if (pmm.type == 0 | pmm.type == 2) {
          for (var in extra.vars) {
            if (var %in% pre.obj$num) {
              yhatobs.list[[i]][[var]] <- imp.data[[var]]
            }else if (var %in% c(pre.obj$logi)) {
              # binary
              var.idx <- pre.obj$logi.idx[[var]]

              if (pmm.link == "logit") {
                yhatobs.list[[i]][[var]] <- as_array(output.data[, var.idx])
              } else if (pmm.link == "prob") {
                transform_fn <- nn_sigmoid()
                yhatobs.list[[i]][[var]] <- as_array(transform_fn(output.data[, var.idx]))
              } else {
                stop("pmm.link has to be either `logit` or `prob`")
              }
            } else if (var %in% c(pre.obj$bin)) {
              # binary
              var.idx <- pre.obj$bin.idx[[var]]

              if (pmm.link == "logit") {
                yhatobs.list[[i]][[var]] <- as_array(output.data[, var.idx])
              } else if (pmm.link == "prob") {
                transform_fn <- nn_sigmoid()
                yhatobs.list[[i]][[var]] <- as_array(transform_fn(output.data[, var.idx]))
              } else {
                stop("pmm.link has to be either `logit` or `prob`")
              }
            } else if (var %in% pre.obj$multi) {
              # multiclass
              var.idx <- pre.obj$multi.idx[[var]]
              # probability for each class of a multiclass variable
              yhatobs.list[[i]][[var]] <- as_array(output.data[, var.idx])
            }
          }
        } else if (pmm.type == "auto") {
          for (var in extra.vars) {
            if (var %in% pre.obj$num) {
              # numeric
              yhatobs.list[[i]][[var]] <- imp.data[[var]]
            } else {
              # binary or multiclass: no pmm
              yhatobs.list[[i]][[var]] <- NULL
            }
          }
        }
      }
    } # save.model=TRUE

    imputed.data[[i]] <- data

    #additional
    imputed.mu[[i]]<-output.mu
    imputed.log.var[[i]]<-output.log.var
  }






  if (isFALSE(save.model)) {
    mivae.obj<-list("imputed.data"=imputed.data,"imputed.mu"=imputed.mu,"imputed.log.var"=imputed.log.var)
    return(mivae.obj)
  } else {
    params <- list()
    params$scaler <- scaler
    params$na.vars <- na.vars
    params$extra.vars <- extra.vars

    params$yhatobs.list <- yhatobs.list
    params$yobs.list <- yobs.list
    params$m <- m
    params$pmm.k <- pmm.k
    params$pmm.type <- pmm.type
    params$pmm.link <- pmm.link

    params$categorical.encoding<-categorical.encoding





    mivae.obj <- list("imputed.data" = imputed.data, "imputed.mu"=imputed.mu,"imputed.log.var"=imputed.log.var, "model.path" = path, "params" = params)
    print(paste("The VAE multiple imputation model is saved in ", path))
    class(mivae.obj) <- "mivaeObj"
    return(mivae.obj)
  }
}

