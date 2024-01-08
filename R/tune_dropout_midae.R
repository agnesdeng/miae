#' Tune dropout rate for midae
#' @param data A dataset on which the midae model will be trained.
#' @param dropout.grid A list containing two vectors: `input.dropout` and
#'   `hidden.dropout`, each specifying the dropout rates to be tested for
#'   the input layer and hidden layers, respectively.
#' @param m The number of imputations to perform.
#' @param categorical.encoding The method used for encoding categorical
#'   variables. Defaults to "embeddings".
#' @param device The computing device to use, either "cpu" or "cuda" for GPU.
#' @param epochs The number of training epochs for each model.
#' @param batch.size The size of the batches used in training.
#' @param subsample The proportion of the data to be used in training.
#'   Defaults to 1, meaning the full dataset is used.
#' @param early.stopping.epochs The number of epochs with no improvement
#'   after which training will be stopped.
#' @param pmm.params A list of parameters for predictive mean matching.
#' @param dae.params A list of parameters for the denoising autoencoder.
#' @param loss.na.scale Boolean flag indicating whether to scale the loss
#'   function based on NA values. Defaults to FALSE.
#' @param verbose Boolean flag to control the verbosity of the function's output.
#' @param print.every.n Specifies how often (in epochs) to print the training
#'   progress. Only relevant if verbose is TRUE.
#' @param save.model Boolean flag indicating whether to save the trained model.
#' @param path File path where the model should be saved if save.model is TRUE.
#'   If NULL and save.model is TRUE, the model is saved in the current directory.
#' @return A list containing the tuned parameters and their corresponding
#'   performance metrics.
#' @examples
#' 1+1
#' @export
tune_dropout_midae<- function(data, dropout.grid = list(input.dropout=c(0,0.25,0.5),hidden.dropout=c(0, 0.25, 0.5)),
                                  m = 5, categorical.encoding = "embeddings", device = "cpu",
                                  epochs = 5, batch.size = 32,
                                  subsample = 1,
                                  early.stopping.epochs = 1,
                                  pmm.params=list(),
                                  dae.params=list(),
                                  loss.na.scale = FALSE,
                                  verbose = TRUE, print.every.n = 1,
                                  save.model = FALSE, path = NULL) {


  device <- torch_device(device)

  dae.params <- do.call("dae_default", dae.params)
  pmm.params <- do.call("dae_pmm_default", pmm.params)


  shuffle <- dae.params$shuffle
  drop.last<- dae.params$drop.last
  input.dropout <- dae.params$input.dropout
  hidden.dropout <- dae.params$hidden.dropout

  #optimizer <- dae.params$optimizer
  optim.name <- dae.params$optimizer
  learning.rate <- dae.params$learning.rate
  weight.decay <- dae.params$weight.decay
  momentum <- dae.params$momentum
  eps <- dae.params$eps
  dampening <- dae.params$dampening
  rho <- dae.params$rho
  alpha <- dae.params$alpha
  learning.rate.decay <- dae.params$learning.rate.decay


  encoder.structure <- dae.params$encoder.structure
  latent.dim <- dae.params$latent.dim
  decoder.structure<- dae.params$decoder.structure
  act <- dae.params$act
  init.weight <- dae.params$init.weight
  scaler <- dae.params$scaler
  lower<-dae.params$lower
  upper<-dae.params$upper
  initial.imp<-dae.params$initial.imp



  pmm.type <- pmm.params$pmm.type
  pmm.k <- pmm.params$pmm.k
  pmm.link <- pmm.params$pmm.link
  pmm.save.vars <- pmm.params$pmm.save.vars



  if(subsample == 1 & early.stopping.epochs>1){
    stop("To use early stopping based on validation error, please set subsample < 1.")
  }

  #if (save.model & is.null(path)) {
  #  stop("Please specify a path to save the imputation model.")
  # }

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

  #data.tensor <- torch_dataset(data, scaler = scaler, device = device)
  #n.features <- data.tensor$.ncol()

  data.tensor<-torch_dataset(data, scaler = scaler, lower=lower,upper=upper, categorical.encoding = categorical.encoding, initial.imp = initial.imp)
  # data.tensor[1]
  #pre.obj$data.tensor
  #if(!torch_is_floating_point(data.tensor)){
  # data.tensor<-data.tensor$to(dtype=torch_float())
  # }


  #n.features <- data.tensor$size()[[2]]

  n.samples <- nrow(data)

  ###change this later
  # check pmm
  sort.result <- sortNA(data)
  sorted.dt <- sort.result$sorted.dt
  sorted.types <- feature_type(sorted.dt)
  sorted.naSums <- colSums(is.na(sorted.dt))
  check_pmm(pmm.type = pmm.type, subsample = subsample, input.dropout = input.dropout, hidden.dropout = hidden.dropout, Nrow = n.samples, sorted.naSums, sorted.types, pmm.k)


  # imputed data
  imputed.data <- vector("list", length = m)
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



  # yhatobs.list
  if (isTRUE(pmm.type == 1)) {
    yhatobs.list <- yhatobs_pmm1(module="dae",
                                 data = data, categorical.encoding = categorical.encoding, device = device, na.loc = na.loc, na.vars = na.vars, extra.vars = extra.vars, pmm.link = pmm.link,
                                 epochs = epochs, batch.size = batch.size, drop.last = drop.last, shuffle = shuffle,
                                 optimizer = optimizer, learning.rate = learning.rate, weight.decay = weight.decay, momentum = momentum, eps = eps,
                                 encoder.structure = encoder.structure, latent.dim = latent.dim, decoder.structure = decoder.structure,
                                 act = act, init.weight = init.weight, scaler = scaler,initial.imp = initial.imp, lower=lower, upper=upper,
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




    #
  } else {

    train.idx <- sample(1:n.samples, size = floor(subsample * n.samples), replace = FALSE)
    valid.idx <- setdiff(1:n.samples, train.idx)


    train.samples <- length(train.idx)
    valid.samples <- length(valid.idx)

    train.original.data<-torch_dataset_idx(data, idx=train.idx, scaler = scaler,lower=lower,upper=upper,  categorical.encoding = categorical.encoding, initial.imp = initial.imp)
    valid.original.data<-torch_dataset_idx(data, idx=valid.idx, scaler = scaler,lower=lower,upper=upper,  categorical.encoding = categorical.encoding, initial.imp = initial.imp)



    train.batches<-batch_set(n.samples = train.samples, batch.size = batch.size, drop.last = drop.last)
    train.batch.set<-train.batches$batch.set
    train.num.batches<-train.batches$num.batches


    valid.batches<-batch_set(n.samples = valid.samples, batch.size = batch.size, drop.last = drop.last)
    valid.batch.set<-valid.batches$batch.set
    valid.num.batches<-valid.batches$num.batches

  }



# grid --------------------------------------------------------------------



  model.params<-expand.grid(dropout.grid)
  n.models<-nrow(model.params)
  Model.list<-vector("list", length=n.models)

  #result.list<-vector("list", length=n.models)

  na.loc <- pre.obj$na.loc
  na.vars <- pre.obj$ordered.names[colSums(na.loc) != 0]
  num.navars<-length(na.vars)


  imputed.missing<-replicate(num.navars, list())
  names(imputed.missing)<-na.vars

  for (var in na.vars) {
    imputed.missing[[var]]<- vector("list", length = n.models)
    n.na<-sum(na.loc[,var])

    for(j in seq_len(n.models)){
      imputed.missing[[var]][[j]]<-data.frame(matrix(NA,nrow=n.na,ncol=m))

      colnames(imputed.missing[[var]][[j]])<-paste0("m",1:m)
    }

  }



  for(j in seq_len(n.models)){

    model <- dae(categorical.encoding = categorical.encoding, n.others = n.others, cardinalities = cardinalities, embedding.dim = embedding.dim,
                 input.dropout = model.params$input.dropout[j],  hidden.dropout = model.params$hidden.dropout[j], encoder.structure = encoder.structure, latent.dim = latent.dim, decoder.structure = decoder.structure, act = act)$to(device = device)

    model <-model$to(device=device)


    model$apply(init_xavier_normal)

    if (init.weight == "he.normal") {
      model$apply(init_he_normal)
    }else if (init.weight == "he.uniform") {
      model$apply(init_he_uniform)
    }else if (init.weight == "xavier.normal") {
      model$apply(init_xavier_normal)
    } else if (init.weight == "xavier.uniform") {
      model$apply(init_xavier_uniform)
    } else if (init.weight == "xavier.midas") {
      model$apply(init_xavier_midas)
    }



    # define the loss function for different variables
    num_loss <- nn_mse_loss(reduction = "mean")
    logi_loss <- nn_bce_with_logits_loss(reduction = "mean")
    bin_loss <- nn_bce_with_logits_loss(reduction = "mean")
    multi_loss <- nn_cross_entropy_loss(reduction = "mean")



    # choose optimizer & learning rate
    if (optim.name  == "adamW") {
      optimizer <- torchopt::optim_adamw(model$parameters, lr = learning.rate, eps = eps, weight_decay = weight.decay)
    } else if (optim.name  == "sgd") {
      optimizer <- optim_sgd(model$parameters, lr = learning.rate, momentum = momentum, dampening = dampening, weight_decay = weight.decay)
    } else if (optim.name  == "adam") {# torch default eps = 1e-08, tensorfolow default eps =1e-07
      optimizer <- optim_adam(model$parameters, lr = learning.rate, eps = eps, weight_decay = weight.decay)
    } else if (optim.name  == "adadelta") {
      optimizer <- optim_adadelta(model$parameters, lr = learning.rate, rho = rho, eps = eps, weight_decay = weight.decay)
    } else if (optim.name  == "adagrad") {
      optimizer <- optim_adagrad(model$parameters, lr = learning.rate, lr_decay = learning.rate.decay, eps = eps, weight_decay = weight.decay)
    } else if (optim.name  == "rmsprop") {
      optimizer <- optim_rmsprop(model$parameters, lr = learning.rate, alpha = alpha, eps = eps, weight_decay = weight.decay, momentum = momentum)
    }

    # epochs: number of iterations
    if(verbose){
      print("Running midae().")
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

      #check:  equivalent
      # permute[1]
      # train.original.data$num.tensor[permute[1],]
      # train.data$num.tensor[1,]



      for(i in 1:train.num.batches){

        b<-list()
        b.index<-train.batch.set[[i]]

        b$data<-lapply(train.data, function(x) x[b.index])
        #index in original full data
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


        #X model(b$data$to(device=device)  can't use to device for a list of object, need to feed in separately
        #Out <- model(b$data$to(dtype = torch_float(),device = device))

        # numeric
        if (length(pre.obj$num) > 0) {
          num.cost <- vector("list", length = length(pre.obj$num))
          names(num.cost) <- pre.obj$num

          for (idx in seq_along(pre.obj$num.idx)) {
            var<-pre.obj$num[idx]
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
            var<-pre.obj$logi[idx]
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
            var<-pre.obj$bin[idx]
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
            var<-pre.obj$multi[idx]
            obs.idx <- which(pre.obj$na.loc[as.array(b$index), var] != TRUE)
            #which(pre.obj$na.loc[as.array(b$index), var] == TRUE)
            multi.cost[[var]] <- multi_loss(input = Out[obs.idx, pre.obj$multi.idx[[var]]], target = multi.tensor[obs.idx, idx])
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



        # Total cost
        cost <- sum(total.num.cost, total.bin.cost, total.multi.cost)

        # zero out the gradients
        optimizer$zero_grad()

        cost$backward()

        # update params
        optimizer$step()


        batch.loss <- cost$item()
        train.loss <- train.loss + batch.loss

        #if (save.model & epoch == epochs) {
        #torch::torch_save(model, path = path)
        # }
        #})
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




          #b$data$num.tensor[1,]
          # b$index[1]
          #data.tensor$num.tensor[b$index[1],]


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
              var<-pre.obj$logi[idx]
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
              var<-pre.obj$bin[idx]
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
              var<-pre.obj$multi[idx]
              obs.idx <- which(pre.obj$na.loc[as.array(b$index), var] != TRUE)
              #which(pre.obj$na.loc[as.array(b$index), var] == TRUE)
              multi.cost[[var]] <- multi_loss(input = Out[obs.idx, pre.obj$multi.idx[[var]]], target = multi.tensor[obs.idx, idx])
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


          batch.loss <- cost$item()
          valid.loss <- valid.loss + batch.loss
          #})
        }
      }





      #each epoch
      if(subsample==1){
        if (verbose & (epoch == 1 | epoch %% print.every.n == 0)) {
          cat(sprintf("Loss at epoch %d: %1f\n", epoch, train.loss / train.num.batches))
        }
        if (save.model & epoch == epochs) {
          torch_save(model, path = path)
        }
      }else if(subsample<1){
        valid.epoch.loss <- valid.loss / valid.num.batches
        if (verbose & (epoch == 1 | epoch %% print.every.n == 0)) {
          cat(sprintf("Loss at epoch %d: training: %3f, validation: %3f\n", epoch, train.loss / train.num.batches,valid.epoch.loss))
        }
        if(early.stopping.epochs >1){
          if(valid.epoch.loss < best.loss){
            best.loss <- valid.epoch.loss
            best.epoch<-epoch
            num.nondecresing.epochs <- 0
            torch_save(model, path = path)
          }else{
            num.nondecresing.epochs <- num.nondecresing.epochs + 1
            if(num.nondecresing.epochs >= early.stopping.epochs){
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


    # model <- torch::torch_load(path = path)
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


      #output.data <- model(data.tensor$num.tensor$to(device = device),data.tensor$logi.tensor$to(device = device),data.tensor$bin.tensor$to(device = device),data.tensor$multi.tensor$to(device = device))

      #output.data = output.data
      # pre.obj = pre.obj
      #scaler = scaler

      output.data<-Out$to(device = "cpu")

      imp.data <- postprocess(output.data = output.data, pre.obj = pre.obj, scaler = scaler)

      if (isFALSE(save.model)) {
        # don't need to save pmm values
        if (is.null(pmm.type)) {
          for (var in na.vars) {
            imputed.missing[[var]][[j]][,i]<- imp.data[[var]][na.loc[, var]]
          }
        } else if (pmm.type == 0 | pmm.type == 2) {
          for (var in na.vars) {
            if (var %in% pre.obj$num) {
              # numeric or binary? check binary
              yhatobs <- imp.data[[var]][!na.loc[, var]]
              yhatmis <- imp.data[[var]][na.loc[, var]]
              imputed.missing[[var]][[j]][,i] <- pmm(yhatobs = yhatobs, yhatmis = yhatmis, yobs = yobs.list[[var]], k = pmm.k)
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

              imputed.missing[[var]][[j]][,i]<-pmm(yhatobs = yhatobs, yhatmis = yhatmis, yobs = yobs.list[[var]], k = pmm.k)

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

              imputed.missing[[var]][[j]][,i]<-pmm(yhatobs = yhatobs, yhatmis = yhatmis, yobs = yobs.list[[var]], k = pmm.k)

              #level.idx <- pmm.multiclass(yhatobs = yhatobs, yhatmis = yhatmis, yobs = yobs.list[[var]], k = pmm.k)
              #data[[var]][na.loc[, var]] <- levels(data[[var]])[level.idx]
            } else if (var %in% pre.obj$multi) {
              # multiclass
              var.idx <- pre.obj$multi.idx[[var]]

              # probability for each class of a multiclass variable
              yhatobs <- as_array(output.data[!na.loc[, var], var.idx])
              yhatmis <- as_array(output.data[na.loc[, var], var.idx])

              level.idx <- pmm.multiclass(yhatobs = yhatobs, yhatmis = yhatmis, yobs = yobs.list[[var]], k = pmm.k)
              imputed.missing[[var]][[j]][,i] <- levels(data[[var]])[level.idx]
            }
          }
        } else if (pmm.type == "auto") {
          for (var in na.vars) {
            yhatobs <- imp.data[[var]][!na.loc[, var]]
            yhatmis <- imp.data[[var]][na.loc[, var]]

            if (var %in% pre.obj$num) {
              # numeric
              imputed.missing[[var]][[j]][,i] <- pmm(yhatobs = yhatobs, yhatmis = yhatmis, yobs = yobs.list[[var]], k = pmm.k)
            } else {
              # binary or multiclass: no pmm
              imputed.missing[[var]][[j]][,i] <- yhatmis
            }
          }
        } else if (pmm.type == 1) {
          for (var in na.vars) {
            yhatobs <- yhatobs.list[[var]]
            if (var %in% pre.obj$num) {
              yhatmis <- imp.data[[var]][na.loc[, var]]
              imputed.missing[[var]][[j]][,i] <- pmm(yhatobs = yhatobs, yhatmis = yhatmis, yobs = yobs.list[[var]], k = pmm.k)
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

              imputed.missing[[var]][[j]][,i] <-pmm(yhatobs = yhatobs, yhatmis = yhatmis, yobs = yobs.list[[var]], k = pmm.k)

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

              imputed.missing[[var]][[j]][,i] <-pmm(yhatobs = yhatobs, yhatmis = yhatmis, yobs = yobs.list[[var]], k = pmm.k)

              #level.idx <- pmm.multiclass(yhatobs = yhatobs, yhatmis = yhatmis, yobs = yobs.list[[var]], k = pmm.k)
              #data[[var]][na.loc[, var]] <- levels(data[[var]])[level.idx]

            } else if (var %in% pre.obj$multi) {
              # multiclass
              var.idx <- pre.obj$multi.idx[[var]]
              yhatmis <- as_array(output.data[na.loc[, var], var.idx])
              level.idx <- pmm.multiclass(yhatobs = yhatobs, yhatmis = yhatmis, yobs = yobs.list[[var]], k = pmm.k)
              imputed.missing[[var]][[j]][,i] <- levels(data[[var]])[level.idx]
            }
          }
        }
      } else {
        # need to save pmm values
        if (is.null(pmm.type)) {
          for (var in na.vars) {
            imputed.missing[[var]][[j]][,i] <- imp.data[[var]][na.loc[, var]]
          }
        } else if (pmm.type == 0 | pmm.type == 2) {
          for (var in na.vars) {
            if (var %in% pre.obj$num) {
              # numeric or binary? check binary
              yhatobs <- imp.data[[var]][!na.loc[, var]]
              yhatmis <- imp.data[[var]][na.loc[, var]]
              imputed.missing[[var]][[j]][,i] <- pmm(yhatobs = yhatobs, yhatmis = yhatmis, yobs = yobs.list[[var]], k = pmm.k)
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

              imputed.missing[[var]][[j]][,i] <- pmm(yhatobs = yhatobs, yhatmis = yhatmis, yobs = yobs.list[[var]], k = pmm.k)
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

              imputed.missing[[var]][[j]][,i] <- pmm(yhatobs = yhatobs, yhatmis = yhatmis, yobs = yobs.list[[var]], k = pmm.k)
            } else if (var %in% pre.obj$multi) {
              # multiclass
              var.idx <- pre.obj$multi.idx[[var]]

              # probability for each class of a multiclass variable
              yhatobs <- as_array(output.data[!na.loc[, var], var.idx])
              yhatmis <- as_array(output.data[na.loc[, var], var.idx])

              level.idx <- pmm.multiclass(yhatobs = yhatobs, yhatmis = yhatmis, yobs = yobs.list[[var]], k = pmm.k)
              imputed.missing[[var]][[j]][,i] <- levels(data[[var]])[level.idx]
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
              imputed.missing[[var]][[j]][,i] <- pmm(yhatobs = yhatobs, yhatmis = yhatmis, yobs = yobs.list[[var]], k = pmm.k)
              # save yhatobs
              yhatobs.list[[i]][[var]] <- yhatobs
            } else {
              # binary or multiclass: no pmm
              imputed.missing[[var]][[j]][,i] <- yhatmis
            }
          }
        } else if (pmm.type == 1) {
          for (var in na.vars) {
            yhatobs <- yhatobs.list[[var]]
            if (var %in% pre.obj$num) {
              yhatmis <- imp.data[[var]][na.loc[, var]]
              imputed.missing[[var]][[j]][,i] <- pmm(yhatobs = yhatobs, yhatmis = yhatmis, yobs = yobs.list[[var]], k = pmm.k)
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

              imputed.missing[[var]][[j]][,i] <-pmm(yhatobs = yhatobs, yhatmis = yhatmis, yobs = yobs.list[[var]], k = pmm.k)

            } else if (var %in% pre.obj$multi) {
              # multiclass
              var.idx <- pre.obj$multi.idx[[var]]
              yhatmis <- as_array(output.data[na.loc[, var], var.idx])
              level.idx <- pmm.multiclass(yhatobs = yhatobs, yhatmis = yhatmis, yobs = yobs.list[[var]], k = pmm.k)
              imputed.missing[[var]][[j]][,i] <- levels(data[[var]])[level.idx]
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



    }

    for(var in na.vars){
      K<-nrow(imputed.missing[[var]][[j]])
      imputed.missing[[var]][[j]]<-imputed.missing[[var]][[j]] %>%
        tibble::add_column(input.dropout=rep(model.params$input.dropout[j],K),
                           hidden.dropout=rep(model.params$hidden.dropout[j],K)
        )
    }

  }#end of j in N model

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




#return a list of [[N]][[M]] all results of N models, each model have M imputed dataset
#Note: not memory efficient
#return all imputed datasets
tune_dropout_midae_all<- function(data, dropout.grid = list(input.dropout=c(0,0.25,0.5),hidden.dropout=c(0, 0.25, 0.5)),
                               m = 5, categorical.encoding = "embeddings", device = "cpu",
                  epochs = 5, batch.size = 32,
                  subsample = 1,
                  early.stopping.epochs = 1,
                  pmm.params=list(),
                  dae.params=list(),
                  loss.na.scale = FALSE,
                  verbose = TRUE, print.every.n = 1,
                  save.model = FALSE, path = NULL) {


  pmm.params <- do.call("dae_pmm_default", pmm.params)
  dae.params <- do.call("dae_default", dae.params)

  model.params<-expand.grid(dropout.grid)
  n.models<-nrow(model.params)

  result.list<-vector("list", length=n.models)

  for(i in seq_len(n.models)){
    dae.params$input.dropout <- model.params$input.dropout[i]
    dae.params$hidden.dropout <- model.params$hidden.dropout[i]
  result.list[[i]]<-midae(data = data,
          m = m, categorical.encoding = categorical.encoding, device = device,
          epochs = epochs, batch.size = batch.size,
          subsample = subsample,
          early.stopping.epochs = early.stopping.epochs,
          dae.params=dae.params,
          pmm.params=pmm.params,
          loss.na.scale = loss.na.scale,
          verbose = verbose, print.every.n = print.every.n,
          save.model = save.model, path = path)
  }

  result.list

}





