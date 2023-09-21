#' Multiple imputation through denoising autoencoders with dropout (Use embeddings)
#' @param data A data frame, tibble or data table with missing values.
#' @param m The number of imputed datasets.
#' @param categorical.encoding The method for representing multi-class categorical features. Can be either "embeddings" or "onehot".
#' @param device Device to use. Either "cpu" or "cuda" for GPU.
#' @param pmm.type The type of predictive mean matching (PMM). Possible values:
#' \itemize{
#'  \item \code{NULL}: Imputations without PMM;
#'  \item \code{0}: Imputations with PMM type 0;
#'  \item \code{1}: Imputations with PMM type 1;
#'  \item \code{2}: Imputations with PMM type 2;
#'  \item \code{"auto"} (Default): Imputations with PMM type 2 for numeric/integer variables; imputations without PMM for categorical variables.
#' }
#' @param pmm.k The number of donors for predictive mean matching. Default: 5
#' @param pmm.link The link for predictive mean matching in binary variables
#' \itemize{
#'  \item \code{"prob"} (Default): use probabilities;
#'  \item \code{"logit"}: use logit values.
#' }
#' @param pmm.save.vars The names of variables whose predicted values of observed entries will be saved. Only use for PMM.
#' @param epochs The number of training epochs (iterations).
#' @param batch.size The size of samples in each batch. Default: 32.
#' @param drop.last Whether or not to drop the last batch. Default: FALSE
#' @param subsample The subsample ratio of training data. Default: 1.
#' @param shuffle Whether or not to shuffle training data. Default: TRUE
#' @param input.dropout The dropout probability of the input layer.
#' @param hidden.dropout The dropout probability of the hidden layers.
#' @param optimizer The name of the optimizer. Options are : "adamW" (default), "adam" and "sgd".
#' @param learning.rate The learning rate. The default value is 0.001.
#' @param weight.decay Weight decay (L2 penalty). The default value is 0.
#' @param momentum Parameter for "sgd" optimizer. It is used for accelerating SGD in the relevant direction and dampens oscillations.
#' @param eps A small positive value used to prevent division by zero for the "adamW" optimizer. Default: 1e-07.
#' @param encoder.structure A vector indicating the structure of encoder. Default: c(128,64,32)
#' @param latent.dim The size of latent layer. The default value is 16.
#' @param decoder.structure A vector indicating the structure of decoder. Default: c(32,64,128)
#' @param act The name of activation function. Can be: "relu", "elu", "leaky.relu", "tanh", "sigmoid" and "identity".
#' @param init.weight Techniques for weights initialization. Can be "he.normal","he.uniform", "xavier.uniform", "xavier.normal" or "xavier.midas".
#' @param scaler The name of scaler for transforming numeric features. Can be "standard", "minmax" ,"decile" or "none".
#' @param loss.na.scale Whether to multiply the ratio of missing values in  a feature to calculate the loss function. Default: FALSE.
#' @param early.stopping.epochs An integer value \code{k}. Mivae training will stop if the validation performance has not improved for \code{k} epochs, only used when \code{subsample}<1. Default: 10.
#' @param verbose Whether or not to print training loss information. Default: TRUE.
#' @param print.every.n If verbose is set to TRUE, print out training loss for every n epochs.
#' @param save.model Whether or not to save the imputation model. Default: FALSE.
#' @param path The path where the final imputation model will be saved.
#' @importFrom torch dataloader nn_mse_loss nn_bce_with_logits_loss nn_cross_entropy_loss optim_adam optim_sgd torch_save torch_load torch_argmax dataloader_make_iter dataloader_next
#' @importFrom torchopt optim_adamw
#' @export
#' @examples
#' withNA.df <- createNA(data = iris, p = 0.2)
#' imputed.data <- midae(data = withNA.df, m = 5, epochs = 5, path = file.path(tempdir(), "midaemodel.pt"))
midae <- function(data, m = 5, categorical.encoding = "embeddings", device = "cpu",
                  epochs = 5, batch.size = 32,
                  subsample = 1,
                  early.stopping.epochs = 1,
                  dae.params=list(),
                  pmm.params=list(),
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

  optimizer <- dae.params$optimizer
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
    # use all available data
    #data.tensor <- torch_dataset(data, scaler = scaler, device = device)
    #train.dl <- torch::dataloader(dataset = data.tensor, batch_size = batch.size, shuffle = shuffle,  drop_last = FALSE)
    #train.num.batches <- length(train.dl)
    #test
    #iter<-train.dl$.iter()
    # b<-iter$.next()
    # b
   train.samples <- n.samples
   train.idx <- 1:n.samples
   #train.idx<-sample(1:n.samples, size = n.samples, replace = FALSE)
   train.batches<-batch_set(n.samples = train.samples, batch.size = batch.size, drop.last = drop.last)
   train.batch.set<-train.batches$batch.set
   train.num.batches<-train.batches$num.batches
   #data.tensor<-torch_dataset(data, scaler = scaler, device = device)
   train.original.data<-data.tensor




    #
  } else {

    train.idx <- sample(1:n.samples, size = floor(subsample * n.samples), replace = FALSE)
    valid.idx <- setdiff(1:n.samples, train.idx)



    #train.ds <- torch_dataset_idx(data = data, idx = train.idx, scaler = scaler)
    #valid.ds <- torch_dataset_idx(data = data, idx = valid.idx, scaler = scaler)

    #train.dl <- dataloader(dataset = train.ds, batch_size = batch.size, shuffle = shuffle,  drop_last = FALSE)
    #valid.dl <- dataloader(dataset = valid.ds, batch_size = batch.size, shuffle = shuffle,  drop_last = FALSE)


    #train.num.batches <- length(train.dl)
    #valid.num.batches <- length(valid.dl)

    train.samples <- length(train.idx)
    valid.samples <- length(valid.idx)

    train.original.data<-torch_dataset_idx(data, idx=train.idx, scaler = scaler,lower=lower,upper=upper,  categorical.encoding = categorical.encoding, initial.imp = initial.imp)
    valid.original.data<-torch_dataset_idx(data, idx=valid.idx, scaler = scaler,lower=lower,upper=upper,  categorical.encoding = categorical.encoding, initial.imp = initial.imp)
    #train.original.data<- data.tensor[train.idx]
    #valid.original.data<-data.tensor[valid.idx]
    #data.tensor<-torch_dataset(data, scaler = scaler, device = device)
   # train.original.data<-dataset_subset(data.tensor,train.idx)$dataset
    #valid.original.data<-dataset_subset(data.tensor,valid.idx)$dataset
      #train.original.data<-data.tensor[train.idx,]
    #valid.original.data<-data.tensor[valid.idx,]



    train.batches<-batch_set(n.samples = train.samples, batch.size = batch.size, drop.last = drop.last)
    train.batch.set<-train.batches$batch.set
    train.num.batches<-train.batches$num.batches


    valid.batches<-batch_set(n.samples = valid.samples, batch.size = batch.size, drop.last = drop.last)
    valid.batch.set<-valid.batches$batch.set
    valid.num.batches<-valid.batches$num.batches

  }




  model <- dae(categorical.encoding = categorical.encoding, n.others = n.others, cardinalities = cardinalities, embedding.dim = embedding.dim,
               input.dropout = input.dropout, hidden.dropout = hidden.dropout, encoder.structure = encoder.structure, latent.dim = latent.dim, decoder.structure = decoder.structure, act = act)$to(device = device)

  model <-model$to(device=device)




  if (init.weight == "he.normal") {
    model$apply(init_he_normal)
  }else if (init.weight == "he.normal.dropout") {
    model$apply(init_he_normal_dropout)
  }else if (init.weight == "he.uniform") {
    model$apply(init_he_uniform)
  }else if (init.weight == "he.normal.elu") {
    model$apply(init_he_normal_elu)
  }else if (init.weight == "he.normal.elu.dropout") {
    model$apply(init_he_normal_elu_dropout)
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



  # define the loss function for different variables
  num_loss <- torch::nn_mse_loss(reduction = "mean")
  logi_loss <- torch::nn_bce_with_logits_loss(reduction = "mean")
  bin_loss <- torch::nn_bce_with_logits_loss(reduction = "mean")
  multi_loss <- torch::nn_cross_entropy_loss(reduction = "mean")



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
      permute<-torch::torch_randperm(train.samples)+1L

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
        permute<-torch::torch_randperm(valid.samples)+1L
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
        torch::torch_save(model, path = path)
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
          torch::torch_save(model, path = path)
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
    model <- torch::torch_load(path = path)
  }

  model<-model$to(device=device)


  # model <- torch::torch_load(path = path)
  model$eval()

  # The whole dataset
 # eval_dl <- torch::dataloader(dataset = data.tensor, batch_size = n.samples, shuffle = FALSE)


  #wholebatch <- eval_dl %>%
    #torch::dataloader_make_iter() %>%
    #torch::dataloader_next()




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
  }




  if (isFALSE(save.model)) {
    return(imputed.data)
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




    midae.obj <- list("imputed.data" = imputed.data, "model.path" = path, "params" = params)
    print(paste("The DAE multiple imputation model is saved in ", path))
    class(midae.obj) <- "midaeObj"
    return(midae.obj)
  }
}



move_to_device<-function(tensor, device){
  if(is.null(tensor)){
    tensor
  }else{
    tensor$to(device=device)
  }
}



#' Auxiliary function for pmm.params for midae
#' @description Auxiliary function for setting up the default pmm-related parameters for midae
dae_pmm_default <-function(pmm.type = "auto", pmm.k = 5, pmm.link = "prob", pmm.save.vars = NULL){
  list(pmm.type = pmm.type, pmm.k = pmm.k, pmm.link = pmm.link, pmm.save.vars = pmm.save.vars)
}

#' Auxiliary function for dae.params
#' @description Auxiliary function for setting up the default dae-related hyperparameters for midae
dae_default<-function(shuffle = TRUE, drop.last = FALSE,
                      input.dropout = 0.2, hidden.dropout = 0.5,
                      optimizer = "adamW", learning.rate = 0.001, weight.decay = 0.01, momentum = 0, dampening = 0, eps = 1e-08, rho = 0.9, alpha = 0.99, learning.rate.decay = 0,
                      encoder.structure = c(128, 64, 32), latent.dim = 16, decoder.structure = c(32, 64, 128),
                      act = "relu", init.weight = "he.normal", scaler = "standard",initial.imp = "sample", lower=0.25, upper=0.75){
  list(shuffle = shuffle, drop.last = drop.last,
       input.dropout = input.dropout, hidden.dropout = hidden.dropout,
       optimizer = optimizer, learning.rate = learning.rate, weight.decay = weight.decay, momentum = momentum, dampening = dampening, eps = eps, rho = rho, alpha = alpha, learning.rate.decay = learning.rate.decay,
       encoder.structure = encoder.structure, latent.dim = latent.dim, decoder.structure = decoder.structure,
       act = act, init.weight = init.weight, scaler = scaler,initial.imp = initial.imp, lower=lower, upper=upper)
}

