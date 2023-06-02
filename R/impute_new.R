#' Impute new data with a saved \code{midae} or \code{mivae} imputation model
#' @param  object A saved imputer object created by \code{midae(..., save.models = TRUE)} or \code{mivae(..., save.models = TRUE)}.
#' @param  newdata A data frame, tibble or data.table. New data with missing values.
#' @param  pmm.k The number of donors for predictive mean matching. If \code{NULL} (the default), the \code{pmm.k} value in the saved imputer object will be used.
#' @param  m The number of imputed datasets. If \code{NULL} (the default), the \code{m} value in the saved imputer object will be used.
#' @return A list of \code{m} imputed datasets for new data.
#' @importFrom torch dataloader torch_load dataloader_make_iter dataloader_next
#' @export
impute_new <- function(object, newdata, pmm.k = NULL, m = NULL, verbose = FALSE) {
  # the saved model
  path <- object$model.path

  # the saved params
  params <- object$params

  scaler <- params$scaler

  categorical.encoding<-params$categorical.encoding


  pre.obj <- preprocess(newdata, scaler = scaler, categorical.encoding = categorical.encoding)

  #torch.data <- torch_dataset(newdata, scaler = scaler, categorical.encoding = categorical.encoding)

  data.tensor<-torch_dataset(newdata, scaler = scaler, categorical.encoding = categorical.encoding)
  #

  n.samples <- nrow(newdata)
  #n.features <- torch.data$.ncol()
  #n.samples <- torch.data$.length()


  cardinalities<-pre.obj$cardinalities
  embedding.dim<-pre.obj$embedding.dim
  origin.names <- colnames(newdata)
  #n.num+n.logi+n.bin
  n.others <- length(origin.names)-length(cardinalities)

  # new data
  new.na.loc <- pre.obj$na.loc
  new.na.vars <- pre.obj$ordered.names[colSums(new.na.loc) != 0]




  # check new data and give some warning messages (unfinished)...........................................................................
  if (length(new.na.vars) == 0) {
    stop("No missing values in new data.")
  }

  na.vars <- params$na.vars
  extra.vars <- params$extra.vars
  save.vars <- c(na.vars, extra.vars)




  if (!is.null(params$pmm.type)) {
    if (params$pmm.type == "auto") {
      # pmm.type="auto":  only match numeric variables

      nopmm.vars <- new.na.vars[!new.na.vars %in% save.vars]
      if (any(nopmm.vars %in% pre.obj$num)) {
        # stop("Some numeric variables in the new data has missing values but they are not specified in `pmm.save.vars`. If you want to use PMM (auto), please re-specify `pmm.save.vars` and re-train the imputer. Otherwise, you can set `pmm.type = NULL`. ")
        # more detail information....................................
        unsaved <- nopmm.vars[nopmm.vars %in% pre.obj$num]
        msg1 <- paste("There exists at least one missing value in the following numeric variable(s): ", paste(unsaved, collapse = ";"),
          ".",
          sep = ""
        )
        msg2 <- paste("However, your hadn't specified them in `pmm.save.vars`.")
        msg3 <- paste("Please either add these variables in the argument `pmm.save.vars` or set `pmm.type = NULL` and re-train the imputer.")
        stop(paste(msg1, msg2, msg3, sep = "\n"))
      }
    } else if (!all(new.na.vars %in% save.vars)) {
      # pmm.type = 0,1,2:  match all variables
      # stop("Some variables in the new data has missing values but they are not specified in `pmm.save.vars`. If you want to use PMM, please re-specify `pmm.save.vars` and re-train the imputer. Otherwise, you can set `pmm.type = NULL`.")
      # more detail information....................................
      unsaved <- new.na.vars[!new.na.vars %in% save.vars]
      msg1 <- paste("There exists at least one missing value in the following variable(s): ", paste(unsaved, collapse = ";"),
        ".",
        sep = ""
      )
      msg2 <- paste("However, your hadn't specified them in `pmm.save.vars`.")
      msg3 <- paste("Please either add these variables in the argument `pmm.save.vars` or set `pmm.type = NULL` and re-train the imputer.")
      stop(paste(msg1, msg2, msg3, sep = "\n"))
    }
  }




  if (is.null(m)) {
    m <- params$m
  } else {
    if (m <= params$m) {
      m <- m
    } else {
      stop("The value of m in impute.new() cannot be larger than the value of m in $impute().")
    }
  }

  # pmm

  pmm.type <- params$pmm.type
  pmm.link <- params$pmm.link
  if (is.null(pmm.k)) {
    pmm.k <- params$pmm.k
  } else {
    pmm.k <- pmm.k
  }


  yobs.list <- params$yobs.list
  yhatobs.list <- params$yhatobs.list








  model <- torch::torch_load(path)

  model$eval()




  # The whole dataset
  #eval_dl <- torch::dataloader(dataset = torch.data, batch_size = n.samples, shuffle = FALSE)


 # wholebatch <- eval_dl %>%
   # torch::dataloader_make_iter() %>%
   # torch::dataloader_next()



  # imputed data
  imputed.data <- vector("list", length = m)


  for (i in seq_len(m)) {


    if(categorical.encoding=="embeddings"){
      Out <- model(num.tensor=data.tensor$num.tensor,logi.tensor=data.tensor$logi.tensor,bin.tensor=data.tensor$bin.tensor, cat.tensor=data.tensor$multi.tensor)
    }else if(categorical.encoding=="onehot"){
      Out <- model(num.tensor=data.tensor$num.tensor,logi.tensor=data.tensor$logi.tensor,bin.tensor=data.tensor$bin.tensor, cat.tensor=data.tensor$onehot.tensor)
    }else{
      stop(cat('categorical.encoding can only be either "embeddings" or "onehot".\n'))
    }

    #output.data <- model(data.tensor$num.tensor$to(device = device),data.tensor$logi.tensor$to(device = device),data.tensor$bin.tensor$to(device = device),data.tensor$multi.tensor$to(device = device))

    #output.data = output.data
    # pre.obj = pre.obj
    #scaler = scaler

   # length(Out)

   # Out[[2]]
    #Out[[1]]
    #Out[[3]]
    #Out
    #output.data <- model(wholebatch$data)

    if (is.list(Out)) {
      # mivae
      output.data<-Out$reconstrx
    }else{
      output.data<-Out
    }

    imp.data <- postprocess(output.data = output.data, pre.obj = pre.obj, scaler = scaler)

    # new.na.loc <- pre.obj$na.loc
    # new.na.vars <- pre.obj$ordered.names[colSums(new.na.loc) != 0]

    # na.vars <- pre.obj$ordered.names[colSums(new.na.loc) != 0]

    if (is.null(pmm.type)) {
      for (var in new.na.vars) {
        newdata[[var]][new.na.loc[, var]] <- imp.data[[var]][new.na.loc[, var]]
      }
    } else if (pmm.type == 0 | pmm.type == 2) {
      for (var in new.na.vars) {
        if (var %in% pre.obj$num) {
          yhatmis <- imp.data[[var]][new.na.loc[, var]]
          newdata[[var]][new.na.loc[, var]] <- pmm(yhatobs = yhatobs.list[[i]][[var]], yhatmis = yhatmis, yobs = yobs.list[[var]], k = pmm.k)
        }else if (var %in% pre.obj$logi) {
          # binary
          var.idx <- pre.obj$logi.idx[[var]]
          if (pmm.link == "logit") {
            yhatmis <- as_array(output.data[new.na.loc[, var], var.idx])
          } else if (pmm.link == "prob") {
            transform_fn <- nn_sigmoid()
            yhatmis <- as_array(transform_fn(output.data[new.na.loc[, var], var.idx]))
          } else {
            stop("pmm.link has to be either `logit` or `prob`")
          }

          # use original data levels, avoiding the case when the new data do not include rare class
          newdata[[var]][new.na.loc[, var]] <- pmm(yhatobs = yhatobs.list[[i]][[var]], yhatmis = yhatmis, yobs = yobs.list[[var]], k = pmm.k)
        } else if (var %in% pre.obj$bin) {
          # binary
          var.idx <- pre.obj$bin.idx[[var]]
          if (pmm.link == "logit") {
            yhatmis <- as_array(output.data[new.na.loc[, var], var.idx])
          } else if (pmm.link == "prob") {
            transform_fn <- nn_sigmoid()
            yhatmis <- as_array(transform_fn(output.data[new.na.loc[, var], var.idx]))
          } else {
            stop("pmm.link has to be either `logit` or `prob`")
          }

          # use original data levels, avoiding the case when the new data do not include rare class
          newdata[[var]][new.na.loc[, var]] <- pmm(yhatobs = yhatobs.list[[i]][[var]], yhatmis = yhatmis, yobs = yobs.list[[var]], k = pmm.k)
        } else if (var %in% pre.obj$multi) {
          # multiclass
          var.idx <- pre.obj$multi.idx[[var]]
          # probability for each class of a multiclass variable
          yhatmis <- as_array(output.data[new.na.loc[, var], var.idx])
          level.idx <- pmm.multiclass(yhatobs = yhatobs.list[[i]][[var]], yhatmis = yhatmis, yobs = yobs.list[[var]], k = pmm.k)
          newdata[[var]][new.na.loc[, var]] <- levels(object$imputed.data[[1]][[var]])[level.idx]
        }
      }

    } else if (pmm.type == 1) {
      for (var in new.na.vars) {
        if (var %in% pre.obj$num) {
          yhatmis <- imp.data[[var]][new.na.loc[, var]]
          newdata[[var]][new.na.loc[, var]] <- pmm(yhatobs = yhatobs.list[[var]], yhatmis = yhatmis, yobs = yobs.list[[var]], k = pmm.k)
        }  else if (var %in% pre.obj$logi) {
          # binary
          var.idx <- pre.obj$logi.idx[[var]]
          if (pmm.link == "logit") {
            yhatmis <- as_array(output.data[new.na.loc[, var], var.idx])
          } else if (pmm.link == "prob") {
            transform_fn <- nn_sigmoid()
            yhatmis <- as_array(transform_fn(output.data[new.na.loc[, var], var.idx]))
          } else {
            stop("pmm.link has to be either `logit` or `prob`")
          }
          # use original data levels, avoiding the case when the new data do not include rare class
          newdata[[var]][new.na.loc[, var]] <- pmm(yhatobs = yhatobs.list[[var]], yhatmis = yhatmis, yobs = yobs.list[[var]], k = pmm.k)
        }else if (var %in% pre.obj$bin) {
          # binary
          var.idx <- pre.obj$bin.idx[[var]]
          if (pmm.link == "logit") {
            yhatmis <- as_array(output.data[new.na.loc[, var], var.idx])
          } else if (pmm.link == "prob") {
            transform_fn <- nn_sigmoid()
            yhatmis <- as_array(transform_fn(output.data[new.na.loc[, var], var.idx]))
          } else {
            stop("pmm.link has to be either `logit` or `prob`")
          }
          # use original data levels, avoiding the case when the new data do not include rare class
          newdata[[var]][new.na.loc[, var]] <- pmm(yhatobs = yhatobs.list[[var]], yhatmis = yhatmis, yobs = yobs.list[[var]], k = pmm.k)
        } else if (var %in% pre.obj$multi) {
          # multiclass
          var.idx <- pre.obj$multi.idx[[var]]
          # probability for each class of a multiclass variable
          yhatmis <- as_array(output.data[new.na.loc[, var], var.idx])
          level.idx <- pmm.multiclass(yhatobs = yhatobs.list[[var]], yhatmis = yhatmis, yobs = yobs.list[[var]], k = pmm.k)
          newdata[[var]][new.na.loc[, var]] <- levels(object$imputed.data[[1]][[var]])[level.idx]
        }
      }
    } else if (pmm.type == "auto") {
      for (var in new.na.vars) {
        if (var %in% pre.obj$num) {
          yhatmis <- imp.data[[var]][new.na.loc[, var]]
          newdata[[var]][new.na.loc[, var]] <- pmm(yhatobs = yhatobs.list[[i]][[var]], yhatmis = yhatmis, yobs = yobs.list[[var]], k = pmm.k)
        } else {
          newdata[[var]][new.na.loc[, var]] <- imp.data[[var]][new.na.loc[, var]]
        }

      }
    }





    imputed.data[[i]] <- newdata
  }

  imputed.data
}
