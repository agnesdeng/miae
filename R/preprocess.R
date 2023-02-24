#' Pre-process data before imputation: scale numeric and one-hot categorical
#' @importFrom torch torch_tensor torch_cat nnf_one_hot
#' @export
preprocess <- function(data, scaler) {

  Types <- feature_type(data)
  # Types
  original.names <- names(Types)

  num <- original.names[Types %in% "numeric"]
  bin <- original.names[Types %in% "binary"]
  multi <- original.names[Types %in% "multiclass"]
  ordered.names <- c(num, bin, multi)
  ordered.types <- Types[ordered.names]

  # ordered data according to numeric>binary>multiclass
  if(data.table::is.data.table(data)){
    ordered.data <- data[, ordered.names, with = FALSE]
  }else{
    ordered.data <- data[ordered.names]
  }

  na.loc <- is.na(ordered.data)


  #if numeric feature exists
  if(length(num)>=1){

    if(scaler!="none"){

      if(scaler=="minmax"){

        if(is.data.table(data)){
          num.obj <- minmax_scaler(data[, num, with = FALSE])
        }else{
          num.obj <- minmax_scaler(data[, num])
        }

        num.mat <- num.obj$minmax.mat
        num.tensor <- torch::torch_tensor(num.mat)
        colmin <- num.obj$colmin
        colmax <- num.obj$colmax



      }else if(scaler=="decile"){

        if(is.data.table(data)){
          num.obj <- decile_scaler(data[, num, with = FALSE])
        }else{
          num.obj <- decile_scaler(data[, num])
        }

        num.mat <- num.obj$decile.mat
        num.tensor <- torch::torch_tensor(num.mat)
        decile1 <- num.obj$decile1
        decile9 <- num.obj$decile9



      }else if(scaler=="standard"){

        if(is.data.table(data)){
          num.obj <- standard_scaler(data[, num, with = FALSE])
        }else{
          num.obj <- standard_scaler(data[, num])
        }

        num.mat <- num.obj$standard.mat
        num.tensor <- torch::torch_tensor(num.mat)
        colmean <- num.obj$colmean
        colsd <- num.obj$colsd


      }




    }else{
      #don't scale numeric data

      num.tibble<- dplyr::mutate_all(data[,num], ~ ifelse(is.na(.), median(., na.rm = TRUE), .))
      num.mat<-as.matrix(num.tibble)

      num.tensor <- torch::torch_tensor(num.mat)


    }

    num.idx <- vector("list", length = length(num))
    names(num.idx) <- num

    if (!is.null(num)) {
      for(i in seq_along(num)){
        num.idx[i]<-i
      }

    }




  }






  #if categorical data exists
  ##one-hot categorical data
  cat.names <- c(bin, multi)

  if(length(cat.names)>=1){

    if(data.table::is.data.table(data)){
      cat.mat <- data[, cat.names, with = FALSE]
    }else{
      cat.mat <- data[, cat.names, drop = FALSE]
    }





    # initial impute with mode
      cat.naSums <- colSums(is.na(cat.mat))
      cat.naidx <- which(cat.naSums != 0)
      cat.navars <- cat.names[cat.naidx]

      for (var in cat.navars) {
        na.idx <- which(is.na(cat.mat[[var]]))
        # Na.idx[[var]] <- na.idx
        # Impute the missing values of a vector with the mode (majority class) of observed values
        cat.mat[[var]] <- imp.mode(vec = cat.mat[[var]], na.idx = na.idx)
      }


      N.levels <- rep(NA, times = length(cat.names))
      names(N.levels) <- cat.names

      for (var in cat.names) {
        N.levels[var] <- nlevels(cat.mat[[var]])
      }




      bin.idx <- vector("list", length = length(bin))
      names(bin.idx) <- bin

      multi.idx <- vector("list", length = length(multi))
      names(multi.idx) <- multi



      if (!is.null(bin)) {
        bin.start <- length(num.idx) + 1
        for (var in bin) {
          bin.idx[[var]] <- bin.start:(bin.start + 1)
          bin.start <- bin.start + 2
        }
      }


      if (!is.null(multi)) {
        multi.start <- bin.start
        for (var in multi) {
          n.levels <- N.levels[var]
          multi.idx[[var]] <- multi.start:(multi.start + n.levels - 1)
          multi.start <- multi.start + n.levels
        }
      }

      # levels for each categorical variables
      cat.levels <- vector("list", length(c(bin, multi)))
      names(cat.levels) <- c(bin, multi)
      for (var in cat.names) {
        cat.levels[[var]] <- levels(data[[var]])
      }



      cat.list <- vector("list", length = length(cat.names))
      names(cat.list) <- cat.names

      for (var in cat.names) {
        cat.list[[var]] <- torch::nnf_one_hot(torch::torch_tensor(as.integer(cat.mat[[var]])))
      }


      # combine numeric data with one-hot encoded categorical variables
      cat.tensor <- torch::torch_cat(cat.list, dim = 2)
      data.tensor <- torch::torch_cat(list(num.tensor, cat.tensor), dim = 2)

  }else{
    #only numeric
    data.tensor <- num.tensor
    bin.idx <- NULL
    multi.idx <-NULL
    cat.levels <-NULL
  }




if(scaler=="minmax"){
  return(list(
    "data.tensor" = data.tensor,
    "na.loc" = na.loc,
    "colmin" = colmin, "colmax" = colmax,
    "original.names" = original.names,
    "ordered.names" = ordered.names,
    "ordered.types" = ordered.types,
    "num" = num,
    "bin" = bin,
    "multi" = multi,
    "num.idx" = num.idx,
    "bin.idx" = bin.idx,
    "multi.idx" = multi.idx,
    "cat.names" = cat.names,
    "cat.levels" = cat.levels
  ))
}else if(scaler=="decile"){
  return(list(
    "data.tensor" = data.tensor,
    "na.loc" = na.loc,
    "decile1" = decile1, "decile9" = decile9,
    "original.names" = original.names,
    "ordered.names" = ordered.names,
    "ordered.types" = ordered.types,
    "num" = num,
    "bin" = bin,
    "multi" = multi,
    "num.idx" = num.idx,
    "bin.idx" = bin.idx,
    "multi.idx" = multi.idx,
    "cat.names" = cat.names,
    "cat.levels" = cat.levels
  ))
}else if(scaler=="standard"){
  return(list(
    "data.tensor" = data.tensor,
    "na.loc" = na.loc,
    "colmean" = colmean, "colsd" = colsd,
    "original.names" = original.names,
    "ordered.names" = ordered.names,
    "ordered.types" = ordered.types,
    "num" = num,
    "bin" = bin,
    "multi" = multi,
    "num.idx" = num.idx,
    "bin.idx" = bin.idx,
    "multi.idx" = multi.idx,
    "cat.names" = cat.names,
    "cat.levels" = cat.levels
  ))
}else{
  return(list(
    "data.tensor" = data.tensor,
    "na.loc" = na.loc,
    "original.names" = original.names,
    "ordered.names" = ordered.names,
    "ordered.types" = ordered.types,
    "num" = num,
    "bin" = bin,
    "multi" = multi,
    "num.idx" = num.idx,
    "bin.idx" = bin.idx,
    "multi.idx" = multi.idx,
    "cat.names" = cat.names,
    "cat.levels" = cat.levels
  ))
}


}


#' This function is used to return the type(numeric,binary,multiclass) of each feature
#' @param  data A data frame, tibble, or data table.
#' @export
feature_type <- function(data) {
  binary <- NULL
  multiclass <- NULL
  features <- colnames(data)
  types <- rep(NA, length(features))
  names(types) <- features

  for (var in features) {
    if (nlevels(data[[var]]) == 0) {
      types[var] <- "numeric"
    } else if (nlevels(data[[var]]) == 2) {
      types[var] <- "binary"
    } else {
      types[var] <- "multiclass"
    }
  }
  return(types)
}


# Impute the missing values of a vector with the mode (majority class) of observed values
imp.mode <- function(vec, na.idx = NULL) {
  # @param vec A vector of numeric values (ideally integer type) or factor
  # @param na.idx Indices of missing values
  if (is.null(na.idx)) {
    na.idx <- which(is.na(vec))
  }

  if (length(na.idx) == 0) {
    stop("This vector contains no missing value.")
  }


  unique.values <- unique(na.omit(vec))
  tab <- tabulate(match(vec, unique.values))
  var.mode <- unique.values[tab == max(tab)]

  n.na <- length(na.idx)
  if (length(var.mode) == 1) {
    # if mode is unique
    vec[na.idx] <- rep(var.mode, n.na)
  } else {
    # if mode is not unique, impute with randomly sampled modes
    vec[na.idx] <- sample(var.mode, size = n.na, replace = TRUE)
  }
  vec
}
