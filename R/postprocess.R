#' postprocess data
#' @param output.data raw output data from autoencoders
#' @param pre.obj preprocess data object
#' @export
postprocess <- function(output.data, pre.obj, scaler) {
  # reverse onehot categorical features
  if (length(pre.obj$cat.names) >= 1) {
    imp.data <- rev_onehot(onehot.data = output.data, pre.obj = pre.obj)
  } else {
    # imp.data <- output.data
    imp.data <- as_array(output.data[, 1:length(pre.obj$num)])
    colnames(imp.data) <- pre.obj$num
  }


  # unscaled numeric features
  if (length(pre.obj$num) >= 1 & scaler != "none") {
    if (scaler == "minmax") {
      imp.data <- rev_minmax_scaler(scaled.data = imp.data, num.names = pre.obj$num, colmin = pre.obj$colmin, colmax = pre.obj$colmax)
    } else if (scaler == "decile") {
      imp.data <- rev_decile_scaler(scaled.data = imp.data, num.names = pre.obj$num, decile1 = pre.obj$decile1, decile9 = pre.obj$decile9)
    } else if (scaler == "standard") {
      imp.data <- rev_standard_scaler(scaled.data = imp.data, num.names = pre.obj$num, colmean = pre.obj$colmean, colsd = pre.obj$colsd)
    }
  }

  imp.data <- as.data.frame(imp.data)

  return(imp.data)
}




#' reverse onehot data
#' @param onehot.data data with onehot features
#' @param pre.obj preprocess data object
#' @importFrom torch nn_sigmoid nn_softmax torch_argmax
#' @export
rev_onehot <- function(onehot.data, pre.obj) {
  if (length(pre.obj$num) >= 1) {
    imp.m <- as_array(onehot.data[, 1:length(pre.obj$num)])
  } else {
    imp.m <- NULL
  }


  if (length(pre.obj$bin) >= 1) {
    for (var in pre.obj$bin) {
      idx <- pre.obj$bin.idx[[var]]
      transform_fn <- nn_sigmoid()
      onehot.data[, idx] <- transform_fn(onehot.data[, idx])
      imp.m <- cbind(imp.m, as_array(torch_argmax(onehot.data[, idx], dim = 2)))
    }
  }

  if (length(pre.obj$multi) >= 1) {
    for (var in pre.obj$multi) {
      idx <- pre.obj$multi.idx[[var]]
      transform_fn <- nn_softmax(dim = 2)
      onehot.data[, idx] <- transform_fn(onehot.data[, idx])
      imp.m <- cbind(imp.m, as_array(torch_argmax(onehot.data[, idx], dim = 2)))
    }
  }

  imp.df <- as.data.frame(imp.m)
  colnames(imp.df) <- pre.obj$ordered.names


  for (var in pre.obj$cat.names) {
    imp.df[[var]] <- pre.obj$cat.levels[[var]][imp.df[[var]]]
  }

  return(imp.df)
}
