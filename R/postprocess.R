#' postprocess data
#' @param output.data raw output data from autoencoders
#' @param pre.obj preprocess data object
#' @export
postprocess <- function(output.data, pre.obj, scaler) {


  if (length(pre.obj$num) >= 1) {
    imp.m <- as_array(output.data[, 1:length(pre.obj$num)])
  } else {
    imp.m <- NULL
  }


  if (length(pre.obj$logi) >= 1) {
    for (var in pre.obj$logi) {
      idx <- pre.obj$logi.idx[[var]]
      transform_fn <- nn_sigmoid()
      preds<-transform_fn(output.data[, idx] )
      preds<-ifelse(preds >= 0.5,TRUE, FALSE)
      imp.m <- cbind(imp.m, preds)
    }
  }

  if (length(pre.obj$bin) >= 1) {
    for (var in pre.obj$bin) {
      idx <- pre.obj$bin.idx[[var]]
      transform_fn <- nn_sigmoid()
      preds<-transform_fn(output.data[, idx] )
      preds<-ifelse(preds >= 0.5,2, 1)
      imp.m <- cbind(imp.m, preds)
    }
  }


  if (length(pre.obj$multi) >= 1) {
    for (var in pre.obj$multi) {
      idx <- pre.obj$multi.idx[[var]]
      transform_fn <- nn_softmax(dim = 2)
      preds<- transform_fn(output.data[, idx])
      imp.m <- cbind(imp.m, as_array(torch_argmax(preds, dim = 2)))
    }
  }

  imp.df <- as.data.frame(imp.m)
  colnames(imp.df) <- pre.obj$ordered.names

  #recover the labels
  if(!is.null(pre.obj$fac.levels)){
    fac.names<-names(pre.obj$fac.levels)
    for (var in fac.names) {
      imp.df[[var]] <- pre.obj$fac.levels[[var]][imp.df[[var]]]
    }

  }

  # unscaled numeric features (note: pre.obj$num includes pre.obj$int)
  if (scaler != "none") {
    if (scaler == "minmax") {
      imp.df <- rev_minmax_scaler(scaled.data = imp.df, num.names = pre.obj$num, colmin = pre.obj$colmin, colmax = pre.obj$colmax)
    } else if (scaler == "decile") {
      imp.df <- rev_decile_scaler(scaled.data = imp.df, num.names = pre.obj$num, decile1 = pre.obj$decile1, decile9 = pre.obj$decile9)
    } else if (scaler == "standard") {
      imp.df <- rev_standard_scaler(scaled.data = imp.df, num.names = pre.obj$num, colmean = pre.obj$colmean, colsd = pre.obj$colsd)
    }
  }



  #recover data types
  imp.df<-imp.df %>%
    dplyr::mutate_at(vars(pre.obj$logi),as.logical)%>%
    dplyr::mutate_if(is.character, as.factor)%>%
    dplyr::mutate_at(vars(pre.obj$int),round)%>%
    dplyr::mutate_at(vars(pre.obj$int),as.integer)

 return(imp.df)

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
