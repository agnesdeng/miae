
#' scale a vector using minmax
#' @param x a vector
#' @export
minmax <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}


#' scale a dataset and return a scaled dataframe, the colmin and colmax of each column
#' @param data A data frame or tibble
#' @importFrom dplyr mutate_all
#' @importFrom stats median
#' @export
minmax_scaler <- function(data) {
  pre <- dplyr::mutate_all(data, ~ ifelse(is.na(.), median(., na.rm = TRUE), .))
  colmin <- apply(pre, 2, min)
  colmax <- apply(pre, 2, max)
  minmax.obj <- NULL
  minmax.obj$minmax.mat <- apply(pre, 2, minmax)
  minmax.obj$colmin <- colmin
  minmax.obj$colmax <- colmax
  return(minmax.obj)
}




#' This function back-transform data to an output as data matrix
#' @param scaled.data A matrix or array with scaled numeric data
#' @param num.names the names of numeric features
#' @param colmin A vector that contains the minimum of each column
#' @param colmax A vector that contains the maximum of each column
#' @export
rev_minmax_scaler <- function(scaled.data, num.names, colmin, colmax) {
  for (var in num.names) {
    scaled.data[, var] <- scaled.data[, var] * (colmax[var] - colmin[var]) + colmin[var]
  }
  scaled.data
}
