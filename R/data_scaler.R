
#'scale a vector using standardize
#'@param x a vector
#'@export
standard<-function(x){
  (x-mean(x))/sd(x)
}

#' scale a dataset using stardardize and return a scaled dataframe, the colmean and colsd of each column
#' @param data A data frame or tibble
#' @importFrom dplyr mutate_all
#' @importFrom stats sd
#' @export
standard_scaler <- function(data) {
  pre <- dplyr::mutate_all(data, ~ ifelse(is.na(.), mean(., na.rm = TRUE), .))
  colmean <- apply(pre, 2, mean)
  colsd <- apply(pre, 2, sd)
  standard.obj <- NULL
  standard.obj$standard.mat <- apply(pre, 2, standard)
  standard.obj$colmean <- colmean
  standard.obj$colsd <- colsd
  return(standard.obj)
}



#' This function back-transform standardized data to an output as data matrix
#' @param scaled.data A matrix or array with scaled numeric data
#' @param num.names the names of numeric features
#' @param colmean A vector that contains the mean of each column
#' @param colsd A vector that contains the standard deviation of each column
#' @export
rev_standard_scaler <- function(scaled.data, num.names, colmean, colsd) {
  for (var in num.names) {
    scaled.data[, var] <- scaled.data[, var] * colsd[var] + colmean[var]
  }
  scaled.data
}


#' scale a vector using minmax
#' @param x a vector
#' @export
minmax <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}


#' scale a dataset using minmax and return a scaled dataframe, the colmin and colmax of each column
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




#' This function back-transform minmax-scaled data to an output as data matrix
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




#' scale a vector using decile
#' @param x a vector
#' @export
decile <- function(x) {
  (x - quantile(x,  probs = c(0.1)) / quantile(x,  probs = c(0.9)) - quantile(x,  probs = c(0.1)))
}



#' scale a dataset using decile and return a scaled dataframe, the 1st decile and the 9th decile of each column
#' @param data A data frame or tibble
#' @importFrom dplyr mutate_all
#' @importFrom stats median
#' @export
decile_scaler <- function(data) {
  pre <- dplyr::mutate_all(data, ~ ifelse(is.na(.), median(., na.rm = TRUE), .))
  decile1 <- apply(pre, 2, quantile, probs = 0.1)
  decile9 <- apply(pre, 2, quantile, probs = 0.9)
  decile.obj <- NULL
  decile.obj$decile.mat <- apply(pre, 2, decile)
  decile.obj$decile1 <- decile1
  decile.obj$decile9 <- decile9
  return(decile.obj)
}




#' This function back-transform decile-scaled data to an output as data matrix
#' @param scaled.data A matrix or array with scaled numeric data
#' @param num.names the names of numeric features
#' @param decile1 A vector that contains the the 1st decile (10th percentile) of each column
#' @param decile9 A vector that contains the the 9th decile (90th percentile) of each column
#' @export
rev_decile_scaler <- function(scaled.data, num.names, decile1, decile9) {
  for (var in num.names) {
    scaled.data[, var] <- scaled.data[, var] * (decile9[var] - decile1[var]) + decile1[var]
  }
  scaled.data
}
