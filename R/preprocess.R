#' Pre-process data before imputation: scale numeric and one-hot categorical
#' @importFrom dplyr select_if
#' @importFrom purrr negate
#' @importFrom torch torch_tensor torch_cat
preprocess <- function(data) {
  Types <- feature_type(data)
  # Types
  original.names <- names(Types)

  num <- original.names[Types %in% "numeric"]
  bin <- original.names[Types %in% "binary"]
  multi <- original.names[Types %in% "multiclass"]
  ordered.names <- c(num, bin, multi)
  ordered.types <- Types[ordered.names]

  # ordered data according to numeric>binary>multiclass
  ordered.data <- data[ordered.names]

  na.loc <- is.na(ordered.data)


  # scale numeric data (pre-impute with median)
  num.obj <- minmax_scaler(data[num])
  # cont.obj<-minmax_scaler(data %>% dplyr::select_if(is.numeric))

  num.mat <- num.obj$minmax.mat
  num.tensor <- torch::torch_tensor(num.mat)
  col.min <- num.obj$colmin
  col.max <- num.obj$colmax


  # one-hot categorical data
  # one-hot encoding (vs embeddings)
  cat.mat <- data[c(bin, multi)]
  # cat.mat<-data %>% dplyr::select_if(purrr::negate(is.numeric))
  cat.names <- c(bin, multi)

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


  num.idx <- NULL
  bin.idx <- vector("list", length = length(bin))
  names(bin.idx) <- bin

  multi.idx <- vector("list", length = length(multi))
  names(multi.idx) <- multi

  if (!is.null(num)) {
    num.idx <- 1:length(num)
  }

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
  onehot.tensor <- torch::torch_cat(list(num.tensor, cat.tensor), dim = 2)


  return(list(
    "onehot.tensor" = onehot.tensor,
    "na.loc" = na.loc,
    "col.min" = col.min, "col.max" = col.max,
    "original.names" = original.names,
    "ordered.names" = ordered.names,
    "ordered.types" = ordered.types,
    "num" = num,
    "bin" = bin,
    "multi" = multi,
    "cat.names" = cat.names,
    "num.idx" = num.idx,
    "bin.idx" = bin.idx,
    "multi.idx" = multi.idx,
    "cat.levels" = cat.levels
  ))
}


#' This function is used to return the type(numeric,binary,multiclass) of each feature
#' @param  data A data frame
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