#Pre-process data before imputation: scale numeric and one-hot categorical
preprocess <- function(data, scaler, lower, upper, categorical.encoding,initial.imp) {
  Types <- feature_type(data)

  # Types
  original.names <- names(Types)

  int <- original.names[Types %in% "integer"]
  num <- original.names[Types %in% c("numeric", "integer")]
  logi <- original.names[Types %in% "logical"]
  bin <- original.names[Types %in% "binary"]
  multi <- original.names[Types %in% "multiclass"]


  ordered.names <- c(num, logi, bin, multi)
  ordered.types <- Types[ordered.names]

  # ordered data according to numeric>binary>multiclass
  if (data.table::is.data.table(data)) {
    ordered.data <- data[, ordered.names, with = FALSE]
  } else {
    ordered.data <- data[,ordered.names, drop = FALSE]
  }

  na.loc <- is.na(ordered.data)


  # test<-original.names[Types %in% "test"]

  # ordered.names <- c(num, bin, multi)
  # ordered.types <- Types[ordered.names]

  # ordered data according to numeric->logical->binary->multiclass
  # if (data.table::is.data.table(data)) {
  # ordered.data <- data[, ordered.names, with = FALSE]
  # } else {
  # ordered.data <- data[ordered.names]
  # }

  # na.loc <- is.na(ordered.data)


  data.tensor <- vector("list", 5)
  names(data.tensor) <- c("num.tensor", "logi.tensor", "bin.tensor", "multi.tensor", "onehot.tensor")

  # if numeric feature exists
  if (length(num) >= 1) {
    # scaler will initial impute numeric values with median
    if (scaler != "none") {
      if (scaler == "minmax") {
        if (is.data.table(data)) {
          num.obj <- minmax_scaler(data[, num, with = FALSE],initial.imp = initial.imp)
        } else {
          num.obj <- minmax_scaler(data[, num, drop = FALSE],initial.imp = initial.imp)
        }

        num.mat <- num.obj$minmax.mat

        colmin <- num.obj$colmin
        colmax <- num.obj$colmax
      } else if (scaler == "robust") {
        if (is.data.table(data)) {
          num.obj <- robust_scaler(data[, num, with = FALSE],initial.imp = initial.imp, lower = lower, upper = upper)
        } else {
          num.obj <- robust_scaler(data[, num, drop = FALSE],initial.imp = initial.imp, lower = lower, upper = upper)
        }

        num.mat <- num.obj$robust.mat

        robust.median <- num.obj$robust.median
        robust.lower <- num.obj$robust.lower
        robust.upper <- num.obj$robust.upper
      } else if (scaler == "standard") {
        if (is.data.table(data)) {
          num.obj <- standard_scaler(data[, num, with = FALSE],initial.imp = initial.imp)
        } else {
          num.obj <- standard_scaler(data[, num, drop = FALSE],initial.imp = initial.imp)
        }

        num.mat <- num.obj$standard.mat

        colmean <- num.obj$colmean
        colsd <- num.obj$colsd
      } else if (scaler == "decile") {
        if (is.data.table(data)) {
          num.obj <- decile_scaler(data[, num, with = FALSE],initial.imp = initial.imp)
        } else {
          num.obj <- decile_scaler(data[, num, drop = FALSE],initial.imp = initial.imp)
        }

        num.mat <- num.obj$decile.mat

        decile1 <- num.obj$decile1
        decile9 <- num.obj$decile9
      }
    } else {
      # don't scale numeric data:  scaler="none"
      if(initial.imp=="mean"){
        num.tibble <- dplyr::mutate_all(data[, num, drop = FALSE], ~ ifelse(is.na(.), mean(., na.rm = TRUE), .))
      }else if(initial.imp=="median"){
        num.tibble <- dplyr::mutate_all(data[, num, drop = FALSE], ~ ifelse(is.na(.), median(., na.rm = TRUE), .))
      }else if(initial.imp=="sample"){
        num.tibble <- dplyr::mutate_all(data[, num, drop = FALSE], ~ ifelse(is.na(.), samples(.),.))
      }


      num.mat <- as.matrix(num.tibble)
    }


    data.tensor$num.tensor <- torch_tensor(num.mat)

    num.idx <- vector("list", length = length(num))
    names(num.idx) <- num

    # if (!is.null(num)) {
    # for (i in seq_along(num)) {
    # num.idx[i] <- i
    # }
    # }

    for (i in seq_along(num)) {
      num.idx[[i]] <- i
    }
  } else {
    # no numeric data
    num.idx <- NULL
    colmin <- NULL
    colmax <- NULL
    colmean <- NULL
    colsd <- NULL
    decile1 <- NULL
    decile9 <- NULL
  }






  ## categorical data
  cat.names <- c(logi, bin, multi)

  if (length(cat.names) < 1) {
    # only numeric

    logi.idx <- NULL
    bin.idx <- NULL
    multi.idx <- NULL
    cardinalities <- NULL
    embedding.dim <- NULL
    fac.levels <- NULL
  } else {
    # levels for each categorical variables
    fac.names <- c(bin, multi)
    if (length(fac.names) > 0) {
      fac.levels <- vector("list", length(fac.names))
      names(fac.levels) <- fac.names

      for (var in fac.names) {
        fac.levels[[var]] <- levels(data[[var]])
      }
    } else {
      fac.levels <- NULL
    }




    if (length(logi) >= 1) {
      # logical variables exist:  -> 0,1
      if (data.table::is.data.table(data)) {
        logi.mat <- data[, logi, with = FALSE]
      } else {
        logi.mat <- data[, logi, drop = FALSE]
      }



      logi.naSums <- colSums(is.na(logi.mat))
      logi.naidx <- which(logi.naSums != 0)


      if (length(logi.naidx) != 0) {
        # initial impute with mode
        logi.navars <- logi[logi.naidx]
        for (var in logi.navars) {
          na.idx <- which(is.na(logi.mat[[var]]))
          # Na.idx[[var]] <- na.idx
          # Impute the missing values of a vector with the mode (majority class) of observed values
          logi.mat[[var]] <- imp.mode(vec = logi.mat[[var]], na.idx = na.idx)
        }
      }

      logi.mat <- logi.mat %>%
        dplyr::mutate_all(.funs = as.integer) %>%
        as.matrix()

      # float type

      data.tensor$logi.tensor <- torch_tensor(logi.mat, dtype = torch_float())
      # data.tensor <- torch::torch_cat(list(data.tensor, logi.tensor), dim = 2)


      logi.idx <- vector("list", length = length(logi))
      names(logi.idx) <- logi

      for (i in seq_along(logi)) {
        logi.idx[[i]] <- length(num.idx) + i
      }
    } else {
      logi.idx <- NULL
      cardinalities <- NULL
      embedding.dim <- NULL
    }



    if (length(bin) >= 1) {
      # binary variables exist: -> 0,1
      if (data.table::is.data.table(data)) {
        bin.mat <- data[, bin, with = FALSE]
      } else {
        bin.mat <- data[, bin, drop = FALSE]
      }

      bin.naSums <- colSums(is.na(bin.mat))
      bin.naidx <- which(bin.naSums != 0)


      if (length(bin.naidx) != 0) {
        # initial impute with mode
        bin.navars <- bin[bin.naidx]
        for (var in bin.navars) {
          na.idx <- which(is.na(bin.mat[[var]]))
          # Na.idx[[var]] <- na.idx
          # Impute the missing values of a vector with the mode (majority class) of observed values
          bin.mat[[var]] <- imp.mode(vec = bin.mat[[var]], na.idx = na.idx)
        }
      }

      bin.mat <- bin.mat %>%
        dplyr::mutate_all(.funs = as.integer) %>%
        dplyr::mutate_all(~ . - 1) %>%
        as.matrix()
      # float type

      data.tensor$bin.tensor <- torch_tensor(bin.mat)
      # data.tensor <- torch::torch_cat(list(data.tensor, bin.tensor), dim = 2)



      bin.idx <- vector("list", length = length(bin))
      names(bin.idx) <- bin


      for (i in seq_along(bin)) {
        bin.idx[[i]] <- length(num.idx) + length(logi.idx) + i
      }
    } else {
      bin.idx <- NULL
      cardinalities <- NULL
      embedding.dim <- NULL
    }




    if (length(multi) >= 1) {
      # multi-class variables exist: 1,2,3,4,...
      if (data.table::is.data.table(data)) {
        multi.mat <- data[, multi, with = FALSE]
      } else {
        multi.mat <- data[, multi, drop = FALSE]
      }

      multi.naSums <- colSums(is.na(multi.mat))
      multi.naidx <- which(multi.naSums != 0)


      if (length(multi.naidx) != 0) {
        # initial impute with mode
        multi.navars <- multi[multi.naidx]
        for (var in multi.navars) {
          na.idx <- which(is.na(multi.mat[[var]]))
          # Na.idx[[var]] <- na.idx
          # Impute the missing values of a vector with the mode (majority class) of observed values
          multi.mat[[var]] <- imp.mode(vec = multi.mat[[var]], na.idx = na.idx)
        }
      }

      # cardinalities: the number of levels
      cardinalities <- multi.mat %>%
        dplyr::summarise(dplyr::across(.cols = dplyr::everything(), .fns = nlevels)) %>%
        unlist()

      multi.idx <- vector("list", length = length(multi))
      names(multi.idx) <- multi


      multi.start <- length(num.idx) + length(logi.idx) + length(bin.idx)
      for (i in seq_along(multi)) {
        multi.idx[[i]] <- multi.start + 1:cardinalities[i]
        multi.start <- multi.idx[[i]][length(multi.idx[[i]])]
      }


      if (categorical.encoding == "embeddings") {
        multi.mat <- multi.mat %>%
          dplyr::mutate_all(.funs = as.integer) %>%
          as.matrix()


        # Note:  long type can't use torch_cat()

        data.tensor$multi.tensor <- torch_tensor(multi.mat)



        # embedding.dim<-ifelse(cardinalities<5,cardinalities +2,
        # ifelse(cardinalities<7, cardinalities+1,
        # ifelse(cardinalities<9, cardinalities,
        # ifelse(cardinalities<16, 8, ceiling(cardinalities/2)))))

        embedding.dim <- ceiling(cardinalities^(1 / 4))




        names(embedding.dim) <- names(cardinalities)
      } else if (categorical.encoding == "onehot") {
        embedding.dim <- NULL


        onehot.list <- vector("list", length = length(multi))
        names(onehot.list) <- multi

        for (var in multi) {

          onehot.list[[var]] <- nnf_one_hot(torch_tensor(as.integer(multi.mat[[var]])), num_classes=cardinalities[var])
        }

        # combine numeric data with one-hot encoded categorical variables
        data.tensor$onehot.tensor <- torch_cat(onehot.list, dim = 2)


        multi.mat <- multi.mat %>%
          dplyr::mutate_all(.funs = as.integer) %>%
          as.matrix()


        # Note:  long type can't use torch_cat()

        data.tensor$multi.tensor <- torch_tensor(multi.mat)

        # head(onehot.list$DMARACER)
        # head(multi.mat$DMARACER)
      } else {
        stop(cat('categorical.encoding can only be either "embeddings" or "onehot".\n'))
      }
    } else {
      cardinalities <- NULL
      embedding.dim <- NULL
      multi.idx <- NULL
    }
  }




  if (scaler == "minmax") {
    return(list(
      "data.tensor" = data.tensor,
      "na.loc" = na.loc,
      "colmin" = colmin, "colmax" = colmax,
      "original.names" = original.names,
      "ordered.names" = ordered.names,
      "ordered.types" = ordered.types,
      "int" = int,
      "num" = num,
      "logi" = logi,
      "bin" = bin,
      "multi" = multi,
      "num.idx" = num.idx,
      "logi.idx" = logi.idx,
      "bin.idx" = bin.idx,
      "multi.idx" = multi.idx,
      "cardinalities" = cardinalities,
      "embedding.dim" = embedding.dim,
      "fac.levels" = fac.levels
    ))
  } else if (scaler == "robust") {
    return(list(
      "data.tensor" = data.tensor,
      "na.loc" = na.loc,
      "robust.lower" = robust.lower, "robust.upper" = robust.upper, "robust.median" = robust.median,
      "original.names" = original.names,
      "ordered.names" = ordered.names,
      "ordered.types" = ordered.types,
      "int" = int,
      "num" = num,
      "logi" = logi,
      "bin" = bin,
      "multi" = multi,
      "num.idx" = num.idx,
      "logi.idx" = logi.idx,
      "bin.idx" = bin.idx,
      "multi.idx" = multi.idx,
      "cardinalities" = cardinalities,
      "embedding.dim" = embedding.dim,
      "fac.levels" = fac.levels
    ))
  } else if (scaler == "standard") {
    return(list(
      "data.tensor" = data.tensor,
      "na.loc" = na.loc,
      "colmean" = colmean, "colsd" = colsd,
      "original.names" = original.names,
      "ordered.names" = ordered.names,
      "ordered.types" = ordered.types,
      "int" = int,
      "num" = num,
      "logi" = logi,
      "bin" = bin,
      "multi" = multi,
      "num.idx" = num.idx,
      "logi.idx" = logi.idx,
      "bin.idx" = bin.idx,
      "multi.idx" = multi.idx,
      "cardinalities" = cardinalities,
      "embedding.dim" = embedding.dim,
      "fac.levels" = fac.levels
    ))
  } else if (scaler == "decile") {
    return(list(
      "data.tensor" = data.tensor,
      "na.loc" = na.loc,
      "decile1" = decile1, "decile9" = decile9,
      "original.names" = original.names,
      "ordered.names" = ordered.names,
      "ordered.types" = ordered.types,
      "int" = int,
      "num" = num,
      "logi" = logi,
      "bin" = bin,
      "multi" = multi,
      "num.idx" = num.idx,
      "logi.idx" = logi.idx,
      "bin.idx" = bin.idx,
      "multi.idx" = multi.idx,
      "cardinalities" = cardinalities,
      "embedding.dim" = embedding.dim,
      "fac.levels" = fac.levels
    ))
  } else {
    return(list(
      "data.tensor" = data.tensor,
      "na.loc" = na.loc,
      "original.names" = original.names,
      "ordered.names" = ordered.names,
      "ordered.types" = ordered.types,
      "int" = int,
      "num" = num,
      "logi" = logi,
      "bin" = bin,
      "multi" = multi,
      "num.idx" = num.idx,
      "logi.idx" = logi.idx,
      "bin.idx" = bin.idx,
      "multi.idx" = multi.idx,
      "cardinalities" = cardinalities,
      "embedding.dim" = embedding.dim,
      "fac.levels" = fac.levels
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
    if (typeof(data[[var]]) == "double") {
      types[var] <- "numeric"
    } else if (typeof(data[[var]]) == "logical") {
      types[var] <- "logical"
    } else {
      # integer
      if (nlevels(data[[var]]) == 0) {
        types[var] <- "integer"
      } else if (nlevels(data[[var]]) == 2) {
        types[var] <- "binary"
      } else if (nlevels(data[[var]]) > 2) {
        types[var] <- "multiclass"
      }
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
