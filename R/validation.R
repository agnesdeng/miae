# Checking for pmm.type constraints
check_pmm <- function(pmm.type, subsample, input.dropout, hidden.dropout, Nrow, sorted.naSums, sorted.types, pmm.k) {
  if (!is.null(pmm.type)) {
    # type
    if (!pmm.type %in% c(0, 1, 2, "auto")) {
      stop("The specified pmm.type is incorrect. It must be one of the following types: NULL,1,2,\"auto\".")
    }

    # dropout

    if ((pmm.type == 1 | pmm.type == 2) & (input.dropout == 0 & hidden.dropout == 0)) {
      stop("MIDAE with PMM type 1 or type 2 requires applying dropout to the input and hidden layers. Please set \`input.dropout > 0\` and \`hidden.dropout > 0\`.")
    }

    if (pmm.type == 0 & subsample < 1) {
      stop("MIDAE with PMM type0 requires using the whole dataset. Subsample ratios < 1 are not allowed.")
    }

    if (pmm.type == 0 & subsample == 1) {
      if (input.dropout != 0 | hidden.dropout != 0) {
        stop("MIDAE with PMM type0 does not allow dropout to apply to the input and hidden layers. Users must set \`input.dropout = 0\` and \`hidden.dropout = 0\`.")
      }
    }

    # if pmm.type=0,1 or 2, all variables need to perform PMM
    if (any(Nrow - sorted.naSums < pmm.k) && pmm.type != "auto") {
      maxNA <- max(sorted.naSums)
      minObs <- Nrow - maxNA
      s1 <- paste("In this dataset, the minimum number of observed values in a variable is ", minObs, ".", sep = "")
      s2 <- paste("However, pmm.k=", pmm.k, ".", sep = "")
      if (minObs == 1) {
        s3 <- paste("Please set pmm.k = 1 .")
      } else {
        s3 <- paste("Please set the value of pmm.k less than or equal to ", minObs, ".", sep = "")
      }
      stop(paste(s1, s2, s3, sep = "\n"))
    }
    # if pmm.type="auto", only numeric variables need to perform PMM
    if (pmm.type == "auto") {
      idx <- which(Nrow - sorted.naSums < pmm.k & sorted.types == "numeric")
      if (length(idx) > 0) {
        maxNA <- max(sorted.naSums[idx])
        minObs <- Nrow - maxNA
        s1 <- paste("In this dataset, the minimum number of observed values in a numeric variable is ", minObs, ".", sep = "")
        s2 <- paste("When pmm.type = \"auto\", type 2 PMM would apply to numeric variables. However, pmm.k=", pmm.k, ".", sep = "")
        if (minObs == 1) {
          s3 <- paste("Please set pmm.k = 1 .")
        } else {
          s3 <- paste("Please set the value of pmm.k less than or equal to ", minObs, ".", sep = "")
        }
        stop(paste(s1, s2, s3, sep = "\n"))
      }
    }
  }
}



# Classify the type of each variable in a dataset
feature_type <- function(data) {
  # @param data A data.frame or a data.table
  # @return The type (numeric/integer/binary/multiclass) of each variable in a dataset

  Types <- sapply(data, class)

  if (any(Types == "character")) {
    stop("Data contains variables of character type. Please change them into factor.")
  }

  # ordinal.idx<-grep("ordered",Types)
  ord.fac <- names(Filter(is.ordered, data))
  if (length(ord.fac) > 0) {
    Types[ord.fac] <- "factor"
  }

  factor.vars <- which(Types == "factor")
  for (fac in factor.vars) {
    if (length(levels(data[[fac]])) == 2) {
      Types[fac] <- "binary"
    } else {
      Types[fac] <- "multiclass"
    }
  }

  return(Types)
}


#' Sort data by increasing number of missing values
#' @import data.table
#' @keywords internal
sortNA <- function(data) {
  # @param data A data table (with missing values NA's)
  # @return A list whose first component is the sorted data, second component is the sorted indices and third component is the sorted variable names according to the amount of missingness.

  Names <- colnames(data)
  na.loc <- is.na(data)
  sorted.idx <- order(colSums(na.loc))
  sorted.names <- Names[sorted.idx]

  if (is.data.table(data)) {
    # data.table
    # sorted.data <- data[, ..sorted.names]
    sorted.data <- data[, sorted.names, with = FALSE]
  } else {
    # data.frame
    sorted.data <- data[, sorted.names]
    sorted.data <- as.data.table(sorted.data)
  }

  # setcolorder(data,sorted.names)
  # will change data
  return(list("sorted.dt" = sorted.data, "sorted.idx" = sorted.idx, "sorted.names" = sorted.names))
}
