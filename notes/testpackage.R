devtools::load_all()
devtools::document()

############
set.seed(2023)
iris.binary<-iris
iris.binary$Gender<-sample(c("F","M"),size=nrow(iris),replace = T)
iris.binary$Gender<-as.factor(iris.binary$Gender)
str(iris.binary)



withNA.df<-createNA(data = iris.binary,p = 0.2)


newdata <- createNA(data = iris.binary,p = 0.2)
colSums(is.na(newdata))


dim(data)
data=withNA.df
m = 5
epochs = 5
batch.size = 150
subsample = 1
shuffle = TRUE
input.dropout = 0
hidden.dropout = 0
optimizer = "adamW"
learning.rate = 0.0001
weight.decay = 0.002
momentum = 0
eps = 1e-07
encoder.structure = c(256, 256, 256)
decoder.structure = c(256, 256, 256)
act = "identity"
init.weight="xavier.midas"
scaler= "none"
verbose = TRUE
print.every.n = 1
save.model = TRUE
loss.na.scale=T


pmm.type = "auto"

pmm.type = NULL

pmm.k = 5
pmm.link = "prob"
pmm.save.vars = NULL
save.model = TRUE
path = file.path(tempdir(),"midaemodel.pt")


####################
my_list <- list(a = 1, b = 2, c = 3, d = 4)

# create a named vector
my_vector <- c(a = 2, b = 3, c = 4, d = 5,e=0.8)

result <- mapply(`*`, my_list, my_vector)
result



imputed.data <- midae(data = withNA.df, m = 5, epochs = 50,
                      batch.size = 30,
                      sampling = 1,
                      shuffle = TRUE,
                      input.dropout = 0.2,
                      hidden.dropout = 0.5,
                      optimizer = "adamW",
                      learning.rate = 0.0001,
                      weight.decay = 0.002,
                      momentum = 0,
                      eps = 1e-07,
                      encoder.structure = c(256, 256, 256),
                      decoder.structure = c(256, 256, 256),
                      act = "elu",
                      init.weight="xavier.midas",
                      scaler= "standard",
                      loss.na.scale = T,
                      verbose = TRUE,
                      print.every.n = 1,
                      save.model = FALSE,
                     path = file.path(tempdir(),"midaemodel.pt"))

imputed.data

show_var(imputation.list = imputed.data,var.name = "Petal.Width", original.data = withNA.df)


imputed.data0 <- midae0(data = withNA.df, m = 5, epochs = 10, scaler = "none", path = file.path(tempdir(),"midaemodel.pt"))

imputed.data0


show_var(imputation.list = imputed.data,var.name = "Sepal.Length", original.data = withNA.df)


df1<-minmax_scaler(data=withNA.df[,1:4])


rev_minmax_scaler(scaled.data=df1$minmax.mat,num.names = names(df1$minmax.mat),colmin=df1$colmin, colmax=df1$colmax)

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



rev_minmax_scaler <- function(scaled.data, num.names, colmin, colmax) {
  for (var in num.names) {
    scaled.data[, var] <- scaled.data[, var] * (colmax[var] - colmin[var]) + colmin[var]
  }
  scaled.data
}
