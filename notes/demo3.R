# library(dplyr)
library(palmerpenguins)


library(devtools)
devtools::document()
devtools::load_all()



library(palmerpenguins)

n <- nrow(penguins)
idx <- sample(1:n, size = round(0.7 * n), replace = FALSE)
train.data <- penguins[idx, ]
test.data <- penguins[-idx, ]



imputed.data <- midae(data = train.data, m = 5, epochs = 5, path = "C:/Users/agnes/Desktop/torch/midaemodel.pt")


imputed.newdata <- impute_new(path = "C:/Users/agnes/Desktop/torch/midaemodel.pt", newdata = test.data, m = 5)
imputed.newdata



imputed.data <- mivae(data = train.data, m = 5, epochs = 5, path = "C:/Users/agnes/Desktop/torch/mivaemodel.pt")


imputed.newdata <- impute_new(path = "C:/Users/agnes/Desktop/torch/mivaemodel.pt", newdata = test.data, m = 5)
imputed.newdata


# devtools::build()
iris.dt <- data.table(iris)
iris.dt
withNA.dt <- createNA(data = iris.dt, p = 0.3)
withNA.dt

data <- withNA.dt
m <- 5
epochs <- 5
batch.size <- 50
input.dropout <- 0.5
latent.dropout <- 0.5
hidden.dropout <- 0.5
optimizer <- "adam"
learning.rate <- 0.001
momentum <- 0
encoder.structure <- c(128, 64, 32)
latent.dim <- 16
decoder.structure <- c(32, 64, 128)
verbose <- TRUE
print.every.n <- 1
directory <- "C:/Users/agnes/Desktop/torch"



imputed.data <- midae(data = withNA.dt, m = 5, epochs = 5, directory = "C:/Users/agnes/Desktop/torch")
imputed.data <- mivae(data = withNA.dt, m = 5, epochs = 5, directory = "C:/Users/agnes/Desktop/torch")

withNA.df <- createNA(data = iris, p = 0.2)
imputed.data <- midae(data = withNA.df, m = 5, epochs = 5, directory = "C:/Users/agnes/Desktop/torch")
imputed.data <- mivae(data = withNA.df, m = 5, epochs = 5, directory = "C:/Users/agnes/Desktop/torch")

imputed.data <- midae(data = withNA.df, m = 5, epochs = 5)

set.seed(2022)
library(mixgb)

midae.imputed0 <- midae(data = nhanes3_newborn, m = 5, epochs = 10, directory = "C:/Users/agnes/Desktop/torch")
# midae.imputed
colSums(is.na(nhanes3_newborn))
show_var(imputation.list = midae.imputed0, var.name = "BMPHEAD", original.data = nhanes3_newborn)
plot_hist(imputation.list = midae.imputed0, var.name = "BMPHEAD", original.data = nhanes3_newborn)



mivae.imputed0 <- mivae(data = nhanes3_newborn, m = 5, epochs = 10, directory = "C:/Users/agnes/Desktop/torch")
# midae.imputed

show_var(imputation.list = mivae.imputed0, var.name = "BMPHEAD", original.data = nhanes3_newborn)
plot_hist(imputation.list = mivae.imputed0, var.name = "BMPHEAD", original.data = nhanes3_newborn)
