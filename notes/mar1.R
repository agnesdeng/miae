create.data<-function(n_obs=500){
  # covariance matrix
  ## Fixed covariance matrix
  cov.m <- matrix(c(1, -.12, -.1, .5, .1,
                        -.12, 1, .1, -.6, .1,
                        -.1, .1, 1, -.5, .1,
                        .5, -.6, -.5, 1, .1,
                        .1, .1, .1, .1, 1),
                      ncol = 5)


  # multivariate normal(4 variables)
  df <- mvrnorm(n = 500, mu= c(0,0,0,0,0), Sigma = cov.m) %>%
    as.data.frame(.)

  colnames(df) <- c("Y","X1","X2","X3","X4")
  df
}



make.missing<-function(data=mar1.df){
  # Generate empty missingness matrix
  M <- matrix(ncol = 5, nrow = 500)

  U1 <- runif(500,0,1)
  M[,1] <- ifelse(U1 < 0.85, 1, 0)
  M[,5] <- ifelse(U1 < 0.85, 1, 0)

  # U5 <- runif(500,0,1)
  # M[,5] <- ifelse(U5 < 0.6, 1, 0)

  # X3 is completely observed
  M[,4] <- 1

  # X1 and X2 are MAR:
  U2 <- runif(500,0,1)
  M[,2] <- ifelse((data[,"X3"] < -1) & (U2 < 0.9), 0, 1)

  U3 <- runif(500,0,1)
  M[,3] <- ifelse((data[,"X3"] < -1) & (U3 < 0.9), 0, 1)

  # Copy D to create missing data matrix
  miss <- data

  # Loop over columns applying missingness
  for (i in 1:5) {
    miss[,i] <- ifelse(M[,i] == 0, NA, data[,i])
  }

  miss

}


mar1.df<-create.data(n_obs=500)
lm(Y~X1+X2,data=mar1.df)

withNA.df<-make.missing(data=mar1.df)
withNA.df

library(devtools)
devtools::load_all()


data=withNA.df
m=10
epochs=5
batch.size=16
input.dropout = 0.8
latent.dropout = 0
hidden.dropout = 0.5
optimizer = "adam"
learning.rate = 1e-4
weight.decay = 0
momentum = 0
encoder.structure = c(256,256,256)
latent.dim = 4
decoder.structure = c(256,256,256)
verbose = TRUE
print.every.n = 1
path = "C:/Users/agnes/Desktop/phd-thesis/my-packages/miae/notes/midas.pt"

midas.data<-midae(data=withNA.df, m=10, epochs=5, batch.size=16,
                  input.dropout = 0.8, latent.dropout = 0, hidden.dropout = 0.5,
                  optimizer = "adam", learning.rate = 1e-4, weight.decay = 0, momentum = 0,
                  encoder.structure = c(256,256,256), latent.dim = 4, decoder.structure = c(256,256,256),
                  verbose = TRUE, print.every.n = 1, path = "C:/Users/agnes/Desktop/phd-thesis/my-packages/miae/notes/midas.pt")

midas.data
plot_hist(imputation.list = midas.data,var.name = "Y",original.data = withNA.df)
plot_hist(imputation.list = midas.data,var.name = "X1",original.data = withNA.df)
plot_hist(imputation.list = midas.data,var.name = "X2",original.data = withNA.df)
plot_hist(imputation.list = midas.data,var.name = "X4",original.data = withNA.df)

plot_2num(imputation.list = midas.data,var.x ="X1" ,var.y ="Y" ,original.data = withNA.df)
plot_2num(imputation.list = midas.data,var.x ="X2" ,var.y ="Y" ,original.data = withNA.df)


library(mice)

mice.data<-mice(data=withNA.df,m=10)
mice.data<-complete(mice.data,action="all")

plot_hist(imputation.list = mice.data,var.name = "Y",original.data = withNA.df)
plot_hist(imputation.list = mice.data,var.name = "X1",original.data = withNA.df)
plot_hist(imputation.list = mice.data,var.name = "X2",original.data = withNA.df)
plot_hist(imputation.list = mice.data,var.name = "X4",original.data = withNA.df)

plot_2num(imputation.list = mice.data,var.x ="X1" ,var.y ="Y" ,original.data = withNA.df)
plot_2num(imputation.list = mice.data,var.x ="X2" ,var.y ="Y" ,original.data = withNA.df)
