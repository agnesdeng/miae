usethis::use_vignette("tune-dropout")
devtools::build_vignettes()

# set working directory to the folder simulation---------------------------------------------------
setwd("C:/Users/agnes/Desktop/phd-thesis/my-projects/miae-paper/simulation")

# server
# setwd("/data/users/yden863/sim1000")


source("sim/common.R")
library(miae)
source("sim/sim_midae.R")



full.df <- readRDS("data/full.rds")


withNA.df <- mar_mix(full.df)

midae.data<-midae(data=withNA.df,m = 5,
                   epochs = 5, batch.size = 100,
                   input.dropout = 0.8, latent.dropout = 0, hidden.dropout = 0.5,
                   optimizer = "adam", learning.rate = 0.001, weight.decay = 0, momentum = 0,
                   encoder.structure = c(256, 256, 256), latent.dim = 4, decoder.structure = c(256, 256, 256),
                   verbose = TRUE, print.every.n = 1,
                   path = "midaemodels/midaetest.pt")

mivae.data<-midae(data=withNA.df,m = 5,
                  epochs = 5, batch.size = 100,
                  input.dropout = 0.8, latent.dropout = 0, hidden.dropout = 0.5,
                  optimizer = "adam", learning.rate = 0.001, weight.decay = 0, momentum = 0,
                  encoder.structure = c(256, 256, 256), latent.dim = 4, decoder.structure = c(256, 256, 256),
                  verbose = TRUE, print.every.n = 1,
                  path = "mivaemodels/mivaetest.pt")

show_var(imputation.list=midae.data,var.name="norm1",original.data=withNA.df)
show_var(imputation.list=mivae.data,var.name="norm1",original.data=withNA.df)

plot_hist(imputation.list=midae.data,var.name="norm1",original.data=withNA.df,true.data=full.df)
plot_hist(imputation.list=mivae.data,var.name="norm1",original.data=withNA.df,true.data=full.df)



dropout.grid<-list(input.dropout=c(0.2,0.5,0.8),hidden.dropout=c(0.2,0.5))

library(miae)
midae.tune<-tune_dae_dropout(data=withNA.df, m=5, epochs = 5, batch.size = 100,
                 dropout.grid = dropout.grid, latent.dropout = 0,
                 optim = "adam", learning.rate = 0.001, weight.decay = 0, momentum = 0,
                 encoder.structure = c(256, 256, 256), latent.dim = 4, decoder.structure = c(256, 256, 256),
                 verbose = TRUE, print.every.n = 1,
                 path = "mivaemodels/mivaetune.pt")


plot_dropout(tune.results = midae.tune,var.name="norm1")
plot_dropout(tune.results = midae.tune,var.name="norm2")





dropout.grid<-list(input.dropout=c(0.1,0.3,0.6,0.9),hidden.dropout=c(0.1,0.3,0.6,0.9))

library(miae)
midae.tune<-tune_dae_dropout(data=withNA.df, m=5, epochs = 5, batch.size = 100,
                             dropout.grid = dropout.grid, latent.dropout = 0,
                             optim = "adam", learning.rate = 0.001, weight.decay = 0, momentum = 0,
                             encoder.structure = c(256, 256, 256), latent.dim = 4, decoder.structure = c(256, 256, 256),
                             verbose = TRUE, print.every.n = 1,
                             path = "mivaemodels/mivaetune.pt")

midae.tune$imputed.missing$norm1

plot_dropout(tune.results = midae.tune,var.name="norm1")
plot_dropout(tune.results = midae.tune,var.name="norm1", xlim=c(-4,4))
plot_dropout(tune.results = midae.tune,var.name="norm2")

