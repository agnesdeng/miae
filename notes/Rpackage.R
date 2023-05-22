setwd("C:/Users/agnes/Desktop/phd-thesis/my-packages/miae")

devtools::load_all()
devtools::document()


set.seed(2023)
withNA.df <- createNA(data = iris, p = 0.2)

torch::torch_manual_seed(2023)
imputed.data <- midae(data = withNA.df, m = 5, epochs = 100, latent.dim=2,subsample = 0.7,batch.size = 30, early_stopping_epochs =1,path = file.path(tempdir(), "midaemodel.pt"))


set.seed(2023)
withNA.df <- createNA(data = iris, p = 0.2)
torch::torch_manual_seed(2023)
imputed.data <- mivae(data = withNA.df, m = 5, epochs = 100, latent.dim=2,subsample = 0.7,batch.size = 30, early_stopping_epochs =1,path = file.path(tempdir(), "midaemodel.pt"))



data, m = 5, device = "cpu", pmm.type = "auto", pmm.k = 5, pmm.link = "prob", pmm.save.vars = NULL,
epochs = 5, batch.size = 32,
subsample = 1, shuffle = TRUE,
input.dropout = 0.2, hidden.dropout = 0.5,
optimizer = "adamW", learning.rate = 0.0001, weight.decay = 0.002, momentum = 0, eps = 1e-07,
encoder.structure = c(128, 64, 32), latent.dim = 16, decoder.structure = c(32, 64, 128),
act = "elu", init.weight = "xavier.normal", scaler = "none",
loss.na.scale = FALSE,
early_stopping_epochs = 1,
verbose = TRUE, print.every.n = 1, save.model = FALSE, path = NULL

withNA.df <- createNA(data = iris, p = 0.2)



data = withNA.df
m = 5
epochs = 5
subsample = 0.7
path = file.path(tempdir(), "midaemodel.pt")


device = "cpu"
pmm.type = "auto"
pmm.k = 5
pmm.link = "prob"
pmm.save.vars = NULL

batch.size =1000
shuffle = TRUE
input.dropout = 0.2
hidden.dropout = 0.5
optimizer = "adamW"
learning.rate = 0.0001
weight.decay = 0.002
momentum = 0
eps = 1e-07
encoder.structure = c(128, 64, 32)
latent.dim = 16
decoder.structure = c(32, 64, 128)
act = "elu"
init.weight = "xavier.normal"
scaler = "none"
loss.na.scale = FALSE
early_stopping_epochs = 2
verbose = TRUE
print.every.n = 1
save.model = FALSE
drop.last = FALSE


n.samples = train.samples
batch.size = batch.size
drop.last = drop.last

#create package
library(usethis)
usethis::create_package(path = "C:/Users/agnes/Desktop/phd-thesis/my-packages/miae")

library(devtools)
library(usethis)

#set github
use_git()
use_gpl3_license()

#create function inside R folder
use_r("createNA")

load_all()
document()
check()
install()

library(miae)

use_readme_rmd()

createNA(data=iris,p=0.3)



data=iris
