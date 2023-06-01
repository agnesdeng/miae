


setwd("C:/Users/agnes/Desktop/phd-thesis/my-projects/miae-paper/supplement/mixgbsim")

source("sim/common.R")


full.df <- readRDS("data/full.rds")

withNA.df <- mar_mix(full.df)


setwd("C:/Users/agnes/Desktop/phd-thesis/my-packages/miae")

devtools::load_all()
devtools::document()





midae.imp<-midae(data=withNA.df, m = 5,
                 categorical.encoding = "embeddings", device = "cpu",
                 pmm.type = "auto", pmm.k = 5,
                 pmm.link = "prob", pmm.save.vars = NULL,
                 epochs = 5, batch.size = 500, drop.last = FALSE,
                 subsample = 0.7, shuffle = TRUE,
                 input.dropout = 0.2, hidden.dropout = 0.5,
                 optimizer = "adamW", learning.rate = 0.0001,
                 weight.decay = 0.002, momentum = 0, eps = 1e-07,
                 encoder.structure = c(128, 64, 32), latent.dim = 16,
                 decoder.structure = c(32, 64, 128),act = "elu",
                 init.weight = "xavier.normal", scaler = "standard",
                 early.stopping.epochs = 5,
                 loss.na.scale = FALSE, verbose = TRUE,
                 print.every.n = 1, save.model = FALSE,
                 path = file.path(tempdir(), "midaemodel.pt"))

midae.imp<-midae(data=withNA.df, m = 5,
                 categorical.encoding = "embeddings", device = "cpu",
                 pmm.type = 1, pmm.k = 5,
                 pmm.link = "prob", pmm.save.vars = NULL,
                 epochs = 2, batch.size = 500, drop.last = FALSE,
                 subsample = 0.7, shuffle = TRUE,
                 input.dropout = 0.2, hidden.dropout = 0.5,
                 optimizer = "adamW", learning.rate = 0.0001,
                 weight.decay = 0.002, momentum = 0, eps = 1e-07,
                 encoder.structure = c(128, 64, 32), latent.dim = 16,
                 decoder.structure = c(32, 64, 128),act = "elu",
                 init.weight = "xavier.normal", scaler = "standard",
                 early.stopping.epochs = 5,
                 loss.na.scale = FALSE, verbose = TRUE,
                 print.every.n = 1, save.model = FALSE,
                 path = file.path(tempdir(), "midaemodel.pt"))

midae.imp<-midae(data=withNA.df, categorical.encoding = "embeddings", m = 5, device = "cuda", pmm.type = "auto", pmm.k = 5, pmm.link = "prob", pmm.save.vars = NULL,
                 epochs = 10, batch.size = 500, drop.last = FALSE,
                 subsample = 0.7, shuffle = TRUE,
                 input.dropout = 0.2, hidden.dropout = 0.5,
                 optimizer = "adamW", learning.rate = 0.0001, weight.decay = 0.002, momentum = 0, eps = 1e-07,
                 encoder.structure = c(128, 64, 32), latent.dim = 16, decoder.structure = c(32, 64, 128),
                 act = "elu", init.weight = "xavier.normal", scaler = "standard",
                 loss.na.scale = FALSE,
                 early.stopping.epochs = 5,
                 verbose = TRUE, print.every.n = 1, save.model = FALSE, path = file.path(tempdir(), "midaemodel.pt"))


midae.imp<-midae(data=withNA.df, categorical.encoding = "onehot", m = 5, device = "cpu", pmm.type = "auto", pmm.k = 5, pmm.link = "prob", pmm.save.vars = NULL,
                 epochs = 10, batch.size = 500, drop.last = FALSE,
                 subsample = 0.7, shuffle = TRUE,
                 input.dropout = 0.2, hidden.dropout = 0.5,
                 optimizer = "adamW", learning.rate = 0.0001, weight.decay = 0.002, momentum = 0, eps = 1e-07,
                 encoder.structure = c(128, 64, 32), latent.dim = 16, decoder.structure = c(32, 64, 128),
                 act = "elu", init.weight = "xavier.normal", scaler = "standard",
                 loss.na.scale = FALSE,
                 early.stopping.epochs = 5,
                 verbose = TRUE, print.every.n = 1, save.model = FALSE, path = file.path(tempdir(), "midaemodel.pt"))



midae.imp<-midae(data=withNA.df, categorical.encoding = "onehot", m = 5, device = "cuda", pmm.type = "auto", pmm.k = 5, pmm.link = "prob", pmm.save.vars = NULL,
                 epochs = 10, batch.size = 500, drop.last = FALSE,
                 subsample = 0.7, shuffle = TRUE,
                 input.dropout = 0.2, hidden.dropout = 0.5,
                 optimizer = "adamW", learning.rate = 0.0001, weight.decay = 0.002, momentum = 0, eps = 1e-07,
                 encoder.structure = c(128, 64, 32), latent.dim = 16, decoder.structure = c(32, 64, 128),
                 act = "elu", init.weight = "xavier.normal", scaler = "standard",
                 loss.na.scale = FALSE,
                 early.stopping.epochs = 5,
                 verbose = TRUE, print.every.n = 1, save.model = FALSE, path = file.path(tempdir(), "midaemodel.pt"))




midae.imp




mivae.imp<-mivae(data=withNA.df, categorical.encoding = "embeddings", m = 5, device = "cpu", pmm.type = "auto", pmm.k = 5, pmm.link = "prob", pmm.save.vars = NULL,
                 epochs = 10, batch.size = 500, drop.last = FALSE,
                 subsample = 0.7, shuffle = TRUE,
                 input.dropout = 0.2, hidden.dropout = 0.5,
                 optimizer = "adamW", learning.rate = 0.0001, weight.decay = 0.002, momentum = 0, eps = 1e-07,
                 encoder.structure = c(128, 64, 32), latent.dim = 16, decoder.structure = c(32, 64, 128),
                 act = "elu", init.weight = "xavier.normal", scaler = "standard",
                 loss.na.scale = FALSE,
                 early.stopping.epochs = 5,
                 verbose = TRUE, print.every.n = 1, save.model = FALSE, path = file.path(tempdir(), "mivaemodel.pt"))



mivae.imp<-mivae(data=withNA.df, beta=1,
                 categorical.encoding = "embeddings", m = 5, device = "cuda",
                 pmm.type = "auto", pmm.k = 5, pmm.link = "prob", pmm.save.vars = NULL,
                 epochs = 3, batch.size = 500, drop.last = FALSE,
                 subsample = 0.7, shuffle = TRUE,
                 input.dropout = 0.2, hidden.dropout = 0.5,
                 optimizer = "adamW", learning.rate = 0.0001, weight.decay = 0.002, momentum = 0, eps = 1e-07,
                 encoder.structure = c(128, 64, 32), latent.dim = 16, decoder.structure = c(32, 64, 128),
                 act = "elu", init.weight = "xavier.normal", scaler = "standard",
                 loss.na.scale = FALSE,
                 early.stopping.epochs = 5,
                 verbose = TRUE, print.every.n = 1, save.model = FALSE, path = file.path(tempdir(), "mivaemodel.pt"))


mivae.imp<-mivae(data=withNA.df, categorical.encoding = "onehot", m = 5, device = "cpu", pmm.type = "auto", pmm.k = 5, pmm.link = "prob", pmm.save.vars = NULL,
                 epochs = 10, batch.size = 500, drop.last = FALSE,
                 subsample = 0.7, shuffle = TRUE,
                 input.dropout = 0.2, hidden.dropout = 0.5,
                 optimizer = "adamW", learning.rate = 0.0001, weight.decay = 0.002, momentum = 0, eps = 1e-07,
                 encoder.structure = c(128, 64, 32), latent.dim = 16, decoder.structure = c(32, 64, 128),
                 act = "elu", init.weight = "xavier.normal", scaler = "standard",
                 loss.na.scale = FALSE,
                 early.stopping.epochs = 5,
                 verbose = TRUE, print.every.n = 1, save.model = FALSE, path = file.path(tempdir(), "mivaemodel.pt"))



mivae.imp<-mivae(data=withNA.df, categorical.encoding = "onehot", m = 5, device = "cuda", pmm.type = "auto", pmm.k = 5, pmm.link = "prob", pmm.save.vars = NULL,
                 epochs = 10, batch.size = 500, drop.last = FALSE,
                 subsample = 0.7, shuffle = TRUE,
                 input.dropout = 0.2, hidden.dropout = 0.5,
                 optimizer = "adamW", learning.rate = 0.0001, weight.decay = 0.002, momentum = 0, eps = 1e-07,
                 encoder.structure = c(128, 64, 32), latent.dim = 16, decoder.structure = c(32, 64, 128),
                 act = "elu", init.weight = "xavier.normal", scaler = "standard",
                 loss.na.scale = FALSE,
                 early.stopping.epochs = 5,
                 verbose = TRUE, print.every.n = 1, save.model = FALSE, path = file.path(tempdir(), "mivaemodel.pt"))




# test cuda--------------------------------------------------------------------

data=withNA.df
categorical.encoding = "embeddings"
beta=1
m = 5
device = "cuda"
pmm.type = "auto"
pmm.k = 5
pmm.link = "prob"
pmm.save.vars = NULL
epochs = 10
batch.size = 500
drop.last = FALSE
subsample = 0.7
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
scaler = "standard"
loss.na.scale = FALSE
early.stopping.epochs = 5
verbose = TRUE
print.every.n = 1
save.model = FALSE
path = file.path(tempdir(), "midaemodel.pt")
path = file.path(tempdir(), "mivaemodel.pt")

library(mixgb)
str(nhanes3_newborn)

withNA.df=nhanes3_newborn
withNA.df$HFF1<-as.logical(as.integer(withNA.df$HFF1)-1)
withNA.df$HYD1[2]<-NA
withNA.df$HSHSIZER[1:100]<-NA
withNA.df$HSAGEIR[c(2,4,6,8,10)]<-NA
str(withNA.df)



midae.imp<-midae(data=withNA.df, categorical.encoding = "embeddings", m = 5, device = "cpu", pmm.type = "auto", pmm.k = 5, pmm.link = "prob", pmm.save.vars = NULL,
                 epochs = 100, batch.size = 500, drop.last = FALSE,
                 subsample = 1, shuffle = TRUE,
                 input.dropout = 0.2, hidden.dropout = 0.5,
                 optimizer = "adamW", learning.rate = 0.0001, weight.decay = 0.002, momentum = 0, eps = 1e-07,
                 encoder.structure = c(128, 64, 32), latent.dim = 16, decoder.structure = c(32, 64, 128),
                 act = "elu", init.weight = "xavier.normal", scaler = "standard",
                 loss.na.scale = FALSE,
                 early.stopping.epochs = 1,
                 verbose = TRUE, print.every.n = 1, save.model = FALSE, path = file.path(tempdir(), "midaemodel.pt"))


midae.imp<-midae(data=withNA.df, categorical.encoding = "onehot", m = 5, device = "cpu", pmm.type = "auto", pmm.k = 5, pmm.link = "prob", pmm.save.vars = NULL,
                 epochs = 100, batch.size = 500, drop.last = FALSE,
                 subsample = 1, shuffle = TRUE,
                 input.dropout = 0.2, hidden.dropout = 0.5,
                 optimizer = "adamW", learning.rate = 0.0001, weight.decay = 0.002, momentum = 0, eps = 1e-07,
                 encoder.structure = c(128, 64, 32), latent.dim = 16, decoder.structure = c(32, 64, 128),
                 act = "elu", init.weight = "xavier.normal", scaler = "standard",
                 loss.na.scale = FALSE,
                 early.stopping.epochs = 1,
                 verbose = TRUE, print.every.n = 1, save.model = FALSE, path = file.path(tempdir(), "midaemodel.pt"))





summary(nhanes3_newborn)
categorical.encoding = "embeddings"
midae.imp<-midae(data=withNA.df, m = 5, device = "cpu", pmm.type = "auto", pmm.k = 5, pmm.link = "prob", pmm.save.vars = NULL,
  epochs = 10, batch.size = 500, drop.last = FALSE,
  subsample = 1, shuffle = TRUE,
  input.dropout = 0.2, hidden.dropout = 0.5,
  optimizer = "adamW", learning.rate = 0.0001, weight.decay = 0.002, momentum = 0, eps = 1e-07,
  encoder.structure = c(128, 64, 32), latent.dim = 16, decoder.structure = c(32, 64, 128),
  act = "elu", init.weight = "xavier.normal", scaler = "standard",
  loss.na.scale = FALSE,
  early.stopping.epochs = 1,
  verbose = TRUE, print.every.n = 1, save.model = FALSE, path = file.path(tempdir(), "midaemodel.pt"))

midae.imp


midae.imp<-midae(data=withNA.df, m = 5, device = "cpu", pmm.type = NULL, pmm.k = 5, pmm.link = "prob", pmm.save.vars = NULL,
                 epochs = 10, batch.size = 500, drop.last = FALSE,
                 subsample = 0.7, shuffle = TRUE,
                 input.dropout = 0.2, hidden.dropout = 0.5,
                 optimizer = "adamW", learning.rate = 0.0001, weight.decay = 0.002, momentum = 0, eps = 1e-07,
                 encoder.structure = c(128, 64, 32), latent.dim = 16, decoder.structure = c(32, 64, 128),
                 act = "elu", init.weight = "xavier.normal", scaler = "standard",
                 loss.na.scale = FALSE,
                 early.stopping.epochs = 1,
                 verbose = TRUE, print.every.n = 1, save.model = FALSE, path = file.path(tempdir(), "midaemodel.pt"))

midae.imp






data=withNA.df
data

data
pre.obj<-preprocess(data,scaler="minmax", categorical.encoding = "embeddings")

pre.obj$data.tensor



pre.obj$cardinalities
pre.obj$data.tensor
pre.obj$bin.idx
pre.obj$multi.idx
pre.obj$embedding.dim
pre.obj$cardinalities






mushroom_data

set.seed(2023)
withNA.df <- createNA(data = iris, p = 0.2)

torch::torch_manual_seed(2023)
imputed.data <- midae(data = withNA.df, m = 5, epochs = 100, latent.dim=2,subsample = 0.7,batch.size = 30, early.stopping.epochs =1,path = file.path(tempdir(), "midaemodel.pt"))


set.seed(2023)
withNA.df <- createNA(data = iris, p = 0.2)
torch::torch_manual_seed(2023)
imputed.data <- mivae(data = withNA.df, m = 5, epochs = 100, latent.dim=2,subsample = 0.7,batch.size = 30, early.stopping.epochs =1,path = file.path(tempdir(), "midaemodel.pt"))



data, m = 5, device = "cpu", pmm.type = "auto", pmm.k = 5, pmm.link = "prob", pmm.save.vars = NULL,
epochs = 5, batch.size = 32,
subsample = 1, shuffle = TRUE,
input.dropout = 0.2, hidden.dropout = 0.5,
optimizer = "adamW", learning.rate = 0.0001, weight.decay = 0.002, momentum = 0, eps = 1e-07,
encoder.structure = c(128, 64, 32), latent.dim = 16, decoder.structure = c(32, 64, 128),
act = "elu", init.weight = "xavier.normal", scaler = "none",
loss.na.scale = FALSE,
early.stopping.epochs = 1,
verbose = TRUE, print.every.n = 1, save.model = FALSE, path = NULL

withNA.df <- createNA(data = iris, p = 0.2)



data = withNA.df
m = 5
epochs = 10
subsample = 0.7
path = file.path(tempdir(), "midaemodel.pt")

categorical.encoding = "embeddings"
categorical.encoding = "onehot"
device = "cuda"
pmm.type = "auto"
pmm.k = 5
pmm.link = "prob"
pmm.save.vars = NULL

batch.size =500
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
scaler = "standard"
loss.na.scale = FALSE
early.stopping.epochs = 2
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
