set.seed(2023)
x<-rnorm(1000,mean=100,sd=30)
x.df<-data.frame(x=x)
withNA.df<-createNA(data=x.df,p=0.5)

robust_scaler(withNA.df,initial.imp = "median")





setwd("C:/Users/agnes/Desktop/phd-thesis/my-projects/miae-paper/supplement/mixgbsim")

source("sim/common.R")


full.df <- readRDS("data/full.rds")

withNA.df <- mar_mix(full.df)


setwd("C:/Users/agnes/Desktop/phd-thesis/my-packages/miae")

devtools::load_all()
devtools::document()


midae.imp<-midae(data=withNA.df, m = 5,
                 categorical.encoding = "onehot", device = "cpu",
                 epochs = 5, batch.size = 1000,
                 early.stopping.epochs = 5, subsample = 0.7,
                 dae.params = list(shuffle = TRUE, drop.last = FALSE,
                                   input.dropout = 0.2, hidden.dropout = 0.5,
                                   optimizer = "adamW", learning.rate = 0.001, weight.decay = 0.01, momentum = 0, dampening = 0, eps = 1e-08, rho = 0.9, alpha = 0.99, learning.rate.decay = 0,
                                   encoder.structure = c(128, 64, 32), latent.dim = 16, decoder.structure = c(32, 64, 128),
                                   act = "relu", init.weight = "he.normal", scaler = "standard",initial.imp = "sample", lower=0.25, upper=0.75),
                 pmm.params = list(pmm.type = "auto", pmm.k = 5, pmm.link = "prob", pmm.save.vars = NULL),
                 loss.na.scale = FALSE,
                 verbose = TRUE,print.every.n = 1,
                 save.model = FALSE,
                 path = file.path(tempdir(), "midaemodel.pt"))



mivae.imp<-mivae(data=withNA.df, m = 5, epochs = 5, batch.size = 1000,
                 categorical.encoding = "onehot", device = "cpu",
                 subsample = 0.7, early.stopping.epochs = 1,
                 vae.params = list(shuffle = TRUE, drop.last = FALSE,
                                   beta = 1, input.dropout = 0, hidden.dropout = 0,
                                   optimizer = "adamW", learning.rate = 0.001, weight.decay = 0.01, momentum = 0, dampening = 0, eps = 1e-08, rho = 0.9, alpha = 0.99, learning.rate.decay = 0,
                                   encoder.structure = c(128, 64, 32), latent.dim = 16, decoder.structure = c(32, 64, 128),
                                   act = "relu", init.weight = "he.normal", scaler = "standard",initial.imp = "sample", lower=0.25,upper=0.75),
                 pmm.params = list(pmm.type = NULL, pmm.k = 5, pmm.link = "prob", pmm.save.vars = NULL),
                 loss.na.scale = FALSE,
                 verbose = TRUE,print.every.n = 1,
                 save.model = FALSE,
                 path = file.path(tempdir(), "mivaemodel.pt"))



data=withNA.df
m = 5
categorical.encoding = "onehot"
device = "cpu"
epochs = 5
batch.size = 500
early.stopping.epochs = 5
subsample = 0.7
dae.params = list(shuffle = TRUE, drop.last = FALSE,
                  input.dropout = 0.2, hidden.dropout = 0.5,
                  optimizer = "adamW", learning.rate = 0.0001,
                  weight.decay = 0.002, momentum = 0, eps = 1e-07,
                  encoder.structure = c(128, 64, 32), latent.dim = 16,
                  decoder.structure = c(32, 64, 128),act = "elu",
                  init.weight = "xavier.normal", scaler = "robust",initial.imp="mean",lower=0.25,upper=0.75)
pmm.params = list(pmm.type = "auto", pmm.k = 5,
                  pmm.link = "prob", pmm.save.vars = NULL)
loss.na.scale = FALSE
verbose = TRUE
print.every.n = 1
save.model = FALSE
path = file.path(tempdir(), "midaemodel.pt")

#usethis::use_news_md()

mivae.imp<-mivae(data=withNA.df, m = 5, epochs = 5, batch.size = 100,
                 categorical.encoding = "embeddings", device = "cpu",
                 subsample = 0.7, early.stopping.epochs = 1,
                 vae.params = list(shuffle = TRUE, drop.last = FALSE,
                                   beta= 1, input.dropout = 0.2, hidden.dropout = 0.5,
                                   optimizer = "adamW", learning.rate = 0.0001,
                                   weight.decay = 0.002, momentum = 0, eps = 1e-07,
                                   encoder.structure = c(128, 64, 32), latent.dim = 16,
                                   decoder.structure = c(32, 64, 128),act = "elu",
                                   init.weight = "xavier.normal", scaler = "robust",initial.imp="median",lower=0.2,upper=0.8),
                 pmm.params = list(pmm.type = "auto", pmm.k = 5,
                                   pmm.link = "prob", pmm.save.vars = NULL),
                 loss.na.scale = FALSE,
                 verbose = TRUE,print.every.n = 1,
                 save.model = FALSE,
                 path = file.path(tempdir(), "mivaemodel.pt"))

data=withNA.df
m = 5
epochs = 5
batch.size = 100
categorical.encoding = "embeddings"
device = "cpu"
subsample = 0.7
early.stopping.epochs = 1
vae.params = list(shuffle = TRUE, drop.last = FALSE,
                  beta= 1, input.dropout = 0.2, hidden.dropout = 0.5,
                  optimizer = "adamW", learning.rate = 0.0001,
                  weight.decay = 0.002, momentum = 0, eps = 1e-07,
                  encoder.structure = c(128, 64, 32), latent.dim = 16,
                  decoder.structure = c(32, 64, 128),act = "elu",
                  init.weight = "xavier.normal", scaler = "robust",initial.imp="median")
pmm.params = list(pmm.type = "auto", pmm.k = 5,
                  pmm.link = "prob", pmm.save.vars = NULL)
loss.na.scale = FALSE
verbose = TRUE
print.every.n = 1
save.model = FALSE
path = file.path(tempdir(), "mivaemodel.pt")


midae.imp<-midae(data=withNA.df, m = 5,
                 categorical.encoding = "embeddings", device = "cpu",
                 epochs = 5, batch.size = 500,
                 early.stopping.epochs = 5, subsample = 0.7,
                 dae.params = list(shuffle = TRUE, drop.last = FALSE,
                                   input.dropout = 0.2, hidden.dropout = 0.5,
                                   optimizer = "adamW", learning.rate = 0.0001,
                                   weight.decay = 0.002, momentum = 0, eps = 1e-07,
                                   encoder.structure = c(128, 64, 32), latent.dim = 16,
                                   decoder.structure = c(32, 64, 128),act = "elu",
                                   init.weight = "he.normal", scaler = "robust",initial.imp="mean",lower=0.25,upper=0.75),
                 pmm.params = list(pmm.type = "auto", pmm.k = 5,
                                   pmm.link = "prob", pmm.save.vars = NULL),
                 loss.na.scale = FALSE,
                 verbose = TRUE,print.every.n = 1,
                 save.model = FALSE,
                 path = file.path(tempdir(), "midaemodel.pt"))



midae.imp<-midae(data=withNA.df, m = 5,
                 categorical.encoding = "embeddings", device = "cpu",
                 epochs = 5, batch.size = 500,
                 early.stopping.epochs = 5, subsample = 0.7,
                 dae.params = list(shuffle = TRUE, drop.last = FALSE,
                                   input.dropout = 0.2, hidden.dropout = 0.5,
                                   optimizer = "adamW", learning.rate = 0.0001,
                                   weight.decay = 0.002, momentum = 0, eps = 1e-07,
                                   encoder.structure = c(128, 64, 32), latent.dim = 16,
                                   decoder.structure = c(32, 64, 128),act = "elu",
                                   init.weight = "xavier.normal", scaler = "standard"),
                 pmm.params = list(pmm.type = "auto", pmm.k = 5,
                                   pmm.link = "prob", pmm.save.vars = NULL),
                 loss.na.scale = FALSE,
                 verbose = TRUE,print.every.n = 1,
                 save.model = FALSE,
                 path = file.path(tempdir(), "midaemodel.pt"))

mivae.imp<-mivae(data=withNA.df, m = 5, epochs = 5, batch.size = 500,
                 categorical.encoding = "embeddings", device = "cpu",
                 subsample = 0.7, early.stopping.epochs = 5,
                 vae.params = list(shuffle = TRUE, drop.last = FALSE,
                                   beta= 1, input.dropout = 0.2, hidden.dropout = 0.5,
                                   optimizer = "adamW", learning.rate = 0.0001,
                                   weight.decay = 0.002, momentum = 0, eps = 1e-07,
                                   encoder.structure = c(128, 64, 32), latent.dim = 16,
                                   decoder.structure = c(32, 64, 128),act = "elu",
                                   init.weight = "xavier.normal", scaler = "standard"),
                 pmm.params = list(pmm.type = "auto", pmm.k = 5,
                                   pmm.link = "prob", pmm.save.vars = NULL),
                 loss.na.scale = FALSE,
                 verbose = TRUE,print.every.n = 1,
                 save.model = FALSE,
                 path = file.path(tempdir(), "mivaemodel.pt"))



set.seed(2023)
x<-rnorm(1000,mean=100,sd=30)
x.df<-data.frame(x=x)
withNA.df<-createNA(x.df,p=0.5)

str(withNA.df)

data=withNA.df
m = 5
categorical.encoding = "embeddings"
device = "cpu"
epochs = 10
batch.size = 100
early.stopping.epochs = 1
subsample = 1
vae.params = list(shuffle = TRUE, drop.last = FALSE,
                  beta= 1, input.dropout = 0, hidden.dropout = 0,
                  optimizer = "adamW", learning.rate = 0.0001,
                  weight.decay = 0.002, momentum = 0, eps = 1e-07,
                  encoder.structure = 2, latent.dim = 2,
                  decoder.structure = 2,act = "elu",
                  init.weight = "xavier.normal", scaler = "robust",initial.imp="median")
pmm.params = list(pmm.type = NULL, pmm.k = 5,
                  pmm.link = "prob", pmm.save.vars = NULL)
loss.na.scale = FALSE
verbose = TRUE
print.every.n = 1
save.model = FALSE
path = file.path(tempdir(), "mivaemodel.pt")


imp.data<-mivae(data=withNA.df, categorical.encoding = "embeddings", m = 5, device = "cpu", pmm.type = "auto", pmm.k = 5, pmm.link = "prob", pmm.save.vars = NULL,
                epochs = 10, batch.size = 500, drop.last = FALSE,
                subsample = 0.7, shuffle = TRUE,
                input.dropout = 0.2, hidden.dropout = 0.5,
                optimizer = "adamW", learning.rate = 0.0001, weight.decay = 0.002, momentum = 0, eps = 1e-07,
                encoder.structure = c(128, 64, 32), latent.dim = 16, decoder.structure = c(32, 64, 128),
                act = "elu", init.weight = "xavier.normal", scaler = "standard",
                loss.na.scale = FALSE,
                early.stopping.epochs = 5,
                verbose = TRUE, print.every.n = 1, save.model = FALSE, path = file.path(tempdir(), "mivaemodel.pt"))




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




imp.data<-mivae(data=withNA.df, categorical.encoding = "embeddings", m = 5, device = "cpu", pmm.type = "auto", pmm.k = 5, pmm.link = "prob", pmm.save.vars = NULL,
                 epochs = 10, batch.size = 500, drop.last = FALSE,
                 subsample = 0.7, shuffle = TRUE,
                 input.dropout = 0.2, hidden.dropout = 0.5,
                 optimizer = "adamW", learning.rate = 0.0001, weight.decay = 0.002, momentum = 0, eps = 1e-07,
                 encoder.structure = c(128, 64, 32), latent.dim = 16, decoder.structure = c(32, 64, 128),
                 act = "elu", init.weight = "xavier.normal", scaler = "standard",
                 loss.na.scale = FALSE,
                 early.stopping.epochs = 5,
                 verbose = TRUE, print.every.n = 1, save.model = FALSE, path = file.path(tempdir(), "mivaemodel.pt"))



imp.obj<-mivae(data=withNA.df, categorical.encoding = "embeddings", m = 5, device = "cpu", pmm.type = 0, pmm.k = 5, pmm.link = "prob", pmm.save.vars = NULL,
                epochs = 4, batch.size = 500, drop.last = FALSE,
                subsample = 1, shuffle = TRUE,
                input.dropout = 0, hidden.dropout = 0,
                optimizer = "adamW", learning.rate = 0.0001, weight.decay = 0.002, momentum = 0, eps = 1e-07,
                encoder.structure = c(128, 64, 32), latent.dim = 16, decoder.structure = c(32, 64, 128),
                act = "elu", init.weight = "xavier.normal", scaler = "standard",
                loss.na.scale = FALSE,
                early.stopping.epochs = 1,
                verbose = TRUE, print.every.n = 1, save.model = TRUE, path = file.path(tempdir(), "mivaemodel.pt"))

imp.obj

object=imp.obj
newdata<-withNA.df[1:1000,]
pmm.k = NULL
m = NULL
verbose = FALSE


new.data<-impute_new(object=imp.obj, newdata=withNA.df[1:1000,], pmm.k = NULL, m = NULL, verbose = FALSE)
new.data


mivae.imp<-mivae(data=withNA.df, categorical.encoding = "embeddings", m = 5, device = "cpu", pmm.type = "auto", pmm.k = 5, pmm.link = "prob", pmm.save.vars = NULL,
                 epochs = 10, batch.size = 500, drop.last = FALSE,
                 subsample = 0.7, shuffle = TRUE,
                 input.dropout = 0.2, hidden.dropout = 0.5,
                 optimizer = "adamW", learning.rate = 0.0001, weight.decay = 0.002, momentum = 0, eps = 1e-07,
                 encoder.structure = c(128, 64, 32), latent.dim = 16, decoder.structure = c(32, 64, 128),
                 act = "elu", init.weight = "xavier.normal", scaler = "standard",
                 loss.na.scale = FALSE,
                 early.stopping.epochs = 5,
                 verbose = TRUE, print.every.n = 1, save.model = TRUE, path = file.path(tempdir(), "mivaemodel.pt"))

mivae.imp$imputed.data
mivae.imp$model.path
mivae.imp$params



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
