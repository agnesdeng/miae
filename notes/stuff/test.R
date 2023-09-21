data=withNA.df
m = 5
categorical.encoding = "onehot"
device = "cpu"
epochs = 20
batch.size = 50
early.stopping.epochs = 1
subsample = 1
dae.params = list(shuffle = TRUE, drop.last = FALSE,
                  input.dropout = 0, hidden.dropout = 0.5,
                  optimizer = "adamW", learning.rate = 0.001, weight.decay = 0.01, momentum = 0, dampening = 0, eps = 1e-08, rho = 0.9, alpha = 0.99, learning.rate.decay = 0,
                  encoder.structure = c(8), latent.dim = 4, decoder.structure = c(8),
                  act = "relu", init.weight = "he.normal", scaler = "standard",initial.imp = "sample", lower=0.25, upper=0.75)
pmm.params = list(pmm.type = NULL, pmm.k = 5, pmm.link = "prob", pmm.save.vars = NULL)
loss.na.scale = FALSE
verbose = TRUE
print.every.n = 1
save.model = FALSE
path = file.path(model.dir, "midaemodel.pt")
