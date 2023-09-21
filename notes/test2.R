data=withNA.df
m = 5
categorical.encoding = "embeddings"
device = "cpu"
epochs = 10
batch.size = 50
early.stopping.epochs = 1
subsample = 1
dae.params = list(shuffle = TRUE, drop.last = FALSE,
                  input.dropout = 0.2, hidden.dropout = 0.5,
                  optimizer = "adamW", learning.rate = 0.0001, weight.decay = 0.002, momentum = 0, dampening = 0, eps = 1e-08, rho = 0.9, alpha = 0.99, learning.rate.decay = 0,
                  encoder.structure = c(128, 64, 32), latent.dim = 16, decoder.structure = c(32, 64, 128),
                  act = "elu", init.weight = "xavier.midas", scaler = "none",initial.imp = "sample", lower=0.25, upper=0.75)
pmm.params = list(pmm.type = "auto", pmm.k = 5, pmm.link = "prob", pmm.save.vars = NULL)
loss.na.scale = FALSE
verbose = TRUE
print.every.n = 1
save.model = FALSE
path = file.path(model.dir, "midaemodel.pt")
