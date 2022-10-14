devtools::document()
devtools::load_all()

withNA.df <- createNA(data=iris,p=0.3)

mivae.data <- mivae(data=withNA.df, m = 5,
                    epochs = 5, batch.size = 32,
                    input.dropout = 0.2, hidden.dropout = 0.5,
                    encoder.structure = c(128, 64, 32), latent.dim = 8, decoder.structure = c(32, 64, 128),
                    init.weight="xavier.normal",
                    scaler = "standard",
                    verbose = TRUE, print.every.n = 1)


mivae.data <- mivae(data=withNA.df, m = 5,
                    epochs = 5, batch.size = 32,
                    input.dropout = 0.2, hidden.dropout = 0.5,
                    encoder.structure = c(128, 64, 32), latent.dim = 8, decoder.structure = c(32, 64, 128),
                    init.weight="xavier.midas",
                    scaler = "standard",
                    verbose = TRUE, print.every.n = 1)


mivae.data <- mivae(data=withNA.df, m = 5,
                    epochs = 5, batch.size = 32,
                    input.dropout = 0.2, hidden.dropout = 0.5,
                    encoder.structure = c(128, 64, 32), latent.dim = 8, decoder.structure = c(32, 64, 128),
                    init.weight="xavier.uniform",
                    scaler = "standard",
                    verbose = TRUE, print.every.n = 1)

mivae.data <- mivae(data=withNA.df, m = 5,
                    epochs = 5, batch.size = 32,
                    input.dropout = 0.2, hidden.dropout = 0.5,
                    encoder.structure = c(128, 64, 32), latent.dim = 8, decoder.structure = c(32, 64, 128),
                    init.weight="kaiming",
                    scaler = "standard",
                    verbose = TRUE, print.every.n = 1)

###minmax
mivae.data <- mivae(data=withNA.df, m = 5,
                    epochs = 5, batch.size = 32,
                    input.dropout = 0.2, hidden.dropout = 0.5,
                    encoder.structure = c(128, 64, 32), latent.dim = 8, decoder.structure = c(32, 64, 128),
                    init.weight="xavier.normal",
                    scaler = "minmax",
                    verbose = TRUE, print.every.n = 1)

mivae.data <- mivae(data=withNA.df, m = 5,
                    epochs = 5, batch.size = 32,
                    input.dropout = 0.2, hidden.dropout = 0.5,
                    encoder.structure = c(128, 64, 32), latent.dim = 8, decoder.structure = c(32, 64, 128),
                    init.weight="xavier.uniform",
                    scaler = "minmax",
                    verbose = TRUE, print.every.n = 1)

mivae.data <- mivae(data=withNA.df, m = 5,
                    epochs = 5, batch.size = 32,
                    input.dropout = 0.2, hidden.dropout = 0.5,
                    encoder.structure = c(128, 64, 32), latent.dim = 8, decoder.structure = c(32, 64, 128),
                    init.weight="xavier.midas",
                    scaler = "minmax",
                    verbose = TRUE, print.every.n = 1)




##
data=withNA.df
m = 5
epochs = 5
input.dropout = 0.2
hidden.dropout = 0.5
batch.size = 32
split.ratio = 0.7
shuffle = TRUE
optimizer = "adamW"
learning.rate = 0.0001
weight.decay = 0.002
momentum = 0
encoder.structure = c(128, 64, 32)
latent.dim = 8
decoder.structure = c(32, 64, 128)
act = "elu"
init.weight="xavier.uniform"
scaler= "standard"
verbose = TRUE
print.every.n = 1
save.model = FALSE
path = NULL


##


mivae.data <- mivae(data=withNA.df, m = 5,
                    epochs = 5, batch.size = 32,
                    input.dropout = 0.2, hidden.dropout = 0.5,
                    encoder.structure = c(10, 8), latent.dim = 4, decoder.structure = c(8,10),
                    init.weight="xavier.normal",
                    scaler = "standard",
                    verbose = TRUE, print.every.n = 1)




mivae.data <- mivae(data=withNA.df, m = 5,
                    epochs = 5, batch.size = 32,
                    input.dropout = 0.2, hidden.dropout = 0.5,
                    init.weight="xavier.uniform",
                    scaler = "standard",
                    verbose = TRUE, print.every.n = 1)



mivae.data <- mivae(data=withNA.df, m = 5,
                    epochs = 5, batch.size = 32,
                    input.dropout = 0.2, hidden.dropout = 0.5,
                    init.weight="kaming",
                    scaler = "standard",
                    verbose = TRUE, print.every.n = 1)


mivae.data <- mivae(data=withNA.df, m = 5,
                    epochs = 5, batch.size = 32,
                    input.dropout = 0.2, hidden.dropout = 0.5,
                    init.weight="xavier.midas",
                    scaler = "standard",
                    verbose = TRUE, print.every.n = 1)


plot_hist(imputation.list=mivae.data,var.name="Sepal.Length",original.data=withNA.df)

plot_density(imputation.list=mivae.data,var.name="Sepal.Length",original.data=withNA.df)



###
data=withNA.df
dropout.grid = list(input.dropout=c(0.2,0.7),hidden.dropout=c(0.3,0.8))
m=5
epochs = 5
batch.size = 32
split.ratio = 0.7
shuffle = TRUE
optimizer = "adamW"
learning.rate = 0.0001
weight.decay = 0.002
momentum = 0
encoder.structure = c(128, 64, 32)
latent.dim = 8
decoder.structure = c(32, 64, 128)
act = "elu"
init.weight="xavier.uniform"
scaler= "standard"
verbose = TRUE
print.every.n = 1
save.model = FALSE
path = NULL
input.dropout=0.2
hidden.dropout=0.3

###

tune.vae<-tune_vae_dropout(data=withNA.df, dropout.grid = list(input.dropout=c(0.2,0.5,0.8),hidden.dropout=c(0.3,0.5,0.7)),
                           m=5, epochs = 5, batch.size = 32,
                           split.ratio = 0.7, shuffle = TRUE,
                           optimizer = "adamW", learning.rate = 0.0001, weight.decay = 0.002, momentum = 0,
                           encoder.structure = c(128, 64, 32), latent.dim = 8, decoder.structure = c(32, 64, 128),
                           act = "elu", init.weight="xavier.uniform", scaler= "standard",
                           verbose = TRUE, print.every.n = 1, save.model = FALSE, path = NULL)


plot_dropout(tune.results=tune.vae,var.name = "Sepal.Length")


mivae.data <- mivae(data=withNA.df, m = 5,
                    epochs = 5, batch.size = 32,
                    input.dropout = 0.2, hidden.dropout = 0.8,
                    scaler = "standard",
                    verbose = TRUE, print.every.n = 1)


plot_hist(imputation.list=mivae.data,var.name="Sepal.Length",original.data=withNA.df)

plot_density(imputation.list=mivae.data,var.name="Sepal.Length",original.data=withNA.df)



