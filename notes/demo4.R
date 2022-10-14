library(devtools)
devtools::document()
devtools::load_all()

if(init.weight=="xavier.normal"){
  model$apply(init_xavier_normal())
}else if(init.weight=="xavier.uniform"){
  model$apply(init_xavier_uniform)
}else if(init.weight=="xavier.midas"){
  model$apply(init_xavier_midas)
}



withNA.df <- createNA(data = iris, p = 0.2)
#no dropout
midae.data <- midae(data = withNA.df, m = 5, epochs = 5,
                    input.dropout = 0, latent.dropout = 0, hidden.dropout = 0,
                    path = "C:/Users/agnes/Desktop/phd-thesis/my-packages/miae/notes/midae.pt")

plot_hist(imputation.list = midae.data,
          var.name="Sepal.Length",
          original.data= withNA.df)

show_var(imputation.list = midae.data,
         var.name="Sepal.Length",
         original.data= withNA.df)

##only drop hidden
midae.data <- midae(data = withNA.df, m = 5, epochs = 5,
                    input.dropout = 0, latent.dropout = 0, hidden.dropout = 0.5,
                    path = "C:/Users/agnes/Desktop/phd-thesis/my-packages/miae/notes/midae.pt")

plot_hist(imputation.list = midae.data,
          var.name="Sepal.Length",
          original.data= withNA.df)

show_var(imputation.list = midae.data,
         var.name="Sepal.Length",
         original.data= withNA.df)


##only drop input (imputed very similar)
midae.data <- midae(data = withNA.df, m = 5, epochs = 5,
                    input.dropout = 0.5, latent.dropout = 0, hidden.dropout = 0,
                    path = "C:/Users/agnes/Desktop/phd-thesis/my-packages/miae/notes/midae.pt")

plot_hist(imputation.list = midae.data,
          var.name="Sepal.Length",
          original.data= withNA.df)

show_var(imputation.list = midae.data,
         var.name="Sepal.Length",
         original.data= withNA.df)

###only drop latent (better than only drop input, but still very similar)
midae.data <- midae(data = withNA.df, m = 5, epochs = 5,
                    input.dropout = 0, latent.dropout = 0.5, hidden.dropout = 0,
                    path = "C:/Users/agnes/Desktop/phd-thesis/my-packages/miae/notes/midae.pt")

plot_hist(imputation.list = midae.data,
          var.name="Sepal.Length",
          original.data= withNA.df)

show_var(imputation.list = midae.data,
         var.name="Sepal.Length",
         original.data= withNA.df)

##drop both input and hidden
midae.data <- midae(data = withNA.df, m = 5, epochs = 5,
                    input.dropout = 0.5, latent.dropout = 0, hidden.dropout = 0.5,
                    path = "C:/Users/agnes/Desktop/phd-thesis/my-packages/miae/notes/midae.pt")

plot_hist(imputation.list = midae.data,
          var.name="Sepal.Length",
          original.data= withNA.df)

show_var(imputation.list = midae.data,
         var.name="Sepal.Length",
         original.data= withNA.df)



# mivae -------------------------------------------------------------------




mivae.data <- mivae(data = withNA.df, m = 5, epochs = 5,
                    input.dropout = 0, latent.dropout = 0, hidden.dropout = 0.5,
                    path = "C:/Users/agnes/Desktop/phd-thesis/my-packages/miae/notes/mivae.pt")

plot_hist(imputation.list = mivae.data,
          var.name="Sepal.Length",
          original.data= withNA.df)

show_var(imputation.list = mivae.data,
         var.name="Sepal.Length",
         original.data= withNA.df)


mivae.data <- mivae(data = withNA.df, m = 5, epochs = 5,
                    input.dropout = 0, latent.dropout = 0, hidden.dropout = 0,
                    path = "C:/Users/agnes/Desktop/phd-thesis/my-packages/miae/notes/mivae.pt")

plot_hist(imputation.list = mivae.data,
          var.name="Sepal.Length",
          original.data= withNA.df)



tune_dropout()

data = withNA.df

path = "C:/Users/agnes/Desktop/phd-thesis/my-packages/miae/notes/mivae.pt"


m = 5
epochs = 5
batch.size = 50
input.dropout = 0
latent.dropout = 0
hidden.dropout = 0
optimizer = "adam"
learning.rate = 0.001
weight.decay = 0
momentum = 0
encoder.structure = c(128, 64, 32)
latent.dim = 8
decoder.structure = c(32, 64, 128)
verbose = TRUE
print.every.n = 1
