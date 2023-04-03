setwd("C:/Users/agnes/Desktop/phd-thesis/my-packages/miae")
devtools::load_all()
devtools::document()





#
#old data have all, new data have 2
set.seed(2023)
iris.binary<-iris
iris.binary$Gender<-sample(c("F","M"),size=nrow(iris),replace = T)
iris.binary$Gender<-as.factor(iris.binary$Gender)
str(iris.binary)
newdata <- createNA(data = iris.binary, var.names=c("Sepal.Length",
                                                    "Gender"
),p = 0.2)
colSums(is.na(withNA.df))

withNA.df <- createNA(data = iris.binary,p = 0.2)
colSums(is.na(newdata))

#
#type null
torch::torch_manual_seed(1234)
imputed.data <- mivae(data = withNA.df, beta=1,
                      pmm.type = NULL, pmm.k = 5, pmm.link="prob", pmm.save.vars = NULL,
                      m = 5, epochs = 5,
                      input.dropout = 0, hidden.dropout = 0,loss.na.scale = TRUE,
                      latent.dim=16,
                      save.model = TRUE, path = file.path(tempdir(), "midaemodel.pt"))


newimpute<-impute_new(object=imputed.data,newdata=newdata)
newimpute


torch::torch_manual_seed(1234)
imputed.data <- mivae(data = withNA.df, subsample=1,
                      pmm.type = 0, pmm.k = 5, pmm.link="prob", pmm.save.vars = NULL,
                      m = 5, epochs = 5,
                      input.dropout = 0, hidden.dropout = 0,loss.na.scale = TRUE,
                      latent.dim=16,
                      save.model = TRUE, path = file.path(tempdir(), "midaemodel.pt"))


newimpute<-impute_new(object=imputed.data,newdata=newdata)
newimpute


torch::torch_manual_seed(1234)
imputed.data <- mivae(data = withNA.df, subsample=0.6,
                      pmm.type = "auto", pmm.k = 5, pmm.link="prob", pmm.save.vars = NULL,
                      m = 5, epochs = 5,
                      input.dropout = 0.2, hidden.dropout = 0.5,loss.na.scale = TRUE,
                      latent.dim=16,
                      save.model = TRUE, path = file.path(tempdir(), "midaemodel.pt"))


newimpute<-impute_new(object=imputed.data,newdata=newdata)
newimpute






############
set.seed(2023)
iris.binary<-iris
iris.binary$Gender<-sample(c("F","M"),size=nrow(iris),replace = T)
iris.binary$Gender<-as.factor(iris.binary$Gender)
str(iris.binary)



withNA.df<-createNA(data = iris.binary,p = 0.2)


newdata <- createNA(data = iris.binary,p = 0.2)
colSums(is.na(newdata))


dim(data)
data=withNA.df
m = 5
epochs = 5
batch.size = 150
subsample = 1
shuffle = TRUE
input.dropout = 0
hidden.dropout = 0
optimizer = "adamW"
learning.rate = 0.0001
weight.decay = 0.002
momentum = 0
eps = 1e-07
encoder.structure = c(256, 256, 256)
latent.dim = 4
decoder.structure = c(256, 256, 256)
act = "identity"
init.weight="xavier.midas"
scaler= "none"
verbose = TRUE
print.every.n = 1
save.model = TRUE
loss.na.scale=T


pmm.type = "auto"

pmm.type = NULL

pmm.k = 5
pmm.link = "prob"
pmm.save.vars = NULL
save.model = TRUE
path = file.path(tempdir(),"midaemodel.pt")
