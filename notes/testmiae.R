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
system.time(
midae(data = withNA.df,
                        pmm.type = NULL, pmm.k = 5, pmm.link="prob", pmm.save.vars = NULL,
                        subsample=1,
                        m = 5, epochs = 5,
                        input.dropout = 0, hidden.dropout = 0,loss.na.scale = TRUE,
                        save.model = TRUE, path = file.path(tempdir(), "midaemodel.pt"))
)

torch::torch_manual_seed(1234)
system.time(
  midae(data = withNA.df, device = "cuda",
        pmm.type = NULL, pmm.k = 5, pmm.link="prob", pmm.save.vars = NULL,
        subsample=1,
        m = 5, epochs = 5,
        input.dropout = 0, hidden.dropout = 0,loss.na.scale = TRUE,
        save.model = TRUE, path = file.path(tempdir(), "midaemodel.pt"))
)

torch::torch_manual_seed(1234)
imputed.data <- midae(data = withNA.df, device = "cpu",
                      pmm.type = NULL, pmm.k = 5, pmm.link="prob", pmm.save.vars = NULL,
                      subsample=1,
                      m = 5, epochs = 5,
                      input.dropout = 0, hidden.dropout = 0,loss.na.scale = TRUE,
                      save.model = TRUE, path = file.path(tempdir(), "midaemodel.pt"))

#
#type null
torch::torch_manual_seed(1234)
imputed.data <- midae(data = withNA.df, device = "cuda",
                      pmm.type = NULL, pmm.k = 5, pmm.link="prob", pmm.save.vars = NULL,
                      subsample=1,
                      m = 5, epochs = 5,
                      input.dropout = 0, hidden.dropout = 0,loss.na.scale = TRUE,
                      save.model = TRUE, path = file.path(tempdir(), "midaemodel.pt"))



newimpute<-impute_new(object=imputed.data,newdata=newdata)
newimpute

#type 0

torch::torch_manual_seed(1234)
imputed.data0 <- midae(data = withNA.df,
                       pmm.type = 0, pmm.k = 5, pmm.link="logit", pmm.save.vars = NULL,
                       m = 5, epochs = 5,
                       input.dropout = 0, hidden.dropout = 0,loss.na.scale = TRUE,
                       save.model = TRUE, path = file.path(tempdir(), "midaemodel.pt"))

object=imputed.data0
imputed.data0$params$yhatobs.list
newimpute<-impute_new(object=imputed.data0,newdata=newdata)



#type 1
torch::torch_manual_seed(1234)
imputed.data01 <- midae(data = withNA.df,
                        pmm.type = 1, pmm.k = 5, pmm.link="logit", pmm.save.vars = NULL,
                        m = 5, epochs = 5,
                        input.dropout = 0.2, hidden.dropout = 0,loss.na.scale = TRUE,
                        save.model = TRUE, path = file.path(tempdir(), "midaemodel.pt"))

object=imputed.data01
object$params$yhatobs.list
newimpute<-impute_new(object=imputed.data01,newdata=newdata)




object$params
newdata
pmm.k = NULL
m = NULL
verbose = FALSE

#type 2
torch::torch_manual_seed(1234)
imputed.data2 <- midae(data = withNA.df,
                       pmm.type = 2, pmm.k = 5, pmm.link="logit", pmm.save.vars = NULL,
                       m = 5, epochs = 5,
                       input.dropout = 0.2, hidden.dropout = 0.5,
                       save.model = TRUE, path = file.path(tempdir(), "midaemodel.pt"))

object=imputed.data2

newimpute<-impute_new(object=imputed.data2,newdata=newdata)
object$params$yhatobs.list

newimpute

#type auto
torch::torch_manual_seed(1234)
imputed.dataA <- midae(data = withNA.df,
                       pmm.type = "auto", pmm.k = 5, pmm.link="logit", pmm.save.vars = NULL,
                       m = 5, epochs = 5,
                       input.dropout = 0.2, hidden.dropout = 0.5,
                       save.model = TRUE, path = file.path(tempdir(), "midaemodel.pt"))

imputed.dataA$params$yhatobs.list

newimpute<-impute_new(object=imputed.dataA,newdata=newdata)

torch::torch_manual_seed(1234)
imputed.dataA <- midae(data = withNA.df,
                       pmm.type = "auto", pmm.k = 5, pmm.link="logit", pmm.save.vars = colnames(withNA.df),
                       m = 5, epochs = 5,
                       input.dropout = 0.2, hidden.dropout = 0.5,
                       save.model = TRUE, path = file.path(tempdir(), "midaemodel.pt"))

imputed.dataA$params$yhatobs.list

newimpute<-impute_new(object=imputed.dataA,newdata=newdata)




#old data have 2, new data have all
set.seed(2023)
iris.binary<-iris
iris.binary$Gender<-sample(c("F","M"),size=nrow(iris),replace = T)
iris.binary$Gender<-as.factor(iris.binary$Gender)
str(iris.binary)
withNA.df <- createNA(data = iris.binary, var.names=c("Sepal.Length",
                                                      "Species"
                                                      ),p = 0.2)
colSums(is.na(withNA.df))

newdata <- createNA(data = iris.binary,p = 0.2)
colSums(is.na(newdata))

#
#type null
torch::torch_manual_seed(1234)
imputed.data <- midae(data = withNA.df,
                      pmm.type = NULL, pmm.k = 5, pmm.link="prob", pmm.save.vars = NULL,
                      m = 5, epochs = 5,
                      input.dropout = 0, hidden.dropout = 0,loss.na.scale = TRUE,
                      save.model = TRUE, path = file.path(tempdir(), "midaemodel.pt"))
newimpute<-impute_new(object=imputed.data,newdata=newdata)
newimpute

#type 0

torch::torch_manual_seed(1234)
imputed.data0 <- midae(data = withNA.df,
                       pmm.type = 0, pmm.k = 5, pmm.link="logit", pmm.save.vars = NULL,
                       m = 5, epochs = 5,
                       input.dropout = 0, hidden.dropout = 0,loss.na.scale = TRUE,
                       save.model = TRUE, path = file.path(tempdir(), "midaemodel.pt"))

object=imputed.data0
newimpute<-impute_new(object=imputed.data0,newdata=newdata)

torch::torch_manual_seed(1234)
imputed.data0 <- midae(data = withNA.df,
                       pmm.type = 0, pmm.k = 5, pmm.link="logit", pmm.save.vars = colnames(withNA.df),
                       m = 5, epochs = 5,
                       input.dropout = 0, hidden.dropout = 0,loss.na.scale = TRUE,
                       save.model = TRUE, path = file.path(tempdir(), "midaemodel.pt"))

object=imputed.data0


object$params$yhatobs.list
newimpute<-impute_new(object=imputed.data0,newdata=newdata)

#type 1
torch::torch_manual_seed(1234)
imputed.data01 <- midae(data = withNA.df,
                        pmm.type = 1, pmm.k = 5, pmm.link="logit", pmm.save.vars = NULL,
                        m = 5, epochs = 5,
                        input.dropout = 0.2, hidden.dropout = 0,loss.na.scale = TRUE,
                        save.model = TRUE, path = file.path(tempdir(), "midaemodel.pt"))

object=imputed.data01

newimpute<-impute_new(object=imputed.data01,newdata=newdata)

object$params
torch::torch_manual_seed(1234)
imputed.data01 <- midae(data = withNA.df,
                        pmm.type = 1, pmm.k = 5, pmm.link="logit", pmm.save.vars = colnames(withNA.df),
                        m = 5, epochs = 5,
                        input.dropout = 0.2, hidden.dropout = 0,loss.na.scale = TRUE,
                        save.model = TRUE, path = file.path(tempdir(), "midaemodel.pt"))

object=imputed.data01

newimpute<-impute_new(object=imputed.data01,newdata=newdata)
object$params$yhatobs.list




object$params
newdata
pmm.k = NULL
m = NULL
verbose = FALSE

#type 2
torch::torch_manual_seed(1234)
imputed.data2 <- midae(data = withNA.df,
                       pmm.type = 2, pmm.k = 5, pmm.link="logit", pmm.save.vars = colnames(withNA.df),
                       m = 5, epochs = 5,
                       input.dropout = 0.2, hidden.dropout = 0.5,
                       save.model = TRUE, path = file.path(tempdir(), "midaemodel.pt"))

object=imputed.data2

newimpute<-impute_new(object=imputed.data2,newdata=newdata)
object$params$yhatobs.list

newimpute

#type auto
torch::torch_manual_seed(1234)
imputed.dataA <- midae(data = withNA.df,
                       pmm.type = "auto", pmm.k = 5, pmm.link="logit", pmm.save.vars = colnames(withNA.df),
                       m = 5, epochs = 5,
                       input.dropout = 0.2, hidden.dropout = 0.5,
                       save.model = TRUE, path = file.path(tempdir(), "midaemodel.pt"))

imputed.dataA$params$yhatobs.list

newimpute<-impute_new(object=imputed.dataA,newdata=newdata)



############ old data and new data have same number of missing variables
set.seed(2023)
iris.binary<-iris
iris.binary$Gender<-sample(c("F","M"),size=nrow(iris),replace = T)
iris.binary$Gender<-as.factor(iris.binary$Gender)
str(iris.binary)



withNA.df<-createNA(data = iris.binary,p = 0.2)


newdata <- createNA(data = iris.binary,p = 0.2)
colSums(is.na(newdata))



#


newdata
pmm.k = NULL
m = NULL
verbose = FALSE

#type null
torch::torch_manual_seed(1234)
imputed.data <- midae(data = withNA.df,
                        pmm.type = NULL, pmm.k = 5, pmm.link="prob", pmm.save.vars = NULL,
                        m = 5, epochs = 5,
                        input.dropout = 0, hidden.dropout = 0,
                        save.model = TRUE, path = file.path(tempdir(), "midaemodel.pt"))
newimpute<-impute_new(object=imputed.data,newdata=newdata)
newimpute




#type 1
torch::torch_manual_seed(1234)
imputed.data01 <- midae(data = withNA.df,
                        pmm.type = 1, pmm.k = 5, pmm.link="logit", pmm.save.vars = NULL,
                        m = 5, epochs = 5,
                        input.dropout = 0.2, hidden.dropout = 0,
                        save.model = TRUE, path = file.path(tempdir(), "midaemodel.pt"))



newimpute<-impute_new(object=imputed.data01,newdata=newdata)





#type 0
torch::torch_manual_seed(1234)
imputed.data0 <- midae(data = withNA.df,
                       pmm.type = 0, pmm.k = 5, pmm.link="logit", pmm.save.vars = NULL,
                       m = 5, epochs = 5,
                       input.dropout = 0, hidden.dropout = 0,
                       save.model = TRUE, path = file.path(tempdir(), "midaemodel.pt"))

object=imputed.data0

newimpute<-impute_new(object=imputed.data0,newdata=newdata)






#type 2
torch::torch_manual_seed(1234)
imputed.data2 <- midae(data = withNA.df,
                       pmm.type = 2, pmm.k = 5, pmm.link="logit", pmm.save.vars = NULL,
                       m = 5, epochs = 5,
                       input.dropout = 0.2, hidden.dropout = 0.5,
                       save.model = TRUE, path = file.path(tempdir(), "midaemodel.pt"))

object=imputed.data2

newimpute<-impute_new(object=imputed.data2,newdata=newdata)

object$params$yhatobs.list
newimpute

#type auto
torch::torch_manual_seed(1234)
imputed.dataA <- midae(data = withNA.df,
                       pmm.type = "auto", pmm.k = 5, pmm.link="logit", pmm.save.vars = NULL,
                       m = 5, epochs = 5,
                       input.dropout = 0.2, hidden.dropout = 0.5,
                       save.model = TRUE, path = file.path(tempdir(), "midaemodel.pt"))



newimpute<-impute_new(object=imputed.dataA,newdata=newdata)

object=imputed.dataA
object$params$yhatobs.list
newimpute


# test --------------------------------------------------------------------
###pmm0

torch::torch_manual_seed(1234)
imputed.data <- midae(data = withNA.df,
                      pmm.type = 0, pmm.k = 5,  pmm.link="logit", pmm.save.vars = c("Sepal.Length","Sepal.Width","Petal.Length",  "Petal.Width","Species"),
                      m = 5, epochs = 5,
                      input.dropout = 0, hidden.dropout = 0,
                      save.model = FALSE, path = file.path(tempdir(), "midaemodel.pt"))

torch::torch_manual_seed(1234)
imputed.data01 <- midae(data = withNA.df,
                        pmm.type = 0, pmm.k = 5, pmm.link="logit", pmm.save.vars = NULL,
                        m = 5, epochs = 5,
                        input.dropout = 0, hidden.dropout = 0,
                        save.model = FALSE, path = file.path(tempdir(), "midaemodel.pt"))


torch::torch_manual_seed(1234)
imputed.data01 <- midae(data = withNA.df,
                        pmm.type = "auto", pmm.k = 5, pmm.link="logit", pmm.save.vars = NULL,
                        m = 5, epochs = 5,
                        input.dropout = 0, hidden.dropout = 0,
                        save.model = TRUE, path = file.path(tempdir(), "midaemodel.pt"))

object=imputed.data01

###pmm0
torch::torch_manual_seed(1234)
imputed.data02 <- midae(data = withNA.df,
                        pmm.type = 0, pmm.k = 5, pmm.link="prob", pmm.save.vars = NULL,
                        m = 5, epochs = 5,
                        input.dropout = 0, hidden.dropout = 0,
                        save.model = FALSE, path = file.path(tempdir(), "midaemodel.pt"))

#even the loss is the same, but match with different values....imputations are different:)

show_var(imputation.list = imputed.data01,var.name="Gender",original.data = withNA.df)
show_var(imputation.list = imputed.data02,var.name="Gender",original.data = withNA.df)

torch::torch_manual_seed(1234)
imputed.data <- midae(data = withNA.df,
                      pmm.type = 1, pmm.k = 5, pmm.link="prob", pmm.save.vars = NULL,
                      m = 5, epochs = 5,
                      input.dropout = 0.2, hidden.dropout = 0.5,
                      save.model = FALSE, path = file.path(tempdir(), "midaemodel.pt"))


torch::torch_manual_seed(1234)
imputed.data <- midae(data = withNA.df,
                      pmm.type = 2, pmm.k = 5, pmm.link="prob", pmm.save.vars = NULL,
                      m = 5, epochs = 5,
                      input.dropout = 0.2, hidden.dropout = 0.5,
                      save.model = FALSE, path = file.path(tempdir(), "midaemodel.pt"))


torch::torch_manual_seed(1234)
imputed.data <- midae(data = withNA.df,
                      pmm.type = "auto", pmm.k = 5, pmm.link="prob", pmm.save.vars = NULL,
                      m = 5, epochs = 5,
                      input.dropout = 0.2, hidden.dropout = 0.5,
                      save.model = FALSE, path = file.path(tempdir(), "midaemodel.pt"))


torch::torch_manual_seed(1234)
imputed.data <- midae(data = withNA.df,
                      pmm.type = NULL, pmm.k = 5, pmm.link="prob", pmm.save.vars = NULL,
                      m = 5, epochs = 5,
                      input.dropout = 0.2, hidden.dropout = 0.5,
                      save.model = FALSE, path = file.path(tempdir(), "midaemodel.pt"))

# test end ----------------------------------------------------------------



withNA.df
data=withNA.df




m = 5
pmm.k = 5
pmm.save.vars = NULL
epochs = 5
batch.size = 32
shuffle = TRUE
optimizer = "adamW"
learning.rate = 0.0001
weight.decay = 0.002
momentum = 0
eps = 1e-07
encoder.structure = c(128, 64, 32)
decoder.structure = c(32, 64, 128)
act = "elu"
init.weight = "xavier.normal"
scaler = "none"
loss.na.scale = FALSE
verbose = TRUE
print.every.n = 1
save.model = FALSE
path = file.path(tempdir(), "midaemodel.pt")


pmm.link="prob"
#pmm.type=0
pmm.type = 1
subsample = 1
input.dropout = 0.2
hidden.dropout = 0.5


data[[var]][na.loc[, var]]
data[[var]]
levels(data[[var]])
levels(sorted.dt[[var]])[3]


pmm.multiclass(yhatobs = yhatobs, yhatmis = yhatmis, yobs = obs.y, k = pmm.k)
nhanes3


data[[var]][na.loc[, var]] <- pmm.multiclass(yhatobs = yhatobs, yhatmis = yhatmis, yobs = obs.y, k = pmm.k)


library(mixgb)

mixgb.data<-mixgb(data=withNA.df,pmm.type = 0)





set.seed(2023)
withNA.df <- createNA(data = iris, p = 0.2)
colSums(is.na(withNA.df))

torch::torch_manual_seed(1234)
imputed.data <- midae(data = withNA.df,
                      pmm.type = "auto", pmm.k = 5, pmm.save.vars = NULL,
                      m = 5, epochs = 5, path = file.path(tempdir(), "midaemodel.pt"))


torch::torch_manual_seed(1234)
imputed.data <- midae(data = withNA.df,
                      pmm.type = "auto", pmm.k = 5, pmm.save.vars = c("Sepal.Length","Sepal.Width","Petal.Length",  "Petal.Width","Species"),
                      m = 5, epochs = 5, path = file.path(tempdir(), "midaemodel.pt"))




set.seed(2023)
withNA.df <- createNA(data = iris, var.names = c("Sepal.Length","Species"), p = 0.2)
colSums(is.na(withNA.df))

torch::torch_manual_seed(1234)
imputed.data <- midae(data = withNA.df,
                      pmm.type = "auto", pmm.k = 5, pmm.save.vars = NULL,
                      m = 5, epochs = 5, path = file.path(tempdir(), "midaemodel.pt"))



###pmm0
torch::torch_manual_seed(1234)
imputed.data <- midae(data = withNA.df,
                      pmm.type = 0, pmm.k = 5, pmm.save.vars = c("Sepal.Length","Sepal.Width","Petal.Length",  "Petal.Width","Species"),
                      m = 5, epochs = 5,
                      input.dropout = 0, hidden.dropout = 0,
                      save.model = FALSE, path = file.path(tempdir(), "midaemodel.pt"))



torch::torch_manual_seed(1234)
imputed.data <- midae(data = withNA.df,
                      pmm.type = 0, pmm.k = 5, pmm.save.vars = c("Sepal.Length","Sepal.Width","Petal.Length",  "Petal.Width","Species"),
                      m = 5, epochs = 5, save.model = FALSE, path = file.path(tempdir(), "midaemodel.pt"))
