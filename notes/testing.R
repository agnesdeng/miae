setwd("C:/Users/agnes/Desktop/phd-thesis/my-packages/miae")

library(devtools)
devtools::document()
devtools::load_all()
library(tidyverse)
library(MASS)
library(gridExtra)


b=80
setwd("C:/Users/agnes/Desktop/phd-thesis/my-projects/miae-paper/midas/mar1")

withNA.df<-read_csv(file = paste0("data_base/mar1_draw_",b,".csv"))
full.df<-read_csv(file = paste0("data_base/mar1_full_",b,".csv"))

midae.data<-midae0(data=withNA.df,m=10,epochs = 5,batch.size = 16,
                   input.dropout = 0.2, hidden.dropout = 0.5,
                   optimizer = "adamW", learning.rate = 0.0001, weight.decay = 0.002, momentum = 0,
                   encoder.structure = c(256,256,256), decoder.structure = c(256,256,256),
                   act = "elu", init.weight="xavier.midas", scale.numeric = FALSE,
                   verbose = TRUE, print.every.n = 1, save.model = FALSE, path = NULL)

plot_density(imputation.list = midae.data,var.name = "X1",original.data = miss,true.data = D)

p1<-plot_density(imputation.list = midae.data ,var.name = "Y",original.data =withNA.df,true.data=full.df )
p2<-plot_density(imputation.list = midae.data ,var.name = "X1",original.data =withNA.df ,true.data=full.df )
p3<-plot_density(imputation.list =midae.data ,var.name = "X2",original.data =withNA.df,true.data=full.df )
p4<-plot_density(imputation.list =midae.data ,var.name = "X4",original.data =withNA.df,true.data=full.df )

grid.arrange(p1, p2, p3,p4,ncol=2,top="midas-settings")

model.form=formula("Y ~ X1 + X2")




data=withNA.df
m=10
epochs = 5
batch.size = 5
input.dropout = 0.2
hidden.dropout = 0.5
optimizer = "adamW"
learning.rate = 0.001
weight.decay = 0.002
momentum = 0
encoder.structure = c(10,20,30)
decoder.structure = c(30,20,10)
#encoder.structure = c(256,256,256)
#decoder.structure = c(256,256,256)
act = "elu"
init.weight="xavier.midas"
scale.numeric = F


pre.obj <- preprocess(data,scale.numeric = scale.numeric)

torch.data <- torch_dataset(data,scale.numeric = scale.numeric)

#torch.data$torch.data

n.features <- torch.data$.ncol()

n.samples <- torch.data$.length()

dl <- torch::dataloader(dataset = torch.data, batch_size = batch.size, shuffle = TRUE)
###
#train.idx <- sample(1:n.samples, size = floor(0.7*n.samples), replace = FALSE)
#valid.idx <- setdiff(1:n.samples, train.idx)

#train.ds <- torch_dataset_idx(data,train.idx)
#valid.ds <- torch_dataset_idx(data,valid.idx)

#train.dl<- dataloader(dataset = train.ds,batch_size = batch.size, shuffle = TRUE)
# valid.dl<- dataloader(dataset = valid.ds,batch_size = batch.size, shuffle = FALSE)

# train.size <- length(train.ds)
#valid.size <- length(valid.ds)
###

#dl <- torch::dataloader(dataset = torch.data, batch_size = batch.size, shuffle = TRUE)
model <- dae(n.features = n.features, input.dropout = input.dropout, hidden.dropout = hidden.dropout, encoder.structure = encoder.structure, decoder.structure = encoder.structure, act = act)

model$parameters$encoder.0.weight

if(init.weight=="xavier.midas"){
  model$apply(init_xavier_midas)
}

model$parameters$encoder.0.weight
# define the loss function for different variables
num_loss <- torch::nn_mse_loss(reduction = "mean")
bin_loss <- torch::nn_bce_with_logits_loss(reduction = "mean")
multi_loss <- torch::nn_cross_entropy_loss(reduction = "mean")


# choose optimizer & learning rate
if(optimizer=="adam"){
  optimizer <- torch::optim_adam(model$parameters, lr = learning.rate, weight_decay = weight.decay)
}else if(optimizer=="sgd"){
  optimizer <- torch::optim_sgd(model$parameters, lr = learning.rate, momentum = momentum, weight_decay = weight.decay)
}else if(optimizer=="adamW"){
  optimizer <- torchopt::optim_adamw(model$parameters, lr = learning.rate, weight_decay = weight.decay)
}



# epochs: number of iterations
epoch=1



  iters<-torch::dataloader_make_iter(dl)
  b1<-torch::dataloader_next(iters)
  b2<-torch::dataloader_next(iters)
  b3<-torch::dataloader_next(iters)
  b4<-torch::dataloader_next(iters)
  b5<-torch::dataloader_next(iters)





for (epoch in seq_len(epochs)) {

  model$train()

  train.loss <- 0



  coro::loop(for (b in dl) { # loop over all batches in each epoch

    b=b1

    #zero out the gradients
    optimizer$zero_grad()

    Out <- model(b$data)

    model$parameters$encoder.0.weight
    model$parameters$encoder.0.weight$grad
    # numeric
    num.cost <- vector("list", length = length(pre.obj$num))
    names(num.cost) <- pre.obj$num

    for (var in pre.obj$num){
      #which(pre.obj$na.loc[,var][as.array(b$index)]!=TRUE)
      obs.idx<-which(pre.obj$na.loc[as.array(b$index),var]!=TRUE)
      #obs.idx<-which(pre.obj$na.loc[as.array(b$index),var]!=FALSE)
     # obs.idx :  integer(0)
      #num.cost[[var]] <- torch_sqrt(num_loss(input = Out[obs.idx, pre.obj$num.idx[[var]]], target = b$data[obs.idx, pre.obj$num.idx[[var]]]))
      num.cost[[var]] <- torch_sqrt(num_loss(input = Out[obs.idx, pre.obj$num.idx[[var]],drop=FALSE], target = b$data[obs.idx, pre.obj$num.idx[[var]],drop=FALSE]))
    }

    total.num.cost <- do.call(sum, num.cost)

    # binary
    bin.cost <- vector("list", length = length(pre.obj$bin))
    names(bin.cost) <- pre.obj$bin

    for (var in pre.obj$bin) {
      obs.idx<-which(pre.obj$na.loc[as.array(b$index),var]!=TRUE)
      bin.cost[[var]] <- bin_loss(input = Out[obs.idx, pre.obj$bin.idx[[var]]], target = b$data[obs.idx, pre.obj$bin.idx[[var]]])
    }

    total.bin.cost <- do.call(sum, bin.cost)

    # multiclass
    multi.cost <- vector("list", length = length(pre.obj$multi))
    names(multi.cost) <- pre.obj$multi

    for (var in pre.obj$multi) {
      obs.idx<-which(pre.obj$na.loc[as.array(b$index),var]!=TRUE)
      multi.cost[[var]] <- multi_loss(input = Out[obs.idx, pre.obj$multi.idx[[var]]], target = torch::torch_argmax(b$data[obs.idx, pre.obj$multi.idx[[var]]], dim = 2))
    }
    total.multi.cost <- do.call(sum, multi.cost)

    # Total cost
   # cost<-total.num.cost+total.bin.cost+total.multi.cost
    cost <- sum(total.num.cost, total.bin.cost, total.multi.cost)



    cost$backward()


    model$parameters$encoder.0.weight
    model$parameters$encoder.0.weight$grad

    #update params
    optimizer$step()

    model$parameters$encoder.0.weight
    model$parameters$encoder.0.weight$grad


    batch.loss <- cost$item()
    train.loss <- train.loss + batch.loss

    if (save.model & epoch == epochs) {
      torch::torch_save(model, path = path)
    }
  })





  if(verbose & (epoch ==1 | epoch %% print.every.n == 0)){
    cat(sprintf("Loss at epoch %d: %1f\n", epoch, train.loss / length(dl)))
  }


}



#model <- torch::torch_load(path = path)
model$eval()


# The whole dataset
eval_dl <- torch::dataloader(dataset = torch.data, batch_size = n.samples, shuffle = FALSE)


wholebatch <- eval_dl %>%
  torch::dataloader_make_iter() %>%
  torch::dataloader_next()

# imputed data
imputed.data <- vector("list", length = m)
na.loc <- pre.obj$na.loc

for (i in seq_len(m)) {
  output.data <- model(wholebatch$data)
  imp.data <- postprocess(output.data = output.data, pre.obj = pre.obj,scale.numeric = scale.numeric)
  na.vars <- pre.obj$ordered.names[colSums(na.loc) != 0]

  for (var in na.vars) {

    data[[var]][na.loc[, var]] <- imp.data[[var]][na.loc[, var]]

  }

  imputed.data[[i]] <- data
}
imputed.data
