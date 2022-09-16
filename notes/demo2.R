library(dplyr)
library(palmerpenguins)

library(devtools)
devtools::document()
devtools::load_all()

original.data<-penguins
pre.obj<-preprocess(data=penguins)


torch.data<-torch_dataset(data=penguins)


n.features<-torch.data$.ncol()
n.samples<-torch.data$.length()

dl<-torch::dataloader(dataset=torch.data,batch_size = 20,shuffle=TRUE)





#simularly we have
#dae <-....
#vae <-....

# Dataloader
model <- ae(n.features=n.features,latent.dim=16)


#define the loss function for different variable
#num_loss = torch::nn_mse_loss(reduction = "mean")
num_loss = torch::nn_mse_loss(reduction = "sum")
#bin_loss = torch::nn_bce_loss()
bin_loss = torch::nn_bce_with_logits_loss()
multi_loss=torch::nn_cross_entropy_loss()



##choose optimizer & learning rate
optimizer <- torch::optim_adam(model$parameters, lr = 0.001)

##test
#The whole dataset
eval_dl<-torch::dataloader(dataset=torch.data,batch_size = n.samples, shuffle = TRUE)
eval_dl

wholebatch <- eval_dl %>% torch::dataloader_make_iter() %>% torch::dataloader_next()
dim(wholebatch)

b=wholebatch
#set number of iterations
epochs <- 5


for(epoch in 1:epochs) {

  epoch.loss<-0

  coro::loop(for (b in dl) {  # loop over all minibatches for one epoch

    Out = model(b)


    pre.obj$bin.idx
    pre.obj$multi.idx

    #numeric
    num.cost<-num_loss(input=Out[,pre.obj$num.idx],target=b[,pre.obj$num.idx])
    total.num.cost<-num.cost/batch.size

    #binary
    bin.cost<-vector("list",length=length(pre.obj$bin))
    names(bin.cost)<-pre.obj$bin

    for(var in pre.obj$bin){
      bin.cost[[var]]<-bin_loss(input=Out[,pre.obj$bin.idx[[var]]],target=b[,pre.obj$bin.idx[[var]]])
    }

    total.bin.cost<-do.call(sum,bin.cost)

    #multiclass
    multi.cost<-vector("list",length=length(pre.obj$multi))
    names(multi.cost)<-pre.obj$multi

    for(var in pre.obj$multi){
      multi.cost[[var]]<-multi_loss(input=Out[,pre.obj$multi.idx[[var]]],target=torch_argmax(b[,pre.obj$multi.idx[[var]]],dim=2))
    }
    total.multi.cost<-do.call(sum,multi.cost)

    cost<-sum(total.num.cost,total.bin.cost,total.multi.cost)
    #cont.cost<-cont_loss(input=Out[,1:5],target=b[,1:5])
    #bin.cost<- bin_loss(input=Out[,12:13],target=b[,12:13])
    #multi.cost<-sum(multi_loss(input=Out[,6:8],target=torch_argmax(b[,6:8],dim=2)),
                    #multi_loss(input=Out[,9:11],target=torch_argmax(b[,9:11],dim=2)))

    #cost=cont.cost+bin.cost+multi.cost

    #
    optimizer$zero_grad()
    cost$backward()
    optimizer$step()


    batch.loss<-cost$item()
    epoch.loss<-epoch.loss+batch.loss

    if(epoch==epochs){
      #torch_save(model,path="C:/Users/agnes/Desktop/torch")
      torch_save(model, paste0("model_", epoch, ".pt"))
    }

  })

  #cat(sprintf("Loss at epoch %d: %1f\n", epoch, 128*l/60000))
  cat(sprintf("Loss at epoch %d: %1f\n", epoch, epoch.loss/length(dl)))


}

last.model<-paste0(paste0("model_",epochs),".pt")
saved_model <- torch_load(last.model)

model <- saved_model
model$eval()

#The whole dataset
eval_dl<-dataloader(dataset=torch.data,batch_size = n.samples, shuffle = TRUE)
eval_dl

wholebatch <- eval_dl %>% dataloader_make_iter() %>% dataloader_next()
dim(wholebatch)


m=5

imputed.data<-vector("list",length=m)

output.data1<-model(wholebatch)
output.data2<-model(wholebatch)
torch_equal(output.data1,output.data2)

#identical(output.data1,output.data2)
output.data1[c(4,272),1]
output.data2[c(4,272),1]


imp.data1<-postprocess(output.data = output.data1,pre.obj = pre.obj)
imp.data2<-postprocess(output.data = output.data2,pre.obj = pre.obj)
identical(imp.data1,imp.data2)

for(i in seq_len(m)){
  output.data<-model(wholebatch)
  imp.data<-postprocess(output.data = output.data,pre.obj = pre.obj)
  na.vars<-pre.obj$ordered.names[colSums(pre.obj$na.loc)!=0]

  for(var in na.vars){
    #tibble
    original.data[[var]][na.loc[,var]]<-imp.data[[var]][na.loc[,var]]
  }

  imputed.data[[i]]<-original.data

}


colSums(is.na(penguins))
show_var(imputation.list=imputed.data,var.name="bill_length_mm",original.data = penguins)
show_var(imputation.list=imputed.data,var.name="bill_depth_mm",original.data = penguins)
show_var(imputation.list=imputed.data,var.name="flipper_length_mm",original.data = penguins)
show_var(imputation.list=imputed.data,var.name="body_mass_g",original.data = penguins)
show_var(imputation.list=imputed.data,var.name="sex",original.data = penguins)










#What else?
#(1) need to mark the location of NAs
#(0) scaling
#(2) only calculate the loss of observed data
#(3) different variant of AE:  denoise with dropout / variational
output.data2=model(wholebatch)
output.data2
identical(output.data,output.data2)

#(4) issues so far: using sum of con_loss+cat_loss is not very good

