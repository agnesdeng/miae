library(dplyr)
library(palmerpenguins)

library(devtools)
devtools::document()
devtools::load_all()



torch.data<-torch_dataset(data=penguins)
length(torch.data)
ncol(torch.data)
n_features<-torch.data$.ncol()

dl<-dataloader(dataset=torch.data,batch_size = 20,shuffle=TRUE)

length(torch.data[1])
length(dl)





onehot(penguins)

preprocess(data=penguins)

penguins %>% glimpse()

penguins
minmax_scaler(penguins)

minmax_scaler(penguins %>% dplyr::select_if(is.numeric))

#penguins[,purrr::map_lgl(penguins,is.numeric)]


#order variable :  numeric > categorical > binary
ordered.names<-c("bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "year","species","island", "sex")



torch_dataset()

onehot(penguins)


library(torch)


#prepare the dataset
penguins_dataset <- dataset(

  name = "penguins_dataset",

  initialize = function(data) {

    data <- na.omit(data)

    # continuous input data (x_cont)
    x_cont <- data[ , c("bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "year")] %>%
      as.matrix()
    x_cont <- torch_tensor(x_cont)


    # categorical input data (x_cat)
    x_cat <- data[ , c("species","island", "sex")]

    #one-hot encoding (vs embeddings)
    species <- nnf_one_hot(torch_tensor(as.integer(x_cat$species)))
    island <- nnf_one_hot(torch_tensor(as.integer(x_cat$island)))
    sex <- nnf_one_hot(torch_tensor(as.integer(x_cat$sex)))


    #combine numeric data with one-hot encoded categorical variables
    onehot.data<-torch_cat(list(x_cont,species,island,sex),dim=2)
    self$onehot.data<-onehot.data


  },

  .getitem = function(index) {
    x <- self$onehot.data[index, ]

    x
  },

  .length = function() {
    self$onehot.data$size()[[1]]
  }

)

return(list("onehot.tensor"=onehot.tensor,
            "col.min"=col.min,"col.max"=col.max,
            "original.names"=original.names,
            "ordered.names"=ordered.names,
            "ordered.types"=ordered.types
))

ds=penguins_dataset(data=penguins)


onehot.tensor=torch_dataset(data=penguins)

#use dataloader to make mini batches
dl<-dataloader(data=ds,batch_size = 20, shuffle = TRUE)
dl
ds[1]



#build an autoencoders
ae <- nn_module(
  "autoencoder",
  initialize = function(n_features, latent_dim) {

    self$encoder <- nn_sequential(
      nn_linear(n_features, 128),
      nn_relu(),
      nn_linear(128, 64),
      nn_relu(),
      nn_linear(64, 32),
      nn_relu(),
      nn_linear(32, latent_dim),
      nn_relu(),
    )

    self$decoder <- nn_sequential(
      nn_linear(latent_dim, 32),
      nn_relu(),
      nn_linear(32, 64),
      nn_relu(),
      nn_linear(64, 128),
      nn_relu(),
      nn_linear(128, n_features),
      nn_sigmoid(),
    )


  },

  forward = function(x) {
    x<-self$encoder(x)
    x<-self$decoder(x)
    x
  }
)

#simularly we have
#dae <-....
#vae <-....

# Dataloader


#model=ae(n_features=ncol(penguins),encoder_structure=c(128,64,32),latent_dim=16,decoder_structure=c(32,64,128))
model=ae(n_features=13,latent_dim=16)


#define the loss function for different variable
cont_loss = nn_mse_loss()
bin_loss = nn_bce_loss()
multi_loss=nn_cross_entropy_loss()



##choose optimizer & learning rate
optimizer <- optim_adam(model$parameters, lr = 0.001)



#set number of iterations
epochs = 10


for(epoch in 1:epochs) {

  epoch.loss<-0

  coro::loop(for (b in dl) {  # loop over all minibatches for one epoch

    Out = model(b)


    cont.cost<-cont_loss(input=Out[,1:5],target=b[,1:5])
    bin.cost<- bin_loss(input=Out[,12:13],target=b[,12:13])
    multi.cost<-sum(multi_loss(input=Out[,6:8],target=torch_argmax(b[,6:8],dim=2)),
                    multi_loss(input=Out[,9:11],target=torch_argmax(b[,9:11],dim=2)))

    cost=cont.cost+bin.cost+multi.cost

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


saved_model <- torch_load("model_10.pt")

model <- saved_model
model$eval()

#The whole dataset
eval_dl<-dataloader(data=ds,batch_size = 333, shuffle = TRUE)
eval_dl

wholebatch <- eval_dl %>% dataloader_make_iter() %>% dataloader_next()
dim(wholebatch)


output.data=model(wholebatch)
output.data

torch_argmax(output.data[,6:8],dim=2)
torch_argmax(output.data[,9:11],dim=2)
torch_argmax(output.data[,12:13],dim=2)


imp.data<-cbind(as_array(output.data[,1:5]),
                as_array(torch_argmax(output.data[,6:8],dim=2)),
                as_array(torch_argmax(output.data[,9:10],dim=2)),
                as_array(torch_argmax(output.data[,12:13],dim=2))
)

colnames(imp.data)=ordered.names
class(imp.data)
imp.df<-as.data.frame(imp.data)
imp.df$species<-levels(penguins$species)[imp.df$species]
imp.df$island<-levels(penguins$island)[imp.df$island]
imp.df$sex<-levels(penguins$sex)[imp.df$sex]
imp.df

penguins

#What else?
#(1) need to mark the location of NAs
#(0) scaling
#(2) only calculate the loss of observed data
#(3) different variant of AE:  denoise with dropout / variational
output.data2=model(wholebatch)
output.data2
identical(output.data,output.data2)

#(4) issues so far: using sum of con_loss+cat_loss is not very good

