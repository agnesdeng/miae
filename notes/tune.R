
library(devtools)
devtools::document()
devtools::load_all()

withNA.df<-createNA(data=iris,p=0.3)

colnames(iris)

dropout.grid<-list(input.dropout=c(0.1,0.3,0.5,0.7),hidden.dropout=c(0.1,0.3,0.5,0.7))

tune.results<-tune_dae_dropout(data=withNA.df,m=5, epochs = 5, batch.size = 50,
                           dropout.grid = dropout.grid, latent.dropout = 0)

plot_dropout(tune.results = tune.results,var.name="Sepal.Length")
plot_dropout(tune.results = tune.results,var.name="Sepal.Width")
plot_dropout(tune.results = tune.results,var.name="Petal.Length")
plot_dropout(tune.results = tune.results,var.name="Sepal.Width")


tune.results<-tune_vae_dropout(data=withNA.df,m=5, epochs = 5, batch.size = 50,
                               dropout.grid = dropout.grid, latent.dropout = 0)

plot_dropout(tune.results = tune.results,var.name="Sepal.Length")
plot_dropout(tune.results = tune.results,var.name="Sepal.Width")
plot_dropout(tune.results = tune.results,var.name="Petal.Length")
plot_dropout(tune.results = tune.results,var.name="Sepal.Width")


library(palmerpenguins)
withNA.df<-createNA(data=penguins,p=0.3)

dropout.grid<-list(input.dropout=c(0.1,0.3,0.5,0.7),latent.dropout=c(0.1,0.3,0.5,0.7))

tune.results<-tune_dropout(data=withNA.df,m=5, epochs = 20, batch.size = 50,
                           dropout.grid = dropout.grid)

colnames(withNA.df)
range(withNA.df$bill_length_mm,na.rm=T)
range(withNA.df$bill_depth_mm,na.rm=T)
range(withNA.df$body_mass_g,na.rm=T)
plot_dropout(tune.results = tune.results,var.name="bill_length_mm")
plot_dropout(tune.results = tune.results,var.name="bill_depth_mm")
plot_dropout(tune.results = tune.results,var.name="flipper_length_mm")
plot_dropout(tune.results = tune.results,var.name="body_mass_g")






data=withNA.df
m=5
epochs = 5
batch.size = 50
latent.dropout = 0
dropout.grid = dropout.grid
optim = "adam"
learning.rate = 0.001
weight.decay = 0
momentum = 0
encoder.structure = c(128, 64, 32)
latent.dim = 8
decoder.structure = c(32, 64, 128)
verbose = TRUE
print.every.n = 1



tune.results<-tune_dropout(data=withNA.df,m=5, epochs = 20, batch.size = 10,
                           dropout.grid = dropout.grid)

tune.results$imputed.missing
tune.results$observed.data

var.name<-"Sepal.Length"
library(ggplot2)

plot_dropoutvar<-function(tune.results, var.name){

  longer.tbl<-lapply(tune.results[[var.name]],tidyr::pivot_longer,cols = starts_with("m"),names_to="set")
   all.tbl<-do.call(rbind,longer.tbl)

   all.tbl <- all.tbl %>%
     dplyr::mutate_at(vars(-("value")),as.factor)

   ggplot(data=all.tbl,aes(x=value,color=set))+
     geom_density()+
     #facet_grid(latent.dropout~input.dropout,labeller=label_both)+
     facet_grid(latent.dropout~input.dropout)+
     scale_x_continuous(sec.axis = sec_axis(~ . , name = "Input Dropout", breaks = NULL, labels = NULL)) +
     scale_y_continuous(sec.axis = sec_axis(~ . , name = "Latent Dropout", breaks = NULL, labels = NULL))
   all.tbl$input.dropout
   all.tbl$latent.dropout

}




tune_dropout<-function(data, m=5, epochs = 10, batch.size = 50,
             hidden.dropout = 0,
             dropout.grid = dropout.grid,
             optim = "adam", learning.rate = 0.001, weight.decay = 0, momentum = 0,
             encoder.structure = c(128, 64, 32), latent.dim = 8, decoder.structure = c(32, 64, 128),
             verbose = TRUE, print.every.n = 1, path = NULL){


  if(is.null(path)){
    #stop("Please specify a path to save the imputation model.")
  }


  pre.obj <- preprocess(data)

  torch.data <- torch_dataset(data)


  n.features <- torch.data$.ncol()

  n.samples <- torch.data$.length()


  ###
  train.idx <- sample(1:n.samples, size = floor(0.7*n.samples), replace = FALSE)
  valid.idx <- setdiff(1:n.samples, train.idx)

  train.ds <- torch_dataset_idx(data,train.idx)
  valid.ds <- torch_dataset_idx(data,valid.idx)

  train.dl<- dataloader(dataset = train.ds,batch_size = batch.size, shuffle = TRUE)
  valid.dl<- dataloader(dataset = valid.ds,batch_size = batch.size, shuffle = FALSE)

  train.size <- length(train.ds)
  valid.size <- length(valid.ds)
  ###

  #dl <- torch::dataloader(dataset = torch.data, batch_size = batch.size, shuffle = TRUE)

  #model <- dae(n.features = n.features, latent.dim = latent.dim, input.dropout = input.dropout, latent.dropout = latent.dropout, hidden.dropout = hidden.dropout, encoder.structure = encoder.structure, decoder.structure = encoder.structure)


  # define the loss function for different variables
  num_loss <- torch::nn_mse_loss(reduction = "sum")
  bin_loss <- torch::nn_bce_with_logits_loss(reduction = "sum")
  multi_loss <- torch::nn_cross_entropy_loss(reduction = "sum")




  # epochs: number of iterations
  model.params<-expand.grid(dropout.grid)
  n.models<-nrow(model.params)


  Model.list<-vector("list", length=n.models)

  #pre-allocate
  #imputed.missing<-replicate(n.models, list())

  na.loc <- pre.obj$na.loc
  na.vars <- pre.obj$ordered.names[colSums(na.loc) != 0]
  num.navars<-length(na.vars)


  imputed.missing<-replicate(num.navars, list())
  names(imputed.missing)<-na.vars

  for (var in na.vars) {
    imputed.missing[[var]]<- vector("list", length = n.models)
    n.na<-sum(na.loc[,var])

    for(i in seq_len(n.models)){
      imputed.missing[[var]][[i]]<-data.frame(matrix(NA,nrow=n.na,ncol=m))

      colnames(imputed.missing[[var]][[i]])<-paste0("m",1:m)
    }

  }







  #run models
  for(i in seq_len(n.models)){

    model <-dae(n.features = n.features, latent.dim = latent.dim, input.dropout = model.params$input.dropout[i], latent.dropout = model.params$latent.dropout[i], hidden.dropout = hidden.dropout, encoder.structure = encoder.structure, decoder.structure = encoder.structure)

     # choose optimizer & learning rate
    if(optim=="adam"){
      optimizer <- torch::optim_adam(model$parameters, lr = learning.rate, weight_decay = weight.decay)
      }else if(optim=="sgd"){
      optimizer <- torch::optim_sgd(model$parameters, lr = learning.rate, momentum = momentum, weight_decay = weight.decay)
    }






    for (epoch in seq_len(epochs)) {
      model$train()

      train.loss <- 0



      coro::loop(for (b in train.dl) { # loop over all batches in each epoch

        Out <- model(b$data)

        # numeric
        num.cost <- vector("list", length = length(pre.obj$num))
        names(num.cost) <- pre.obj$num

        for (var in pre.obj$num){
          obs.idx<-which(pre.obj$na.loc[as.array(b$index),var]!=TRUE)
          num.cost[[var]] <- num_loss(input = Out[obs.idx, pre.obj$num.idx[[var]]], target = b$data[obs.idx, pre.obj$num.idx[[var]]])
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
        cost <- sum(total.num.cost, total.bin.cost, total.multi.cost)

        #zero out the gradients
        optimizer$zero_grad()

        cost$backward()

        #update params
        optimizer$step()


        batch.loss <- cost$item()
        train.loss <- train.loss + batch.loss

        #if (epoch == epochs) {
         # torch::torch_save(model, path = path)
        #}
      })


      model$eval()
      valid.loss<-0

      #validation loss
      coro::loop(for (b in valid.dl){
        Out <- model(b$data)

        # numeric
        num.cost <- vector("list", length = length(pre.obj$num))
        names(num.cost) <- pre.obj$num

        for (var in pre.obj$num){
          obs.idx<-which(pre.obj$na.loc[as.array(b$index),var]!=TRUE)
          num.cost[[var]] <- num_loss(input = Out[obs.idx, pre.obj$num.idx[[var]]], target = b$data[obs.idx, pre.obj$num.idx[[var]]])
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
        cost <- sum(total.num.cost, total.bin.cost, total.multi.cost)


        batch.loss <- cost$item()
        valid.loss <- valid.loss + batch.loss





      })



      if(verbose & (epoch ==1 | epoch %% print.every.n == 0)){
        cat(sprintf("Loss at epoch %d: training: %3f, validation: %3f\n", epoch, train.loss / train.size, valid.loss / valid.size))
      }

      #model <- torch::torch_load(path = path)

    }
      # The whole dataset
      eval_dl <- torch::dataloader(dataset = torch.data, batch_size = n.samples, shuffle = FALSE)


      wholebatch <- eval_dl %>%
        torch::dataloader_make_iter() %>%
        torch::dataloader_next()



      # imputed data
      Model.list[[i]]<-model




      for (j in seq_len(m)) {
        output.data <- model(wholebatch$data)
        imp.data <- postprocess(output.data = output.data, pre.obj = pre.obj)
        for (var in na.vars) {
          imputed.missing[[var]][[i]][,j]<- imp.data[[var]][na.loc[, var]]
        }
      }


      for(var in na.vars){
        K<-nrow(imputed.missing[[var]][[i]])
        imputed.missing[[var]][[i]]<-imputed.missing[[var]][[i]] %>%
        tibble::add_column(input.dropout=rep(model.params$input.dropout[i],K),
                          latent.dropout=rep(model.params$latent.dropout[i],K)
                          )
      }




    }#end of n models




    observed.data<-vector("list",length=num.navars)
    names(observed.data)<-na.vars

    for(var in na.vars){
      observed.data[[var]]<-data[[var]][!is.na(data[[var]])]
    }

     return(list("imputed.missing"=imputed.missing,"observed.data"=observed.data))
  }










