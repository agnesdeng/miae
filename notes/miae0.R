
miae(data=withNA.df,m=5,epochs=2)


mnist_dataset <- dataset(
  
  name = "mnist_dataset",
  initialize = function() {
    self$data <- self$mnist_data()
  },
  
  .getitem = function(index) {
    x <- self$data[index, ]
    
    x
  },
  .length = function() {
    self$data$size()[[1]]
  },
  
  
  mnist_data = function() {
    #784 inputs (pixel 28 x 28, each value between 0-255, normalise them)
    input <- torch_tensor(mnist$train$images / 255)
    input
  }
)

# Dataloader
mnist.data<-mnist_dataset()



miae<-function(mnist.data,m=5,epochs=5,latent.dim=16,learning.rate=0.001, batch.size=2000){
  

  dl <- dataloader(mnist.data, batch_size = batch.size, shuffle = TRUE)
  #length(dl)
  #300 batches, each with size 200
  #batch <- dataloader_make_iter(dl) %>% dataloader_next()
 # dim(batch)
  
  
  
  # Set VAEs latent dimension
  
  
  
 
  
  model=ae(n_features = 28*28,latent_dim=latent.dim)
  

  optimizer <- optim_adam(model$parameters, lr = learning.rate)
  loss.fn<-nn_mse_loss(reduction="mean")
  
  for(epoch in seq_len(epochs)){
    
    losses<-0
    
    coro::loop(for (b in dl){
      #forward
      output<-model(b)
      loss<-loss.fn(output,b)
      #backward
      optimizer$zero_grad()
      loss$backward()
      optimizer$step()
      batch_loss<-loss$item()
      losses<-losses+batch_loss
      #cat(sprintf("Loss at epoch %d: %1f\n", epoch, losses)) 
    })
    
    #print message
    cat(sprintf("Loss at epoch %d: %1f\n", epoch, losses/length(dl))) 
    
    
  }
  
  
}