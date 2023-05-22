
batch_set<-function(n.samples, batch.size, drop.last){


if(n.samples %% batch.size == 0){

  if(drop.last){
    drop.last <- FALSE
    warnings("Each batch has the same number of samples; no need to drop the last batch.")
  }

  num.batches <- n.samples/batch.size
  batch.set<-list()
  for(i in 1:num.batches){
    batch.set[[i]]<-(batch.size*(i-1) + 1):(i*batch.size)
  }


}else{

  if(drop.last){
    num.batches <-floor(n.samples/batch.size)
    batch.set<-list()
    if(num.batches>0){
      for(i in 1:num.batches){
        batch.set[[i]]<-(batch.size*(i-1) + 1):(i*batch.size)
      }
    }else{
      stop("There is only one batch in either training or valid set, and the size is smaller than batch.size. Can't drop the last batch. Please respecify either batch.size, subsample or drop.last.")
    }


  }else{
    num.batches <-ceiling(n.samples/batch.size)
    #the last batch is with smaller number of samples
    batch.set<-list()
    if(num.batches>1){
      for(i in 1:(num.batches-1)){
        batch.set[[i]]<-(batch.size*(i-1) + 1):(i*batch.size)
      }
      batch.set[[num.batches]]<-(batch.size*(num.batches-1) + 1):n.samples

    }else{
      batch.set[[num.batches]]<-1:n.samples
    }

  }


}

  return(list("batch.set"= batch.set,"num.batches" = num.batches))

}
