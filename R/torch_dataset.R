#' prepare the dataset
#' @importFrom torch dataset torch_tensor torch_cat nnf_one_hot
#' @export
torch_dataset <- torch::dataset(
  name = "torch_dataset",
  initialize = function(data,scaler,lower,upper,categorical.encoding, initial.imp) {
    pre.obj <- preprocess(data,scaler,lower,upper,categorical.encoding, initial.imp)

    #self$torch.data <- pre.obj$data.tensor
    self$num.tensor<-pre.obj$data.tensor$num.tensor
    self$logi.tensor<-pre.obj$data.tensor$logi.tensor
    self$bin.tensor<-pre.obj$data.tensor$bin.tensor
    self$multi.tensor<-pre.obj$data.tensor$multi.tensor
    self$onehot.tensor<-pre.obj$data.tensor$onehot.tensor

  },
  .getitem = function(index) {
    #data <- self$torch.data[index, ]
    num.tensor <- self$num.tensor[index, ]
    logi.tensor <- self$logi.tensor[index, ]
    bin.tensor <- self$bin.tensor[index, ]
    multi.tensor<- self$multi.tensor[index, ]
    onehot.tensor<- self$onehot.tensor[index, ]
    return(list("num.tensor"= num.tensor,"logi.tensor"=logi.tensor,"bin.tensor"=bin.tensor,"multi.tensor"=multi.tensor,"onehot.tensor"= onehot.tensor))
  },
  .length = function() {

    if(!is.null(self$num.tensor)){
      dim(self$num.tensor)[1]
    }else if(!is.null(self$logi.tensor)){
      dim(self$logi.tensor)[1]
    }else if(!is.null(self$bin.tensor)){
      dim(self$bin.tensor)[1]
    }else{
      dim(self$multi.tensor)[1]
    }

  }
  #,
  # .ncol = function() {
  #   self$torch.data$size()[[2]]
  # }
)


#' prepare the dataset
#' @importFrom torch dataset torch_tensor torch_cat nnf_one_hot
#' @export
torch_dataset_idx <- torch::dataset(
  name = "torch_dataset_idx",
  initialize = function(data,idx,scaler,lower,upper,categorical.encoding, initial.imp) {

    pre.obj <- preprocess(data[idx, , drop = FALSE],scaler,lower,upper,categorical.encoding, initial.imp)
    self$num.tensor<-pre.obj$data.tensor$num.tensor
    self$logi.tensor<-pre.obj$data.tensor$logi.tensor
    self$bin.tensor<-pre.obj$data.tensor$bin.tensor
    self$multi.tensor<-pre.obj$data.tensor$multi.tensor
    self$onehot.tensor<-pre.obj$data.tensor$onehot.tensor
  },
  .getitem = function(index) {
    #data <- self$torch.data[index, ]
    num.tensor <- self$num.tensor[index, ]
    logi.tensor <- self$logi.tensor[index, ]
    bin.tensor <- self$bin.tensor[index, ]
    multi.tensor<- self$multi.tensor[index, ]
    onehot.tensor<- self$onehot.tensor[index, ]
    return(list("num.tensor"= num.tensor,"logi.tensor"=logi.tensor,"bin.tensor"=bin.tensor,"multi.tensor"=multi.tensor,"onehot.tensor"= onehot.tensor))
  },
  .length = function() {

    if(!is.null(self$num.tensor)){
      dim(self$num.tensor)[1]
    }else if(!is.null(self$logi.tensor)){
      dim(self$logi.tensor)[1]
    }else if(!is.null(self$bin.tensor)){
      dim(self$bin.tensor)[1]
    }else{
      dim(self$multi.tensor)[1]
    }

  }
  #,
  #.ncol = function() {
  #  self$torch.data$size()[[2]]
  # }
)
