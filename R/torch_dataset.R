#' prepare the dataset
#' @importFrom torch dataset torch_tensor torch_cat nnf_one_hot
torch_dataset <- torch::dataset(
  name = "torch_dataset",
  initialize = function(data) {
    pre.obj <- preprocess(data)

    self$torch.data <- pre.obj$onehot.tensor
    # pre.obj<-preprocess(data)
    # torch.data<-pre.obj$onehot.tensor
  },
  .getitem = function(index) {
    x <- self$torch.data[index, ]

    x
  },
  .length = function() {
    self$torch.data$size()[[1]]
  },
  .ncol = function() {
    self$torch.data$size()[[2]]
  }
)
