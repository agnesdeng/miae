#' prepare the dataset
#' @importFrom torch dataset torch_tensor torch_cat nnf_one_hot
#' @export
torch_dataset <- torch::dataset(
  name = "torch_dataset",
  initialize = function(data) {
    pre.obj <- preprocess(data)

    self$torch.data <- pre.obj$data.tensor

  },
  .getitem = function(index) {
    data <- self$torch.data[index, ]

    return(list("data"=data,"index"=index))

  },
  .length = function() {
    self$torch.data$size()[[1]]
  },
  .ncol = function() {
    self$torch.data$size()[[2]]
  }
)


#' prepare the dataset
#' @importFrom torch dataset torch_tensor torch_cat nnf_one_hot
#' @export
torch_dataset_idx <- torch::dataset(
  name = "torch_dataset",
  initialize = function(data,idx) {
    pre.obj <- preprocess(data[idx,])

    self$torch.data <- pre.obj$data.tensor

  },
  .getitem = function(index) {
    data <- self$torch.data[index, ]

    return(list("data"=data,"index"=index))

  },
  .length = function() {
    self$torch.data$size()[[1]]
  },
  .ncol = function() {
    self$torch.data$size()[[2]]
  }
)
