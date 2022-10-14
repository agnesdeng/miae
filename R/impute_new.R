#' Impute new data with a saved \code{midae} or \code{mivae} imputation model
#' @param  path A path to the saved imputation model
#' @param  newdata A data frame, tibble or data.table. New data with missing values.
#' @param  scaler The name of scaler for transforming numeric features. Can be "standard", "minmax" or "none".
#' @param  m The number of imputed datasets. Default: 5.
#' @return A list of \code{m} imputed datasets for new data.
#' @importFrom torch dataloader torch_load dataloader_make_iter dataloader_next
#' @export
impute_new <- function(path, newdata, scaler = "none", m = 5) {


  model <- torch::torch_load(path)

  model$eval()


  pre.obj <- preprocess(newdata, scaler = scaler)

  torch.data <- torch_dataset(newdata, scaler = scaler)


  n.features <- torch.data$.ncol()
  n.samples <- torch.data$.length()





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
    if(is.list(output.data)){
      imp.data <- postprocess(output.data = output.data$reconstrx, pre.obj = pre.obj, scaler = scaler)
    }else{
      imp.data <- postprocess(output.data = output.data, pre.obj = pre.obj, scaler = scaler)
    }

    na.vars <- pre.obj$ordered.names[colSums(na.loc) != 0]

    for (var in na.vars) {

      newdata[[var]][na.loc[, var]] <- imp.data[[var]][na.loc[, var]]

    }

    imputed.data[[i]] <- newdata
  }

  imputed.data


}
