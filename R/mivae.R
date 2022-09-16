#' multiple imputation through autoencoders
#' @param data data with missing values
#' @param m the number of imputed datasets
#' @param epochs the number of training epochs
#' @param latent.dim the size of latent layer
#' @param learning.rate learning rate
#' @param batch.size the size of each batch
#' @param encoder.structure the size of each layer in the encoder
#' @param decoder.structure the size of each layer in the decoder
#' @importFrom torch dataloader nn_mse_loss nn_bce_with_logits_loss nn_cross_entropy_loss optim_adam torch_save torch_load torch_argmax dataloader_make_iter dataloader_next
#' @export
mivae <- function(data, m = 5, epochs = 10, latent.dim = 16, learning.rate = 0.001, batch.size = 50, encoder.structure = c(128, 64, 32), decoder.structure = c(32, 64, 128)) {
  pre.obj <- preprocess(data)


  torch.data <- torch_dataset(data)


  n.features <- torch.data$.ncol()
  n.samples <- torch.data$.length()

  dl <- torch::dataloader(dataset = torch.data, batch_size = batch.size, shuffle = TRUE)


  model <- vae(n.features = n.features, latent.dim = latent.dim, encoder.structure = encoder.structure, decoder.structure = encoder.structure)


  # define the loss function for different variable
  num_loss <- torch::nn_mse_loss(reduction = "sum")
  bin_loss <- torch::nn_bce_with_logits_loss()
  multi_loss <- torch::nn_cross_entropy_loss()


  ## choose optimizer & learning rate
  optimizer <- torch::optim_adam(model$parameters, lr = learning.rate)



  # epochs: number of iterations

  for (epoch in seq_len(epochs)) {
    epoch.loss <- 0

    coro::loop(for (b in dl) { # loop over all minibatches for one epoch

      Out <- model(b)

      # numeric
      num.cost <- num_loss(input = Out$reconstrx[, pre.obj$num.idx], target = b[, pre.obj$num.idx])
      total.num.cost <- num.cost / batch.size

      # binary
      bin.cost <- vector("list", length = length(pre.obj$bin))
      names(bin.cost) <- pre.obj$bin

      for (var in pre.obj$bin) {
        bin.cost[[var]] <- bin_loss(input = Out$reconstrx[, pre.obj$bin.idx[[var]]], target = b[, pre.obj$bin.idx[[var]]])
      }

      total.bin.cost <- do.call(sum, bin.cost)

      # multiclass
      multi.cost <- vector("list", length = length(pre.obj$multi))
      names(multi.cost) <- pre.obj$multi

      for (var in pre.obj$multi) {
        multi.cost[[var]] <- multi_loss(input = Out$reconstrx[, pre.obj$multi.idx[[var]]], target = torch::torch_argmax(b[, pre.obj$multi.idx[[var]]], dim = 2))
      }
      total.multi.cost <- do.call(sum, multi.cost)

      # Total cost (reconstruction loss)
      cost <- sum(total.num.cost, total.bin.cost, total.multi.cost)

      # KL loss
      mu <- Out$mu
      log.var <- Out$log.var

      #
      # kl.div =  1 + log.var - mu$pow(2) - log.var$exp()
      # kl.div.sum = - 0.5 *kl.div$sum()

      kl.divergence <- torch_mean(-0.5 * torch_sum(1 + log.var - mu$pow(2) - log.var$exp()))

      total.cost <- cost + kl.divergence

      #
      optimizer$zero_grad()
      total.cost$backward()
      optimizer$step()


      batch.loss <- total.cost$item()
      epoch.loss <- epoch.loss + batch.loss

      if (epoch == epochs) {
        # torch_save(model,path="C:/Users/agnes/Desktop/torch")
        torch::torch_save(model, paste0("model_", epoch, ".pt"))
      }
    })

    # cat(sprintf("Loss at epoch %d: %1f\n", epoch, 128*l/60000))
    cat(sprintf("Loss at epoch %d: %1f\n", epoch, epoch.loss / length(dl)))
  }


  last.model <- paste0(paste0("model_", epochs), ".pt")
  model <- torch::torch_load(last.model)

  model$eval()

  # The whole dataset
  eval_dl <- torch::dataloader(dataset = torch.data, batch_size = n.samples, shuffle = TRUE)
  eval_dl

  wholebatch <- eval_dl %>%
    torch::dataloader_make_iter() %>%
    torch::dataloader_next()

  # imputed data
  imputed.data <- vector("list", length = m)
  na.loc <- pre.obj$na.loc

  for (i in seq_len(m)) {
    output.list <- model(wholebatch)
    imp.data <- postprocess(output.data = output.list$reconstrx, pre.obj = pre.obj)
    na.vars <- pre.obj$ordered.names[colSums(na.loc) != 0]

    for (var in na.vars) {

      if(is.data.table(data)){
        #data.table
        stop("Compatibility with data.table is coming soon.")
      }else{
        data[[var]][na.loc[, var]] <- imp.data[[var]][na.loc[, var]]
      }

    }

    imputed.data[[i]] <- data
  }
  imputed.data
}
