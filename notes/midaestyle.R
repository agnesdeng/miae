#' Multiple imputation through denoising autoencoders with dropout
#' @description This function...
#' @param data A data frame, tibble or data table with missing values.
#' @param m The number of imputed datasets.
#' @param pmm.type The type of predictive mean matching (PMM). Possible values:
#' \itemize{
#'  \item \code{NULL}: Imputations without PMM;
#'  \item \code{0}: Imputations with PMM type 0;
#'  \item \code{1}: Imputations with PMM type 1;
#'  \item \code{2}: Imputations with PMM type 2;
#'  \item \code{"auto"} (Default): Imputations with PMM type 2 for numeric/integer variables; imputations without PMM for categorical variables.
#' }
#' @param pmm.k The number of donors for predictive mean matching. Default: 5
#' @param pmm.save.vars The names of variables whose predicted values of observed entries will be saved. Only use for PMM.
#' @param epochs The number of training epochs (iterations).
#' @param batch.size The size of samples in each batch. Default: 32.
#' @param subsample The subsample ratio of training data. Default: 1.
#' @param shuffle Whether or not to shuffle training data. Default: TRUE
#' @param input.dropout The dropout probability of the input layer.
#' @param hidden.dropout The dropout probability of the hidden layers.
#' @param optimizer The name of the optimizer. Options are : "adamW" (default), "adam" and "sgd".
#' @param learning.rate The learning rate. The default value is 0.001.
#' @param weight.decay Weight decay (L2 penalty). The default value is 0.
#' @param momentum Parameter for "sgd" optimizer. It is used for accelerating SGD in the relevant direction and dampens oscillations.
#' @param eps A small positive value used to prevent division by zero for the "adamW" optimizer. Default: 1e-07.
#' @param encoder.structure A vector indicating the structure of encoder. Default: c(128,64,32)
#' @param decoder.structure A vector indicating the structure of decoder. Default: c(32,64,128)
#' @param act The name of activation function. Can be: "relu", "elu", "leaky.relu", "tanh", "sigmoid" and "identity".
#' @param init.weight Techniques for weights initialization. Can be "xavier.uniform" or "kaiming.uniform".
#' @param scaler The name of scaler for transforming numeric features. Can be "standard", "minmax" ,"decile" or "none".
#' @param loss.na.scale Whether to multiply the ratio of missing values in  a feature to calculate the loss function. Default: FALSE.
#' @param verbose Whether or not to print training loss information. Default: TRUE.
#' @param print.every.n If verbose is set to TRUE, print out training loss for every n epochs.
#' @param save.model Whether or not to save the imputation model. Default: FALSE.
#' @param path The path where the final imputation model will be saved.
#' @importFrom torch dataloader nn_mse_loss nn_bce_with_logits_loss nn_cross_entropy_loss optim_adam optim_sgd torch_save torch_load torch_argmax dataloader_make_iter dataloader_next
#' @importFrom torchopt optim_adamw
#' @export
#' @examples
#' withNA.df <- createNA(data = iris, p = 0.2)
#' imputed.data <- midae(data = withNA.df, m = 5, epochs = 5, path = file.path(tempdir(), "midaemodel.pt"))
