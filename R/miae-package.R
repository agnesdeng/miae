#' \pkg{miae}: Multiple Imputation Through Autoencoders
#' @name miae-package
#' @docType package
#' @description Miae offers a scalable solution for imputing large datasets using various types of autoencoders.
#' @import data.table
#' @import ggplot2
#' @importFrom magrittr %>%
#' @rawNamespace import(rlang, except = ":=")
#' @importFrom stats median rnorm sd complete.cases na.omit reformulate predict quantile rbinom
#' @importFrom torch nn_module nn_module_list nn_sequential nnf_dropout
#' @importFrom torch nn_linear nn_relu nn_elu nn_selu nn_identity nn_tanh nn_sigmoid nn_leaky_relu
#' @importFrom torch dataset dataloader dataloader_make_iter dataloader_next torch_tensor torch_cat nnf_one_hot as_array
#' @importFrom torch torch_device
#' @importFrom torch torch_mul torch_zeros torch_mean torch_sum torch_float torch_exp torch_randn_like torch_randperm
#' @importFrom torch nn_init_zeros_ nn_init_kaiming_normal_ nn_init_kaiming_uniform_ nn_init_xavier_normal_ nn_init_xavier_uniform_
#' @importFrom torch nn_mse_loss nn_bce_with_logits_loss nn_cross_entropy_loss
#' @importFrom torch torch_argmax nn_softmax
#' @importFrom torch torch_save torch_load
#' @importFrom torch optim_adam optim_adadelta optim_adagrad optim_rmsprop optim_sgd
#' @importFrom torchopt optim_adamw
# @references


"_PACKAGE"
