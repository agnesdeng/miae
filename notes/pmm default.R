#' Auxiliary function for pmm.params for midae
#' @description Auxiliary function for setting up the default pmm-related parameters for midae
#' @param pmm.type The type of predictive mean matching (PMM). Possible values:
#' \itemize{
#'  \item \code{NULL}: Imputations without PMM;
#'  \item \code{0}: Imputations with PMM type 0;
#'  \item \code{1}: Imputations with PMM type 1;
#'  \item \code{2}: Imputations with PMM type 2;
#'  \item \code{"auto"} (default): Imputations with PMM type 2 for numeric/integer variables; imputations without PMM for categorical variables.
#' }
#' @param pmm.k The number of donors for predictive mean matching. Default: 5
#' @param pmm.link The link for predictive mean matching in binary variables
#' \itemize{
#'  \item \code{"prob"} (default): use probabilities;
#'  \item \code{"logit"}: use logit values.
#' }
#' @param pmm.save.vars The names of variables whose predicted values of observed entries will be saved. Only use for PMM.
#' @export
dae_pmm_default <- function(pmm.type = NULL, pmm.k = 5, pmm.link = "prob", pmm.save.vars = NULL) {
  list(pmm.type = pmm.type, pmm.k = pmm.k, pmm.link = pmm.link, pmm.save.vars = pmm.save.vars)
}



#' Auxiliary function for pmm.params for mivae
#' @description Auxiliary function for setting up the default pmm-related parameters for mivae
#' @param pmm.type The type of predictive mean matching (PMM). Possible values:
#' \itemize{
#'  \item \code{NULL}(Default): Imputations without PMM;
#'  \item \code{0}: Imputations with PMM type 0;
#'  \item \code{1}: Imputations with PMM type 1;
#'  \item \code{2}: Imputations with PMM type 2;
#'  \item \code{"auto"}: Imputations with PMM type 2 for numeric/integer variables; imputations without PMM for categorical variables.
#' }
#' @param pmm.k The number of donors for predictive mean matching. Default: 5
#' @param pmm.link The link for predictive mean matching in binary variables
#' \itemize{
#'  \item \code{"prob"} (Default): use probabilities;
#'  \item \code{"logit"}: use logit values.
#' }
#' @param pmm.save.vars The names of variables whose predicted values of observed entries will be saved. Only use for PMM.
#' @export
vae_pmm_default <- function(pmm.type = NULL, pmm.k = 5, pmm.link = "prob", pmm.save.vars = NULL) {
  list(pmm.type = pmm.type, pmm.k = pmm.k, pmm.link = pmm.link, pmm.save.vars = pmm.save.vars)
}
