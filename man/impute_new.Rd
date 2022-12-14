% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/impute_new.R
\name{impute_new}
\alias{impute_new}
\title{Impute new data with a saved \code{midae} or \code{mivae} imputation model}
\usage{
impute_new(path, newdata, scaler = "none", m = 5)
}
\arguments{
\item{path}{A path to the saved imputation model}

\item{newdata}{A data frame, tibble or data.table. New data with missing values.}

\item{scaler}{The name of scaler for transforming numeric features. Can be "standard", "minmax" or "none".}

\item{m}{The number of imputed datasets. Default: 5.}
}
\value{
A list of \code{m} imputed datasets for new data.
}
\description{
Impute new data with a saved \code{midae} or \code{mivae} imputation model
}
