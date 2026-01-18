# Package index

## All functions

- [`createNA()`](createNA.md) : Create missing values for a dataset

- [`dae_default()`](dae_default.md) : Auxiliary function for dae.params

- [`dae_pmm_default()`](dae_pmm_default.md) : Auxiliary function for
  pmm.params for midae

- [`data_clean()`](data_clean.md) : Data cleaning

- [`decile()`](decile.md) : scale a vector using decile

- [`decile_scaler()`](decile_scaler.md) : scale a dataset using decile
  and return a scaled dataframe, the 1st decile and the 9th decile of
  each column

- [`feature_type()`](feature_type.md) : This function is used to return
  the type(numeric,binary,multiclass) of each feature

- [`impute_new()`](impute_new.md) :

  Impute new data with a saved `midae` or `mivae` imputation model

- [`miae`](miae-package.md) [`miae-package`](miae-package.md) :

  miae: Multiple Imputation Through Autoencoders

- [`midae()`](midae.md) : Multiple imputation through denoising
  autoencoders with dropout

- [`minmax()`](minmax.md) : scale a vector using minmax

- [`minmax_scaler()`](minmax_scaler.md) : scale a dataset using minmax
  and return a scaled dataframe, the colmin and colmax of each column

- [`mivae()`](mivae.md) : multiple imputation through variational
  autoencoders

- [`mivae.latent()`](mivae.latent.md) : multiple imputation through
  variational autoencoders (latent) haven't change default setting yet

- [`newborn`](newborn.md) : NHANES III (1988-1994) newborn data

- [`nhanes3`](nhanes3.md) : A small subset of the NHANES III (1988-1994)
  newborn data

- [`plot_dropout()`](plot_dropout.md) : plot the density of the impiuted
  values of a numeric variable using different dropout probabilities

- [`postprocess()`](postprocess.md) : postprocess data

- [`rev_decile_scaler()`](rev_decile_scaler.md) : This function
  back-transform decile-scaled data to an output as data matrix

- [`rev_minmax_scaler()`](rev_minmax_scaler.md) : This function
  back-transform minmax-scaled data to an output as data matrix

- [`rev_onehot()`](rev_onehot.md) : reverse onehot data

- [`rev_robust_scaler()`](rev_robust_scaler.md) : This function
  back-transform robust-scaled data to an output as data matrix

- [`rev_standard_scaler()`](rev_standard_scaler.md) : This function
  back-transform standardized data to an output as data matrix

- [`robust()`](robust.md) : scale a vector using robust scaler

- [`robust_scaler()`](robust_scaler.md) : scale a dataset using robust
  scaler and return a scaled dataframe, the median, the lower quantile
  (25% by default) and the upper quantile (75% by default) of each
  column

- [`samples()`](samples.md) : Random sampling from observed values

- [`standard()`](standard.md) : scale a vector using standardize

- [`standard_scaler()`](standard_scaler.md) : scale a dataset using
  stardardize and return a scaled dataframe, the colmean and colsd of
  each column

- [`tune_dropout_midae()`](tune_dropout_midae.md) : Tune dropout rate
  for midae

- [`vae_default()`](vae_default.md) : Auxiliary function for vae.params
