# Auxiliary function for pmm.params for midae

Auxiliary function for setting up the default pmm-related parameters for
midae

## Usage

``` r
dae_pmm_default(
  pmm.type = NULL,
  pmm.k = 5,
  pmm.link = "prob",
  pmm.save.vars = NULL
)
```

## Arguments

- pmm.type:

  The type of predictive mean matching (PMM). Possible values:

  - `NULL`: Imputations without PMM;

  - `0`: Imputations with PMM type 0;

  - `1`: Imputations with PMM type 1;

  - `2`: Imputations with PMM type 2;

  - `"auto"` (default): Imputations with PMM type 2 for numeric/integer
    variables; imputations without PMM for categorical variables.

- pmm.k:

  The number of donors for predictive mean matching. Default: 5

- pmm.link:

  The link for predictive mean matching in binary variables

  - `"prob"` (default): use probabilities;

  - `"logit"`: use logit values.

- pmm.save.vars:

  The names of variables whose predicted values of observed entries will
  be saved. Only use for PMM.
