library(torch)
library(purrr)
library(readr)
library(dplyr)
library(ggplot2)
library(ggrepel)

download.file(
  "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data",
  destfile = "agaricus-lepiota.data"
)

mushroom_data <- read_csv(
  "agaricus-lepiota.data",
  col_names = c(
    "poisonous",
    "cap-shape",
    "cap-surface",
    "cap-color",
    "bruises",
    "odor",
    "gill-attachment",
    "gill-spacing",
    "gill-size",
    "gill-color",
    "stalk-shape",
    "stalk-root",
    "stalk-surface-above-ring",
    "stalk-surface-below-ring",
    "stalk-color-above-ring",
    "stalk-color-below-ring",
    "veil-type",
    "veil-color",
    "ring-type",
    "ring-number",
    "spore-print-color",
    "population",
    "habitat"
  ),
  col_types = rep("c", 23) %>% paste(collapse = "")
) %>%
  # can as well remove because there's just 1 unique value
  select(-`veil-type`)


mushroom_dataset <- dataset(
  name = "mushroom_dataset",

  initialize = function(indices) {
    data <- self$prepare_mushroom_data(mushroom_data[indices, ])
    self$xcat <- data[[1]][[1]]
    self$xnum <- data[[1]][[2]]
    self$y <- data[[2]]
  },

  .getitem = function(i) {
    xcat <- self$xcat[i, ]
    xnum <- self$xnum[i, ]
    y <- self$y[i, ]

    list(x = list(xcat, xnum), y = y)
  },

  .length = function() {
    dim(self$y)[1]
  },

  prepare_mushroom_data = function(input) {

    input <- input %>%
      mutate(across(.fns = as.factor))

    target_col <- input$poisonous %>%
      as.integer() %>%
      `-`(1) %>%
      as.matrix()

    categorical_cols <- input %>%
      select(-poisonous) %>%
      select(where(function(x) nlevels(x) != 2)) %>%
      mutate(across(.fns = as.integer)) %>%
      as.matrix()

    numerical_cols <- input %>%
      select(-poisonous) %>%
      select(where(function(x) nlevels(x) == 2)) %>%
      mutate(across(.fns = as.integer)) %>%
      as.matrix()

    list(list(torch_tensor(categorical_cols), torch_tensor(numerical_cols)),
         torch_tensor(target_col))
  }
)



train_indices <- sample(1:nrow(mushroom_data), size = floor(0.8 * nrow(mushroom_data)))
valid_indices <- setdiff(1:nrow(mushroom_data), train_indices)

train_ds <- mushroom_dataset(train_indices)
train_dl <- train_ds %>% dataloader(batch_size = 256, shuffle = TRUE)

valid_ds <- mushroom_dataset(valid_indices)
valid_dl <- valid_ds %>% dataloader(batch_size = 256, shuffle = FALSE)

test.ds<-mushroom_dataset(1:5)
test.ds$.length()
########################################################################
str(mushroom_data)
#16:Long type
train_ds$xcat
#5: Float
train_ds$xnum
#response (binary) :Float
train_ds$y


mushroom_data <- mushroom_data %>%
  mutate(across(.fns = as.factor))


mushroom_data
mushroom_data$poisonous %>%
  #1,2
  as.integer() %>%
  #0,1
  `-`(1) %>%
  #column vector
  as.matrix()



categorical_cols <- mushroom_data %>%
  select(-poisonous) %>%
  select(where(function(x) nlevels(x) != 2)) %>%
  #1,2,3,4,....
  mutate(across(.fns = as.integer)) %>%
  as.matrix()

numerical_cols <- mushroom_data %>%
  select(-poisonous) %>%
  select(where(function(x) nlevels(x) == 2)) %>%
  #binary:  1,2
  mutate(across(.fns = as.integer)) %>%
  as.matrix()


####################################
embedding_module <- nn_module(

  initialize = function(cardinalities) {
    self$embeddings = nn_module_list(lapply(cardinalities, function(x) nn_embedding(num_embeddings = x, embedding_dim = ceiling(x/2))))
  },

  forward = function(x) {
    embedded <- vector(mode = "list", length = length(self$embeddings))
    for (i in 1:length(self$embeddings)) {
      embedded[[i]] <- self$embeddings[[i]](x[ , i])
    }
    torch_cat(embedded, dim = 2)
  }
)


net <- nn_module(
  "mushroom_net",

  initialize = function(cardinalities,
                        num_numerical,
                        fc1_dim,
                        fc2_dim) {
    self$embedder <- embedding_module(cardinalities)
    self$fc1 <- nn_linear(sum(map(cardinalities, function(x) ceiling(x/2)) %>% unlist()) + num_numerical, fc1_dim)
    self$fc2 <- nn_linear(fc1_dim, fc2_dim)
    self$output <- nn_linear(fc2_dim, 1)
  },

  forward = function(xcat, xnum) {
    embedded <- self$embedder(xcat)
    all <- torch_cat(list(embedded, xnum$to(dtype = torch_float())), dim = 2)
    all %>% self$fc1() %>%
      nnf_relu() %>%
      self$fc2() %>%
      self$output() %>%
      nnf_sigmoid()
  }
)



cardinalities <- map(
  mushroom_data[ , 2:ncol(mushroom_data)], compose(nlevels, as.factor)) %>%
  keep(function(x) x > 2) %>%
  unlist() %>%
  unname()

num_numerical <- ncol(mushroom_data) - length(cardinalities) - 1

fc1_dim <- 16
fc2_dim <- 16

model <- net(
  cardinalities,
  num_numerical,
  fc1_dim,
  fc2_dim
)


model
model$embedder$embeddings[[1]]
#dim: 6 x 3
summary(train_ds$xcat[,1])
cardinalities <- map(
  mushroom_data[ , 2:ncol(mushroom_data)], compose(nlevels, as.factor)) %>%
  keep(function(x) x > 2) %>%
  unlist() %>%
  unname()

cardinalities

model$embedder$embeddings[[1]]
model$embedder$embeddings[[2]]

model$embedder
model$fc1
sum(ceiling(cardinalities/2))
num_numerical
57+5

model$fc2
nn_linear(sum(map(cardinalities, function(x) ceiling(x/2)) %>% unlist()) + num_numerical, fc1_dim)
