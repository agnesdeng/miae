library(torch)
library(palmerpenguins)
library(magrittr)

penguins_dataset <- dataset(

  name = "penguins_dataset",

  initialize = function() {
    self$data <- self$prepare_penguin_data()
  },

  .getitem = function(index) {

    x <- self$data[index, 2:-1]
    y <- self$data[index, 1]$to(torch_long())

    list(x, y)
  },

  .length = function() {
    self$data$size()[[1]]
  },

  prepare_penguin_data = function() {

    input <- na.omit(penguins)
    # conveniently, the categorical data are already factors
    input$species <- as.numeric(input$species)
    input$island <- as.numeric(input$island)
    input$sex <- as.numeric(input$sex)

    input <- as.matrix(input)
    torch_tensor(input)
  }
)


tuxes <- penguins_dataset()
tuxes$.length()
tuxes$.getitem(1)


dl <- tuxes %>% dataloader(batch_size = 8)

iter <- dl$.iter()
b <- iter$.next()
b


penguins_dataset_batching <- dataset(

  name = "penguins_dataset_batching",

  initialize = function() {
    self$data <- self$prepare_penguin_data()
  },

  # the only change is that this went from .getitem to .getbatch
  .getbatch = function(index) {

    x <- self$data[index, 2:-1]
    y <- self$data[index, 1]$to(torch_long())

    list(x, y)
  },

  .length = function() {
    self$data$size()[[1]]
  },

  prepare_penguin_data = function() {

    input <- na.omit(penguins)
    # conveniently, the categorical data are already factors
    input$species <- as.numeric(input$species)
    input$island <- as.numeric(input$island)
    input$sex <- as.numeric(input$sex)

    input <- as.matrix(input)
    torch_tensor(input)
  }
)


tuxes <- penguins_dataset_batching()
tuxes$.length()
tuxes$.getbatch(1)
tuxes$prepare_penguin_data()


dl <- tuxes %>% dataloader(batch_size = 8)

iter <- dl$.iter()
b <- iter$.next()
b
