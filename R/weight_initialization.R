# weight initialization

# He/Kaiming Uniform
init_he_uniform <- function(m, mode = "fan_in", slope = 0, nonlinearity = "relu") {
  if (any(class(m) == "nn_linear")) {
    nn_init_kaiming_uniform_(m$weight, a = slope, mode = mode, nonlinearity = nonlinearity)
    nn_init_zeros_(m$bias)
  }
}

# He/Kaiming Normal
init_he_normal <- function(m, mode = "fan_in", slope = 0, nonlinearity = "relu") {
  if (any(class(m) == "nn_linear")) {
    nn_init_kaiming_normal_(m$weight, a = slope, mode = mode, nonlinearity = nonlinearity)
    nn_init_zeros_(m$bias)
  }
}




# Xavier Uniform
init_xavier_uniform <- function(m, gain = 1) {
  if (any(class(m) == "nn_linear")) {
    nn_init_xavier_uniform_(m$weight, gain = gain)
    nn_init_zeros_(m$bias)
  }
}

# Xavier Normal
init_xavier_normal <- function(m, gain = 1) {
  if (any(class(m) == "nn_linear")) {
    nn_init_xavier_normal_(m$weight, gain = gain)
    nn_init_zeros_(m$bias)
  }
}




# MIDAS default
init_xavier_midas <- function(m, gain = 1 / sqrt(2)) {
  if (any(class(m) == "nn_linear")) {
    nn_init_xavier_normal_(m$weight, gain = gain)
    nn_init_zeros_(m$bias)
  }
}
