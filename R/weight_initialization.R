# weight initialization


# ELU He/Kaiming Normal
init_he_normal_elu <- function(m) {
  if (any(class(m) == "nn_linear")) {
    nn_init_kaiming_normal_(m$weight, a = 0.538816, mode = "fan_in", nonlinearity = "leaky_relu")
    nn_init_zeros_(m$bias)
  }
}

# ELU He/Kaiming Normal accounting for dropout p=0.5
init_he_normal_elu_dropout <- function(m) {
  if (any(class(m) == "nn_linear")) {
    nn_init_kaiming_normal_(m$weight, a = 1.257237, mode = "fan_in", nonlinearity = "leaky_relu")
    nn_init_zeros_(m$bias)
  }
}


# SELU He/Kaiming Normal
init_he_normal_selu <- function(m) {
  if (any(class(m) == "nn_linear")) {
    nn_init_kaiming_normal_(m$weight, a = 0, mode = "fan_in", nonlinearity = "linear")
    nn_init_zeros_(m$bias)
  }
}



#leaky.relu He/Kaiming Normal
init_he_normal_leaky.relu <- function(m) {
  if (any(class(m) == "nn_linear")) {
    nn_init_kaiming_normal_(m$weight, a = 0.01, mode = "fan_in", nonlinearity = "leaky_relu")
    nn_init_zeros_(m$bias)
  }
}



# He/Kaiming Normal #ReLU: gain=sqrt(2)
init_he_normal <- function(m) {
  if (any(class(m) == "nn_linear")) {
    nn_init_kaiming_normal_(m$weight, a = 0, mode = "fan_in", nonlinearity = "relu")
    nn_init_zeros_(m$bias)
  }
}


# He/Kaiming Normal accounting for dropout
init_he_normal_dropout <- function(m) {
  if (any(class(m) == "nn_linear")) {
    nn_init_kaiming_normal_(m$weight, a = 1, mode = "fan_in", nonlinearity = "leaky_relu")
    nn_init_zeros_(m$bias)
  }
}




# He/Kaiming Uniform #ReLU: gain=sqrt(2)
init_he_uniform <- function(m) {
  if (any(class(m) == "nn_linear")) {
    nn_init_kaiming_uniform_(m$weight, a = 0, mode = "fan_in", nonlinearity = "relu")
    nn_init_zeros_(m$bias)
  }
}






# Xavier Normal
init_xavier_normal <- function(m) {
  if (any(class(m) == "nn_linear")) {
    nn_init_xavier_normal_(m$weight, gain = 1)
    nn_init_zeros_(m$bias)
  }
}



# Xavier Uniform
init_xavier_uniform <- function(m) {
  if (any(class(m) == "nn_linear")) {
    nn_init_xavier_uniform_(m$weight, gain = 1)
    nn_init_zeros_(m$bias)
  }
}





# MIDAS default
init_xavier_midas <- function(m) {
  if (any(class(m) == "nn_linear")) {
    nn_init_xavier_normal_(m$weight, gain = 1 / sqrt(2))
    nn_init_zeros_(m$bias)
  }
}
