
install.packages("reticulate")
install.packages("tensorflow")

#for windows
library(reticulate)
path_to_python <- install_python()
#virtualenv_create("r-reticulate", python = path_to_python)
conda_create("r-reticulate")

library(tensorflow)
install_tensorflow(
  method = "conda",
  envname = "r-reticulate")

install.packages("keras")
library(keras)
install_keras(
  method = "conda",
  envname = "r-reticulate")

library(tensorflow)
tf$constant("Hello Tensorflow!")
tf$config$list_physical_devices("GPU")



#for others

#install.packages("reticulate")
#install.packages("tensorflow")


#library(reticulate)
#path_to_python <- install_python()
#virtualenv_create("r-reticulate", python = path_to_python)

#library(tensorflow)
#install_tensorflow(envname = "r-reticulate")


#library(tensorflow)
#tf$constant("Hello Tensorflow!")

