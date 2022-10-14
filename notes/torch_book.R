library(torch)

x <- torch_ones(2, 2, requires_grad = TRUE)
x
y <- x$mean()
y
y$grad_fn
#Actual computation of gradients is triggered by calling backward() on the output tensor.
y$backward()
#x has a non-null field termed grad that stores the gradient of y with respect to x:
x$grad


x1 <- torch_ones(2, 2, requires_grad = TRUE)
x2 <- torch_tensor(1.1, requires_grad = TRUE)
y <- x1 * (x2 + 2)

x1
x2
y
z<-y$pow(2) * 3
z
out <- z$mean()
out

out$grad_fn
## how to compute the gradient for pow in z = y.pow(2) * 3
out$grad_fn$next_functions
out$grad_fn$next_functions[[1]]$next_functions
# how to compute the gradient for the multiplication in y = x * (x + 2)
out$grad_fn$next_functions[[1]]$next_functions[[1]]$next_functions
# how to compute the gradient for the two branches of y = x * (x + 2),
# where the left branch is a leaf node (AccumulateGrad for x1)
out$grad_fn$next_functions[[1]]$next_functions[[1]]$next_functions[[1]]$next_functions



#If we now call out$backward(), all tensors in the graph will have their respective gradients calculated.
out$backward()

#intermediate grad won't be stored unless we allow it
z$grad

####
x1 <- torch_ones(2, 2, requires_grad = TRUE)
x2 <- torch_tensor(1.1, requires_grad = TRUE)
y <- x1 * (x2 + 2)

x1
x2
y
z<-y$pow(2) * 3
z
out <- z$mean()
out

y$retain_grad()
z$retain_grad()
#If we now call out$backward(), all tensors in the graph will have their respective gradients calculated.
out$backward()

#intermediate grad stored this time
z$grad
y$grad
x1$grad
x2$grad
#########



library(torch)

### generate training data -----------------------------------------------------

# input dimensionality (number of input features)
d_in <- 3
# output dimensionality (number of predicted features)
d_out <- 1
# number of observations in training set
n <- 100


# create random data
x <- torch_randn(n, d_in)
y <- x[, 1, NULL] * 0.2 - x[, 2, NULL] * 1.3 - x[, 3, NULL] * 0.5 + torch_randn(n, 1)


### initialize weights ---------------------------------------------------------

# dimensionality of hidden layer
d_hidden <- 32
# weights connecting input to hidden layer
w1 <- torch_randn(d_in, d_hidden, requires_grad = TRUE)
# weights connecting hidden to output layer
w2 <- torch_randn(d_hidden, d_out, requires_grad = TRUE)

# hidden layer bias
b1 <- torch_zeros(1, d_hidden, requires_grad = TRUE)
# output layer bias
b2 <- torch_zeros(1, d_out, requires_grad = TRUE)

### network parameters ---------------------------------------------------------

learning_rate <- 1e-4

### training loop --------------------------------------------------------------

for (t in 1:200) {
  ### -------- Forward pass --------

  y_pred <- x$mm(w1)$add(b1)$clamp(min = 0)$mm(w2)$add(b2)

  ### -------- compute loss --------
  loss <- (y_pred - y)$pow(2)$sum()
  if (t %% 10 == 0)
    cat("Epoch: ", t, "   Loss: ", loss$item(), "\n")

  ### -------- Backpropagation --------

  # compute gradient of loss w.r.t. all tensors with requires_grad = TRUE
  loss$backward()

  ### -------- Update weights --------

  # Wrap in with_no_grad() because this is a part we DON'T
  # want to record for automatic gradient computation
  with_no_grad({
    w1 <- w1$sub_(learning_rate * w1$grad)
    w2 <- w2$sub_(learning_rate * w2$grad)
    b1 <- b1$sub_(learning_rate * b1$grad)
    b2 <- b2$sub_(learning_rate * b2$grad)

    # Zero gradients after every pass, as they'd accumulate otherwise
    w1$grad$zero_()
    w2$grad$zero_()
    b1$grad$zero_()
    b2$grad$zero_()
  })

}




###
library(torch)
l <- nn_linear(3, 1)
l
l$parameters



data  <- torch_randn(10, 3)
data
out <- l(data)
out

out$grad_fn
l$weight$grad
l$bias$grad
#error, because out is {10,1} , not a scaler
out$backward()


###
torch_tensor(10)
torch_tensor(10)$`repeat`(10)
torch_tensor(10)$`repeat`(10)$unsqueeze(1)
torch_tensor(10)$`repeat`(10)$unsqueeze(1)$t()


d_avg_d_out <- torch_tensor(10)$`repeat`(10)$unsqueeze(1)$t()
d_avg_d_out
out$backward(gradient = d_avg_d_out)
out


l$weight$grad
l$bias$grad


#
x1<-torch_tensor(c(1,2,3),requires_grad = T)
x2<-torch_tensor(c(3,4,5),requires_grad = T)
x3<-torch_tensor(c(6,7,8),requires_grad = T)

x1
x2
x3
y<-3*x1+2*x2*x2+torch_log(x3)
y



gradients<-torch_tensor(c(0.1,1,0.0001))
y$backward(gradient=gradients,retain_graph=TRUE)
#dy/dx1=3     [3*0.1, 3*1, 3*0.001]
x1$grad
#dy/dx2=4*x2  [4*3*0.1,4*4*1,4*5*0.0001 ]
x2$grad
#dy/dx3=1/x3  [1/6*0.1,1/7,1/8*0.0001]
x3$grad
c(1/6,1/7,1/8)
c(1/6*0.1,1/7,1/8*0.0001)


l <- nn_linear(3, 1)
l
l$parameters
data  <- torch_ones(2, 3)
data
out <- l(data)
out

out$grad_fn
l$weight$grad
l$bias$grad
#error, because out is {10,1} , not a scaler
out$backward()


###
torch_tensor(2)
torch_tensor(2)$`repeat`(2)
torch_tensor(2)$`repeat`(2)$unsqueeze(1)
torch_tensor(2)$`repeat`(2)$unsqueeze(1)$t()


d_avg_d_out <- torch_tensor(2)$`repeat`(2)$unsqueeze(1)$t()
d_avg_d_out
out$backward(gradient = d_avg_d_out/2)
out

torch_tensor(c(1,1))$unsqueeze(1)$t()
out$backward(gradient = torch_tensor(c(1,1))$unsqueeze(1)$t())
out

l$weight
l$weight$grad
l$bias
l$bias$grad


d_avg_d_out <- torch_tensor(2)$`repeat`(2)$unsqueeze(1)$t()
d_avg_d_out/2
out$backward(gradient = d_avg_d_out/2)
out
out
data
l$weight
l$weight$grad
l$bias
l$bias$grad


x=torch_tensor(c(1,2,3,4),requires_grad = T)
z=2*x
z
loss=z$sum(dim=1)
loss
z$sum()



# Loss --------------------------------------------------------------------


library(torch)
x <- torch_randn(c(3, 2, 3))
y <- torch_zeros(c(3, 2, 3))

nnf_mse_loss(x, y)


loss <- nn_mse_loss()

loss(x, y)


data <- torch_randn(1, 3)

model <- nn_linear(3, 1)
model$parameters

#
optimizer <- optim_adam(model$parameters, lr = 0.01)
optimizer


optimizer$param_groups[[1]]$params


#calcuate gradient but not update parameters
out <- model(data)
out$backward()

optimizer$param_groups[[1]]$params
model$parameters


#update parameters
optimizer$step()

optimizer$param_groups[[1]]$params
model$parameters



x <- torch_randn(n, d_in)
y <- x[, 1, NULL] * 0.2 - x[, 2, NULL] * 1.3 - x[, 3, NULL] * 0.5 + torch_randn(n, 1)


x
y

x[, 1, NULL]
x[,1]



# input dimensionality (number of input features)
d_in <- 3
# output dimensionality (number of predicted features)
d_out <- 1
# number of observations in training set
n <- 100


# create random data
x <- torch_randn(n, d_in)
y <- x[, 1, NULL] * 0.2 - x[, 2, NULL] * 1.3 - x[, 3, NULL] * 0.5 + torch_randn(n, 1)



# dimensionality of hidden layer
d_hidden <- 32

model <- nn_sequential(
  nn_linear(d_in, d_hidden),
  nn_relu(),
  nn_linear(d_hidden, d_out)
)

### network parameters ---------------------------------------------------------

# for adam, need to choose a much higher learning rate in this problem
learning_rate <- 0.08

optimizer <- optim_adam(model$parameters, lr = learning_rate)

### training loop --------------------------------------------------------------

for (t in 1:200) {

  ### -------- Forward pass --------

  y_pred <- model(x)

  ### -------- compute loss --------
  loss <- nnf_mse_loss(y_pred, y, reduction = "sum")
  if (t %% 10 == 0)
    cat("Epoch: ", t, "   Loss: ", loss$item(), "\n")

  ### -------- Backpropagation --------

  # Still need to zero out the gradients before the backward pass, only this time,
  # on the optimizer object
  optimizer$zero_grad()

  # gradients are still computed on the loss tensor (no change here)
  loss$backward()

  ### -------- Update weights --------

  # use the optimizer to update model parameters
  optimizer$step()
}
