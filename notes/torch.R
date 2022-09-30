cardinalities = c(length(levels(penguins$species)),length(levels(penguins$island)), length(levels(penguins$sex)))
cardinalities

purrr::map(cardinalities, function(x) ceiling(x/2))
sum(purrr::map(cardinalities, function(x) ceiling(x/2)) %>% unlist())

n_cont=5
sum(purrr::map(cardinalities, function(x) ceiling(x/2)) %>% unlist()) + n_cont


lapply(cardinalities, function(x) nn_embedding(num_embeddings = x, embedding_dim = ceiling(x/2)))


nn_module_list(lapply(cardinalities, function(x) nn_embedding(num_embeddings = x, embedding_dim = ceiling(x/2))))

loss <- nn_cross_entropy_loss()
input <- torch_randn(3, 5, requires_grad = TRUE)
target <- torch_randint(low = 1, high = 5, size = 3, dtype = torch_long())


output <- loss(input, target)
output
input
target


m <- nn_log_softmax(1)
input <- torch_randn(2, 3)
output <- m(input)


input
output


m <- nn_log_softmax(dim = 2)
loss <- nn_nll_loss()
# input is of size N x C = 3 x 5
input <- torch_randn(3, 5, requires_grad = TRUE)
# each element in target has to have 0 <= value < C
target <- torch_tensor(c(2, 1, 5), dtype = torch_long())
output <- loss(m(input), target)
output$backward()


input
target

loss <- nn_mse_loss()
loss.sum<-nn_mse_loss(reduction = "sum")
input <- torch_randn(3, 5, requires_grad = TRUE)
target <- torch_randn(3, 5)
output <- loss(input, target)

input<-torch_tensor(matrix(c(2,3,4,8),ncol=2))
input
target<-torch_tensor(matrix(c(3,3,5,6),ncol=2))
target
loss(input,target)
loss.sum(input,target)

((2-3)^2+0+(4-5)^2+(8-6)^2)/4

input <- torch_randn(3, 5, requires_grad = TRUE)
target <- torch_randn(3, 5)
output <- loss(input, target)

library(torch)
m1 <- nn_dropout(p = 0.2)
input <- torch_randn(10, 2)
input0<-input
output <- m1(input)
output1<-m1(input)
output2<-m1(input)

loss <- nn_mse_loss()
torch_equal(output1,output2)


m2 <- nn_dropout(p = 0.2,inplace = TRUE)

output2 <- m2(input)
output2
output3<-m2(input)
output3
torch_equal(output2,output3)
#TRUE
#after running output3, output2 will change to output3
