library(torch)


torch_ones(c(2,3))
ones<-torch_ones(3)

ones<-torch_ones(3)
zeros<-torch_zeros(3)

ones
#overwrite values
ones[1]<-3
ones[2]<-2
ones[3]<-1
ones

tensors<-torch_tensor(data=c(4,1,5,3,2,1))
tensors
tensors[1]
tensors[2]
tensors

tensors<-torch_tensor(data=matrix(c(1,2,3,4),nrow=2))
tensors
dim(tensors)

tensors[2,2]
tensors
#first row
tensors[1]
tensors[1,]
#second row
tensors[2]
tensors[2,]


lm<-nn_linear(1,1)
lm

lm(torch_tensor(2))
lm(torch_tensor(1))

lm$bias
lm$weight

x<-torch_ones(c(10,1))
x
lm(x)


x = torch_tensor(c(1, 2, 3, 4))
#row
torch_unsqueeze(x, 1)
#column
torch_unsqueeze(x, 2)



#transform
x<-torch_tensor(c(1,2,3))
m<-nn_softmax(1)
m(x)
output<-m(x)
output
output$sum()

#dim=1 column
x<-torch_tensor(matrix(c(1,1,2,2,3,3),nrow=2))
m<-nn_softmax(1)
m(x)
output<-m(x)
output
output$sum()

#dim=2 row
x<-torch_tensor(matrix(c(1,1,2,2,3,3),nrow=2))
m<-nn_softmax(2)
m(x)
output<-m(x)
output
output$sum()



#

m <- nn_sigmoid()
loss <- nn_bce_loss()
input <- torch_randn(3, requires_grad = TRUE)
input
target <- torch_rand(3)
target
output1 <- loss(m(input), target)
output1

loss2<-nn_bce_with_logits_loss()
output2<-loss2(input,target)
output2

output <- loss(input, target)
output



x = torch_randn(c(3, 2))
y = torch_ones(c(3, 2))
x
y
torch_where(x > 0, x, y)
