target = torch_ones(c(2,2))
output=torch_full(c(2,2),fill_value = 1.5)
criterion = nn_bce_with_logits_loss()
criterion(output,target)
target
output


sigmoid<-function(x){
  exp(x)/(1+exp(x))
}
1.5*log(sigmoid(1))+(1-1.5)*log(1-sigmoid(1))


loss <- nn_mse_loss()
input <- torch_randn(3, 2, requires_grad = TRUE)
target <- torch_randn(3, 2)
output <- loss(input, target)
output

c1=mean((target[,1]-input[,1])^2)
c2=mean((target[,2]-input[,2])^2)
sum(c1,c2)/2

loss <- nn_mse_loss(reduction = "sum")
output <- loss(input, target)
output
output/6
