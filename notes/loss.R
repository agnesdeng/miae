library(torch)
torch_manual_seed(1234)
input <- torch_randn(3, requires_grad = TRUE)
#$random_(1,2): generate integers range from 1 to 2 not including 2
target <- torch_empty(3)$random_(0, 2)

m <- nn_sigmoid()

bce.loss <- nn_bce_loss()
bcelogit.loss <- nn_bce_with_logits_loss()

m(input)
input
target
ibce.loss(m(input), target)
bcelogit.loss(input, target)

1*0.5115

torch_manual_seed(1234)
#0
input <- torch_tensor(0)
#m(0)=0.5
m(input)
#input <- torch_randn(1, requires_grad = TRUE)
#$random_(1,2): generate integers range from 1 to 2 not including 2
target <- torch_ones(1)

bce.loss <- nn_bce_loss(reduction="sum")
bcelogit.loss <- nn_bce_with_logits_loss()


bce.loss(m(input), target)
bcelogit.loss(input, target)

-1*log(0.5115)-1*log(1-0.5993)-1*log(1-0.2667)

(-1*log(0.5115)-1*log(1-0.5993)-1*log(1-0.2667))/3


-1*log(m(input)[1])-1*log(1-m(input)[2])-1*log(1-m(input)[3])


input
target

input
m(input)
target



##cce
loss <- nn_cross_entropy_loss(reduction="sum")
input <- torch_randn(3, 5, requires_grad = TRUE)
input
#1,2,3,4
target <- torch_randint(low = 1, high = 5, size = 3, dtype = torch_long())
target
output <- loss(input, target)


torch_empty(3,dtype=torch_long())$random_(1,5)

torch_randn(3,5)
raw.p=torch_randn(3,5)
m=nn_softmax(dim=2)
m(raw.p)

#input, predicted probabiliteis for each of the 5 classes
input <- torch_randn(3, 5, requires_grad = TRUE)
input

target.raw <- torch_randn(3, 5, requires_grad = TRUE)
target.raw
m=nn_softmax(dim=2)
#row sum to one, do softmax on each row
target<-m(target.raw)
#true probabilities of the target
target
sum(target[1,])
loss <- nn_cross_entropy_loss()
loss(input,target)




input
torch_empty(3,dtype=torch_long())
#1,2,3,4
torch_empty(3,dtype=torch_long())$random_(from = 1, to =5)
#0,1,2,3,4
torch_empty(3,dtype=torch_long())$random_(from = 0, to =5)

#1,2,3,4,5
torch_empty(3,dtype=torch_long())$random_(from = 1, to =6)



input <- torch_randn(3, 5, requires_grad = TRUE)
input

target<-torch_empty(3,dtype=torch_long())
target.labels=target$random_(from = 1, to =6)
target.labels
loss(input,target.labels)

target.labels2=target.labels+1
target.labels2$to(dtype=torch_long())
target.labels2<-target.labels2$to(dtype=torch_long())
target.labels2
loss(input,target.labels2)

m(input)
#torch dimension
mat=matrix(c(1,0,1,0,1,0),nrow=2,ncol=3)
mat

mat.torch=torch_tensor(mat)
mat.torch
#dim is more like axis
#do column sums, leave one row of 3 values
torch_sum(mat.torch,dim=1)
#do row sum, leave one column of 2 values
torch_sum(mat.torch,dim=2)


#
torch_randn(3,5)
raw.p=torch_randn(3,5)
raw.p
#do colum wise softmax
m=nn_softmax(dim=1)
m(raw.p)

sum.all=sum(exp(raw.p[,1]))
exp(raw.p[1,1])/sum.all
m(raw.p)



#do row wise softmax
m=nn_softmax(dim=2)
m(raw.p)

sum.all=sum(exp(raw.p[1,]))
exp(raw.p[1,1])/sum.all
m(raw.p)



exp(raw.p[1,1])/sum.all

torch_manual_seed(1234)
bcelogit.loss <- nn_bce_with_logits_loss(reduction = "sum")

output <- loss(input, target)

target
output




input

target

output


torch_manual_seed(1234)
loss <- nn_bce_with_logits_loss(reduction = "sum")
input <- torch_randn(1, requires_grad = TRUE)
target <- torch_empty(1)$random_(1, 2)
output <- loss(input, target)


input

target

output


torch_empty(3)$random_(1, 2)
torch_empty(3)
output$backward()

target <- torch_ones(10, 64, dtype = torch_float32()) # 64 classes, batch size = 10
output <- torch_full(c(10, 64), 1.5) # A prediction (logit)
pos_weight <- torch_ones(64) # All weights are equal to 1
criterion <- nn_bce_with_logits_loss(pos_weight = pos_weight)
criterion(output, target) # -log(sigmoid(1.5))



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



m <- nn_sigmoid()
loss <- nn_bce_loss()
input <- torch_randn(3, requires_grad = TRUE)
target <- torch_rand(3)
output <- loss(m(input), target)
output$backward()
