library(torch)
torch_manual_seed(1234)
input <- torch_randn(3, requires_grad = TRUE)
#$random_(1,2): generate integers range from 1 to 2 not including 2
target <- torch_empty(3)$random_(1, 2)

m <- nn_sigmoid()

bce.loss <- nn_bce_loss()
bcelogit.loss <- nn_bce_with_logits_loss()



bce.loss(m(input), target)
bcelogit.loss(input, target)



torch_manual_seed(1234)
#0
input <- torch_tensor(0)
#m(0)=0.5
m(input)
#input <- torch_randn(1, requires_grad = TRUE)
#$random_(1,2): generate integers range from 1 to 2 not including 2
target <- torch_ones(1)


bcelogit.loss <- nn_bce_with_logits_loss()


bce.loss(m(input), target)
bcelogit.loss(input, target)

1*log(0.5)







input
target


output <- loss(m(input), target)
output

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