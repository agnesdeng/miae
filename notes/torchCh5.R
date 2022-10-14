library(torch)


Model<-function(input,params){
  params[1]*input+params[2]
}


loss_fn<-function(output,target){
  mean((output-target)^2)
}


training_loop<-function(epochs,learning_rate,params,input,target,print_every_n=1){

  for(epoch in 1:epochs){

    #if(epoch!=1){
    #zero-out the gradient
    # params$grad$zero_()
    #}

    output<-Model(input,params)
    loss<-loss_fn(output,target)
    loss
    loss$backward()


    params$grad
    params-(learning_rate*params$grad)


    with_no_grad({
      params$sub_(learning_rate * params$grad)
      params$grad$zero_()

    })




    if(epoch %% print_every_n==0){
      print(sprintf("Loss at epoch %d: %1f", epoch, loss$item()))
    }

  }
 params
}

#learning rate too small, overshoot, loss become inf
training_loop(epochs=15,learning_rate=1e-2,
              params=torch_tensor(c(1,0),requires_grad = T),
              input=input,target=target)


#change learning rate from 1e2 to 1e4
#better, but now at the end the loss decreases very slowly and eventually stalls.
training_loop(epochs=100,learning_rate=1e-4,
              params=torch_tensor(c(1,0),requires_grad = T),
              input=input,target=target)


##normalize data
inputS=input*0.1

training_loop(epochs=100,learning_rate=1e-2,
              params=torch_tensor(c(1,0),requires_grad = T),
              input=inputS,target=target)


training_loop(epochs=5000,learning_rate=1e-2,
              params=torch_tensor(c(1,0),requires_grad = T),
              input=inputS,target=target,print_every_n=500)


##

epochs=10
learning_rate=1e-4
params=torch_tensor(c(1,0),requires_grad = T)
target=torch_tensor(c(0.5,14,15,28,11,8,3,-4,6,13,21))

input=torch_tensor(c(35.7,55.9,58.2,81.9,56.3,48.9,33.9,21.8,48.4,60.4,68.4))


Target=torch_tensor(matrix(c(0.5,14,15,28,11,8,3,-4,6,13,21,0.5,14,15,28,11,8,3,-4,6,13,21),ncol=2))

Input=torch_tensor(matrix(c(35.7,55.9,58.2,81.9,56.3,48.9,33.9,21.8,48.4,60.4,68.4,
                            35.7,55.9,58.2,81.9,56.3,48.9,33.9,21.8,48.4,60.4,68.4),ncol=2))

print_every_n=1

Model<-function(Input,params){
  params[1]*Input+params[2]
}


loss_fn1<-function(Output,Target){
  mean((Output-Target)^2)
}

loss_fn2<-nn_mse_loss()


loss_fn3<-function(Output,Target){
  L<-list()
  L[[1]]<-mean((Output[,1]-Target[,1])^2)
  L[[2]]<-mean((Output[,2]-Target[,2])^2)
  do.call(sum,L)
}


for(epoch in 1:epochs){

  #if(epoch!=1){
  #zero-out the gradient
  # params$grad$zero_()
  #}

  Output<-Model(Input,params)
  Output-Target
  (Output-Target)^2
  mean((Output-Target)^2)


  loss1<-loss_fn1(Output,Target)
  loss1

  loss2<-loss_fn2(Output,Target)
  loss2

  loss3<-loss_fn3(Output,Target)
  loss3

  params
  params$grad


  #loss1$backward()
  #loss2$backward()
  loss3$backward()

  params$grad
  params-(learning_rate*params$grad)




  with_no_grad({
    params$sub_(learning_rate * params$grad)
    params$grad$zero_()

  })




  if(epoch %% print_every_n==0){
    print(sprintf("Loss at epoch %d: %1f", epoch, loss3$item()))
  }

}
params

# notes -------------------------------------------------------------------
loss <- nn_mse_loss()
input <- torch_randn(3, 5, requires_grad = TRUE)
target <- torch_randn(3, 5)
output <- loss(input, target)
output$backward()

input
target
output



target=torch_tensor(c(0.5,14,15,28,11,8,3,-4,6,13,21))
input=torch_tensor(c(35.7,55.9,58.2,81.9,56.3,48.9,33.9,21.8,48.4,60.4,68.4))
target
input

w=torch_ones(11)
b=torch_zeros(11)

#require gradient of params
w<-torch_tensor(1,requires_grad = T)
b<-torch_tensor(0,requires_grad = T)

output0=Model0(input=input,w=w,b=b)
output0
output=Model(input=input,params=params)
output



loss<-loss_fn(output=output,target=target)
loss

loss$backward()
#check the gradient of params
params$grad
#zero-out the gradient
params$grad$zero_()
params$grad


library(torch)
epochs=5
learning_rate=1e-5
params=torch_tensor(c(1,0),requires_grad = T)
target=torch_tensor(c(0.5,14,15,28,11,8,3,-4,6,13,21))
input=torch_tensor(c(35.7,55.9,58.2,81.9,56.3,48.9,33.9,21.8,48.4,60.4,68.4))

Model<-function(input,params){
  params[1]*input+params[2]
}



params[1]*input+params[2]

loss_fn<-function(output,target){
  mean((output-target)^2)
}





Model0<-function(input,w,b){
  w*input+b
}


training_loop<-function(epochs,learning_rate,w,b,input,target){

   for(epoch in 1:epochs){

    #if(epoch!=1){
      #zero-out the gradient
     # params$grad$zero_()
    #}

    output<-Model(input,w,b)
    loss<-loss_fn(output,target)
    loss
    loss$backward()

    w$grad
    b$grad
    w-(learning_rate*w$grad)
    b-(learning_rate*b$grad)

    with_no_grad({
      #params<-params-(learning_rate*params$grad)
      w<-w$sub_(learning_rate * w$grad)
      w
      b<-b$sub_(learning_rate * b$grad)
      b

      w$grad$zero_()
      b$grad$zero_()
      })


    loss$item()

    if(epoch %% 500==0){
      print(sprintf("Loss at epoch %d: %1f\n", epoch, loss$item()))
    }



  }
  c(w,b)
}






training_loop(epochs=5000,learning_rate=1e-3,
              params=torch_tensor(c(1,0),requires_grad = T),
              input=input,target=target)

epochs=5
learning_rate=1e-2
w<-torch_tensor(1,requires_grad = T)
b<-torch_tensor(0,requires_grad = T)
#params=torch_tensor(c(1,0),requires_grad = T)
input=input
target=target
###broadcasting
x1=torch_ones(1)
x1
x2=torch_ones(2)
x2
x1*y
x2*y

##
y=torch_ones(c(3,1))
y

z=torch_ones(c(1,3))
z

y*z
##

a=torch_ones(c(2,1,1))
a

y*z*a

###
