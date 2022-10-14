library(torch)
#Note: set.seed doesn't work!
set.seed(2022)
torch_randn(2, 3, 4)


#have to use torch_manual_seed
torch_manual_seed(2022)
torch_randn(2, 3, 4)
#2 matrices, each have 3 rows and 4 columns 


#
z6<-torch_zeros(6)
z6
m32<-torch_zeros(c(3,2))
m32

o6<-torch_ones(6)
o6
o32<-torch_ones(c(3,2))
o32

t6<-torch_tensor(6)
t6

t6<-torch_tensor(1:6)
t6
t6[2]
#seqeunce subset, 1,3,5th element
t6[1:6:2]
t6[slc(start=1,end = 6,step=2)]

#Yes: the last element
t6[-1]
#No: error
t6[N]
#Yes: from 3 to the last element (3,4,5,6)
t6[3:N]

m23<-t6$reshape(c(2,3))
m23


m<-torch_tensor(matrix(1:6,nrow=3,byrow = T))
m
m$shape
dim(m)
nrow(m)
ncol(m)
m
m[1,]
m[,1]
m[2,2]


#dropping dimensions

#By default, when indexing by a single integer, this dimension will be dropped to avoid the singleton dimension:
x <- torch_randn(2, 3)
x
x[1,]$shape
x[1,,drop = FALSE]$shape


#add dimension
x <- torch_tensor(1)
x
x$shape
x[, newaxis]$shape
x[, newaxis, newaxis]$shape


##
x <- torch_tensor(1:6)
x
x$shape
x[, newaxis]$shape
x[, newaxis, newaxis]$shape

#You can also use NULL instead of newaxis:
x[,NULL]$shape

#Sometimes we don’t know how many dimensions a tensor has, but we do know what to do with 
#the last available dimension, or the first one. To subsume all others, we can use ..:
z <- torch_tensor(1:125)$reshape(c(5,5,5))
z
z[1,..]
z[1,,]

#first row of each matrix
z[,1,]
#first column of each matrix (and put them by row)
z[,,1]
z[..,1]


x <- torch_randn(4,4)
x
x[c(1,3), c(1,3)]

x
#[row,column]
x[c(TRUE, FALSE, TRUE, FALSE), c(TRUE, FALSE, TRUE, FALSE)]



#The above examples also work if the index were long or boolean tensors, 
#instead of R vectors. It’s also possible to index with multi-dimensional boolean tensors:
x <- torch_tensor(rbind(
  c(1,2,3),
  c(4,5,6)
))
x
x[x>3]




# input dimensionality (number of input features)
d_in <- 3


# output dimensionality (number of predicted features)
d_out <- 1


# number of observations in training set
n <- 6


# create random data
# input
torch_manual_seed(2022)
x <- torch_randn(n, d_in)
x

#subseting the first column
x[, 1, drop = FALSE]
x[, 1, drop = TRUE]
x[,1]


x[,1,drop=FALSE]$shape
x[,1,drop=TRUE]$shape
x[,1]$shape





# target
torch_manual_seed(2022)
r1<-torch_randn(n, 1)
r1
s1<- x[, 1, drop = FALSE] * 0.2 - x[, 2, drop = FALSE] * 1.3 - x[, 3, drop = FALSE] * 0.5 
s1
y1<-s1+r1
y1


torch_manual_seed(2022)
r1<-torch_randn(n, 1)
r1
s2<-x[, 1] * 0.2 - x[, 2] * 1.3 - x[, 3] * 0.5
s2
y2<-s2+r1
y2

s2t<-torch_tensor(matrix(rep(as_array(s2),6),ncol=6,byrow = TRUE))
s2t
s2t+r1

torch_manual_seed(2022)
y <- x[, 1] * 0.2 - x[, 2] * 1.3 - x[, 3] * 0.5 + torch_randn(n, 1)
y


#
a6<-torch_tensor(1:6)
a6

b6<-torch_tensor(seq(0.1,0.6,by=0.1))
b6

a6+b6


a61<-torch_tensor(1:6)$reshape(c(6,1))
a61

b61<-torch_tensor(seq(0.1,0.6,by=0.1))$reshape(c(6,1))
b61

a61+b61


a6+b61
a6t<-torch_tensor(data = matrix(rep(1:6,6),ncol=6, byrow=TRUE))
a6t
a6t+b61


a61+b6
a61t<-torch_tensor(data = matrix(rep(1:6,6),ncol=6, byrow=FALSE))
a61t
a61t+b6

a = torch_randn(c(4))
a
torch_add(a, 20)
a+20



torch_diag(a6)
torch_diag_embed(a6)
torch_diagflat(a6)
torch_eye(6)

torch_full(list(2, 3), 3.141592)


a <- torch_tensor(c(1, 2, 3))
b <- torch_tensor(c(4, 5, 6))
torch_hstack(list(a,b))


a6t<-torch_transpose(a6,dim0 = 1,dim1 = 2)
  
x = torch_randn(c(2, 3))
x
torch_transpose(x, 1, 2)
sub1<-x[, 1] * 0.2
sub2<-x[, 2] * 1.3
sub3<-x[, 3] * 0.5
sub1-sub2-sub3

torch_manual_seed(2022)
r1<-torch_randn(n, 1)
r1
sub1-sub2-sub3+r1

