vec<-torch_randn(5)
mat<-torch_randn(128,5)
vec
mat

out<-torch_matmul(mat,vec)
out

out$mean()
out$std()

scaling.factor<-100
mat.s<-mat/scaling.factor
mat.s


out.s<-torch_matmul(mat.s,vec)
out.s

out.s$mean()
out.s$std()

t1 = torch_randn(c(3, 4))
t2 = torch_randn(c(4))
t1
t2
torch_matmul(t1,t2)


