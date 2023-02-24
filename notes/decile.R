rr<-replicate(1000,range(rnorm(10000)))
range(rnorm(10))
t(rr)
summary(t(rr))

quantile(rnorm(1000),probs = c(0.25,0.75))

quantile(rnorm(1000),probs = c(0.1,0.9))


quantile(rnorm(1000),probs = c(0,1))

onesample<-rnorm(1000)
summary(onesample)
summary(quantile(onesample, probs=c(0.1,0.9)))
