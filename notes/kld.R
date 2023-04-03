# e.g., if latent.dim=4, 
#return values in a matrix with dim(batch.size, latent.dim) 
1 + log.var - mu^2 - log.var$exp()


#reduce the rows (colsums)  (return kld values with latent.dim, e.g.,4)
-0.5 *torch_sum(1 + log.var - mu^2 - log.var$exp(), dim = 1)
#average over the 4 dimension of the latent space and take the mean, this values is the mean of the latent dimensions, but it is the sum of the whole batch
torch_mean(-0.5 * torch_sum(1 + log.var - mu^2 - log.var$exp(), dim = 1))
#this returns the mean of the latent dimensions and the mean of a batch
torch_mean(-0.5 * torch_sum(1 + log.var - mu^2 - log.var$exp(), dim = 1))/mu$size(1)
#this returns the mean of the batch but the sum of latent dimensions
torch_mean(-0.5 * torch_sum(1 + log.var - mu^2 - log.var$exp(), dim = 1))/mu$size(1)*mu$size(2)


#reduce the columns (rowSums), (return kld values with batch.size)
#the sum of 4 dimensions of latent space
-0.5 *torch_sum(1 + log.var - mu^2 - log.var$exp(), dim = 2)
#the mean over a batch of the sum of kld the latent dimensions
torch_mean(-0.5 * torch_sum(1 + log.var - mu^2 - log.var$exp(), dim = 2),dim = 1)
#equivalent to
#the mean over a batch of the sum of kld the latent dimensions
torch_mean(-0.5 * torch_sum(1 + log.var - mu^2 - log.var$exp(), dim = 2))
#equivalent to
#this returns the mean of the batch but the sum of latent dimensions
torch_mean(-0.5 * torch_sum(1 + log.var - mu^2 - log.var$exp(), dim = 1))/mu$size(1)*mu$size(2)

#not equal to this!!!!
torch_mean(-0.5 * torch_sum(1 + log.var - mu^2 - log.var$exp()))
#not equal to this!!!!
-0.5 * torch_sum(1 + log.var - mu^2 - log.var$exp())



#original implementation:wrong
#kl.divergence <- torch_mean(-0.5 * torch_sum(1 + log.var - mu$pow(2) - log.var$exp()))