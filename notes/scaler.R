## Packages
setwd("C:/Users/agnes/Desktop/phd-thesis/my-packages/miae")

library(devtools)
devtools::document()
devtools::load_all()


b=1
setwd("C:/Users/agnes/Desktop/phd-thesis/my-projects/miae-paper/midas/mar1")

withNA.df<-readr::read_csv(file = paste0("data_base/mar1_draw_",b,".csv"))
full.df<-readr::read_csv(file = paste0("data_base/mar1_full_",b,".csv"))

 

minmaxscaled.obj<-minmax_scaler(full.df)
minmaxscaled.df<-as.data.frame(minmaxscaled.obj$minmax.mat)
revminmax.df<-rev_minmax_scaler(scaled.data = minmaxscaled.df,num.names = colnames(full.df),
                          colmin=minmaxscaled.obj$colmin, colmax=minmaxscaled.obj$colmax)


standscaled.obj<-standardize_scaler(full.df)
standscaled.df<-as.data.frame(standscaled.obj$stand.mat)
revstandardize.df<-rev_standardize_scaler(scaled.data = standscaled.df,num.names = colnames(full.df),
                                          colmean=standscaled.obj$colmean, colsd=standscaled.obj$colsd)



full.tbl<-tidyr::pivot_longer(data=full.df,cols = everything(),values_to ="value",names_to = "variable" )
ggplot(data=full.tbl,aes(x=value,color=variable))+
  geom_density()



minmax.tbl<-tidyr::pivot_longer(data=minmaxscaled.df,cols = everything(),values_to ="value",names_to = "variable" )
ggplot(data=minmax.tbl,aes(x=value,color=variable))+
  geom_density()




stand.tbl<-tidyr::pivot_longer(data=standscaled.df,cols = everything(),values_to ="value",names_to = "variable" )
ggplot(data=stand.tbl,aes(x=value,color=variable))+
  geom_density()




rev.tbl<-tidyr::pivot_longer(data=rev.df,cols = everything(),values_to ="value",names_to = "variable" )
ggplot(data=rev.tbl,aes(x=value,color=variable))+
  geom_density()


  
