#library(dplyr)
library(palmerpenguins)

library(devtools)
devtools::document()
devtools::load_all()
devtools::build()


set.seed(2022)
miae.imputed<-miae(data=penguins,encoder.structure = c(32,16),decoder.structure = c(16,32))
#miae.imputed<-miae(data=penguins)
#miae.imputed
colSums(is.na(penguins))
show_var(imputation.list=miae.imputed,var.name="bill_length_mm",original.data = penguins)



##
set.seed(2022)
midae.imputed<-midae(data=penguins,m=5,epochs=10,
                     encoder.structure = c(32,16),decoder.structure = c(16,32),
                     dropout.prob = 0.5)
#midae.imputed
#colSums(is.na(penguins))
show_var(imputation.list=midae.imputed,var.name="bill_length_mm",original.data = penguins)
#show_var(imputation.list=mivae.imputed,var.name="bill_depth_mm",original.data = penguins)
#show_var(imputation.list=mivae.imputed,var.name="flipper_length_mm",original.data = penguins)
#show_var(imputation.list=mivae.imputed,var.name="body_mass_g",original.data = penguins)
#show_var(imputation.list=mivae.imputed,var.name="sex",original.data = penguins)





set.seed(2022)
mivae.imputed<-mivae(data=penguins,m=5,epochs=10,
                     encoder.structure = c(32,16),decoder.structure = c(16,32))
#mivae.imputed
#colSums(is.na(penguins))
show_var(imputation.list=mivae.imputed,var.name="bill_length_mm",original.data = penguins)
#show_var(imputation.list=mivae.imputed,var.name="bill_depth_mm",original.data = penguins)
#show_var(imputation.list=mivae.imputed,var.name="flipper_length_mm",original.data = penguins)
#show_var(imputation.list=mivae.imputed,var.name="body_mass_g",original.data = penguins)
#show_var(imputation.list=mivae.imputed,var.name="sex",original.data = penguins)




##data.frame


withNA.df<-createNA(iris)

set.seed(2022)
midae.imputed<-midae(data=withNA.df,m=5,epochs=5,
                     encoder.structure = c(32,16),decoder.structure = c(16,32),
                     dropout.prob = 0.5)
#midae.imputed
#colSums(is.na(withNA.df))
show_var(imputation.list=midae.imputed,var.name="Sepal.Length",original.data = withNA.df)



set.seed(2022)
mivae.imputed<-mivae(data=withNA.df,m=5,epochs=5,
                     encoder.structure = c(32,16),decoder.structure = c(16,32))
#midae.imputed
#colSums(is.na(withNA.df))
show_var(imputation.list=mivae.imputed,var.name="Sepal.Length",original.data = withNA.df)
