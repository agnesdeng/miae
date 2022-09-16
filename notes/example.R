devtools::document()
devtools::load_all()
withNA.df=createNA(data=iris,p=0.3)

#create a variational autoencoder imputer with your choice of settings or leave it as default
MIDAE=Midae$new(withNA.df,iteration=20,input_drop=0.2,hidden_drop=0.3,n_h=4L)

#training
MIDAE$train()

#impute m datasets
imputed.data=MIDAE$impute(m = 5)
