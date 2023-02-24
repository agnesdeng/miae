#weight initialization
init_xavier_uniform<-function(m){
  if(any(class(m)=="nn_linear")){
    nn_init_xavier_uniform_(m$weight)
    nn_init_zeros_(m$bias)
  }

}

init_xavier_normal<-function(m){
  if(any(class(m)=="nn_linear")){
    nn_init_xavier_normal_(m$weight,gain=1.0)
    nn_init_zeros_(m$bias)
  }
}


init_xavier_midas<-function(m){
  if(any(class(m)=="nn_linear")){
    nn_init_xavier_normal_(m$weight,gain=1/sqrt(2))
    nn_init_zeros_(m$bias)
  }
}
