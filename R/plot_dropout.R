#'plot the density of the impiuted values of a numeric variable using different dropout probabilities
#' @param tune.results object returned by tune_dropout()
#' @param var.name the name of a numeric variable
#' @param xlim the left and right limit of the x-axis. Default: NULL.
#' @param ylim the lower and upper limit of the y-axis. Default: NULL.
#' @export
plot_dropout<-function(tune.results, var.name, xlim = NULL, ylim = NULL){

  observed.vec<-tune.results$observed.data[[var.name]]
  observed.df<-data.frame(observed=observed.vec)

  longer.tbl<-lapply(tune.results$imputed.missing[[var.name]],tidyr::pivot_longer,cols = starts_with("m"),names_to="set")
  all.tbl<-do.call(rbind,longer.tbl)

  all.tbl <- all.tbl %>%
    dplyr::mutate_at(vars(-("value")),as.factor)

  P<-ggplot(data=all.tbl)+
    geom_density(aes(x=value,color=set))+
    #facet_grid(hidden.dropout~input.dropout,labeller=label_both)+
    facet_grid(hidden.dropout~input.dropout)+
    geom_density(data=observed.df,aes(x=observed))+
    labs(title=var.name)

  if(is.null(xlim)){
    P<-P+scale_x_continuous(sec.axis = sec_axis(~ . , name = "Input Dropout", breaks = NULL, labels = NULL))

  }else{
    P<-P+scale_x_continuous(limits=xlim,sec.axis = sec_axis(~ . , name = "Input Dropout", breaks = NULL, labels = NULL))

  }


  if(is.null(ylim)){
    P+ scale_y_continuous(sec.axis = sec_axis(~ . , name = "Hidden Dropout", breaks = NULL, labels = NULL))
  }else{
    P+ scale_y_continuous(limits=ylim,sec.axis = sec_axis(~ . , name = "Hidden Dropout", breaks = NULL, labels = NULL))

  }



}



