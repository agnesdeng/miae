#'plot the density of the impiuted values of a numeric variable using different dropout probabilities
#' @param tune.results object returned by tune_dropout()
#' @param var.name the name of a numeric variable
plot_dropout<-function(tune.results, var.name){

  observed.vec<-tune.results$observed.data[[var.name]]
  observed.df<-data.frame(observed=observed.vec)

  longer.tbl<-lapply(tune.results$imputed.missing[[var.name]],tidyr::pivot_longer,cols = starts_with("m"),names_to="set")
  all.tbl<-do.call(rbind,longer.tbl)

  all.tbl <- all.tbl %>%
    dplyr::mutate_at(vars(-("value")),as.factor)

  ggplot(data=all.tbl)+
    geom_density(aes(x=value,color=set))+
    #facet_grid(latent.dropout~input.dropout,labeller=label_both)+
    facet_grid(latent.dropout~input.dropout)+
    geom_density(data=observed.df,aes(x=observed))+
    scale_x_continuous(sec.axis = sec_axis(~ . , name = "Input Dropout", breaks = NULL, labels = NULL)) +
    scale_y_continuous(limits=c(0,1),sec.axis = sec_axis(~ . , name = "Latent Dropout", breaks = NULL, labels = NULL))


}



