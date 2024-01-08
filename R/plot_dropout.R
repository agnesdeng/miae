#'plot the density of the impiuted values of a numeric variable using different dropout probabilities
#' @param tune.results object returned by tune_dropout()
#' @param var.name the name of a numeric variable
#' @param xlim the left and right limit of the x-axis. Default: NULL.
#' @param ylim the lower and upper limit of the y-axis. Default: NULL.
#' @export
plot_dropout<-function(tune.results, var.name, xlim = NULL, ylim = NULL){

  color.values<-c("burlywood4","#4E72FF", "springgreen3", "orange","orchid2")

  observed.vec<-tune.results$observed.data[[var.name]]
  observed.df<-data.frame(observed=observed.vec)

  longer.tbl<-lapply(tune.results$imputed.missing[[var.name]],tidyr::pivot_longer,cols = dplyr::starts_with("m"),names_to="set")
  all.tbl<-do.call(rbind,longer.tbl)

  all.tbl <- all.tbl %>%
    dplyr::mutate_at(vars(-("value")),as.factor)

  P<-ggplot(data=all.tbl)+
    geom_density(aes(x=value,color=set,fill=set),alpha=0.1,linewidth=1)+
    #facet_grid(hidden.dropout~input.dropout,labeller=label_both)+
    facet_grid(hidden.dropout~input.dropout)+
    geom_density(data=observed.df,aes(x=observed))+
    labs(x= var.name)+
    guides(color = guide_legend(title="Imputed set"),fill = guide_legend(title="Imputed set"))+
    scale_color_manual(values=color.values)+
    scale_fill_manual(values=color.values)+
    theme(
      axis.title.x = element_text(size = 26, margin = margin(t = 15, r = 0, b = 0, l = 0), ),
      axis.title.y = element_text(size = 26, margin = margin(0, r = 15, 0, l = 0)),
      axis.text.x = element_text(size = 23),
      axis.text.y = element_text(size = 23),
      panel.spacing.x = unit(0.5, "in"),
      strip.text = element_text(size = 23), # change facet text size
      legend.title = element_text(face = "bold", size = 30), # change legend title font size
      legend.text = element_text(size = 25),
      legend.position = "right",
      legend.key.size = unit(2, "cm"), # change legend key size
      legend.key.height = unit(2, "cm"), # change legend key height
      legend.key.width = unit(1.5, "cm"), # change legend key width
      legend.spacing.y = unit(1, "cm")
    )

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



