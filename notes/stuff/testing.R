

setwd("C:/Users/agnes/Desktop/phd-thesis/my-packages/miae")

devtools::load_all()
devtools::document()

library(grid)
library(ggplot2)



setwd("C:/Users/agnes/Desktop/phd-thesis/my-packages/vismi")

devtools::load_all()
devtools::document()

setwd("C:/Users/agnes/Desktop/phd-thesis/Thesis/thesis/R/CH5 miae/supplement/hyperparameter")

#library(ggpubr)
#library(Amelia)
#library(mice)
#library(mixgb)
#library(miae)

source("sim/common.R")


full.df <- readRDS("data/full.rds")

set.seed(1234)
withNA.df <- mar1_data(full.df)

colSums(is.na(withNA.df))

model.dir<-"C:/Users/agnes/Desktop/phd-thesis/Thesis/thesis/R/CH5 miae/Temp"
dir.path <- "C:/Users/agnes/Desktop/phd-thesis/Thesis/thesis/figures/miae"













# ----------------------------------------------------------

# epoch 100 using ELU and midas default---------------------------------------------------------------


torch_manual_seed(2023)
epoch100vaeM<-mivae(data=withNA.df, m = 5,
                      categorical.encoding = "onehot", device = "cpu",
                      epochs = 100, batch.size = 50,
                      early.stopping.epochs = 1, subsample = 1,
                      vae.params = list(shuffle = TRUE, drop.last = FALSE,
                                        input.dropout = 0, hidden.dropout = 0,
                                        optimizer = "adamW", learning.rate = 0.0001, weight.decay = 0.002, momentum = 0, dampening = 0, eps = 1e-08, rho = 0.9, alpha = 0.99, learning.rate.decay = 0,
                                        encoder.structure = c(128, 64, 32), latent.dim = 16, decoder.structure = c(32, 64, 128),
                                        act = "elu", init.weight = "xavier.midas", scaler = "none",initial.imp = "sample", lower=0.25, upper=0.75),
                      pmm.params = list(pmm.type = NULL, pmm.k = 5, pmm.link = "prob", pmm.save.vars = NULL),
                      loss.na.scale = FALSE,
                      verbose = TRUE,print.every.n = 1,
                      save.model = FALSE,
                      path = file.path(model.dir, "mivaemodel.pt"))

p1<-plot_hist(imputation.list=epoch100vaeM, var.name="X1", original.data = withNA.df,true.data=full.df,
              color.pal = c("black","gray50","#4E72FF", "springgreen3", "orange","orchid2","burlywood4"))+
  theme(
    plot.title = element_blank(),
    plot.subtitle = element_blank(),
    strip.text = element_text(size = 23, face = "plain"),
    axis.title.x = element_text(size = 26, margin = margin(t = 10, r = 0, b = 0, l = 0), ),
    axis.title.y = element_text(size = 26, margin = margin(0, r = 5, 0, l = 0)),
    axis.text.x = element_text(size = 21),
    axis.text.y = element_text(size = 21),
    panel.spacing.x = unit(0.4, "cm")
  )

jpeg(
  filename = file.path(dir.path, "/epoch100x1vaeM.jpeg"),
  width = 15, height = 4, units = "in", res = 300, pointsize = 1
)
grid.draw(p1, recording = T)
dev.off()

p1<-plot_2num(imputation.list=epoch100vaeM, var.x="X3", var.y="Y", original.data = withNA.df,true.data=full.df,
              color.pal = c("black","gray50","#4E72FF", "springgreen3", "orange","orchid2","burlywood4"))+
  theme(
    plot.title = element_blank(),
    plot.subtitle = element_blank(),
    strip.text = element_text(size = 23, face = "plain"),
    axis.title.x = element_text(size = 26, margin = margin(t = 10, r = 0, b = 0, l = 0), ),
    axis.title.y = element_text(size = 26, margin = margin(0, r = 5, 0, l = 0)),
    axis.text.x = element_text(size = 21),
    axis.text.y = element_text(size = 21),
    panel.spacing.x = unit(0.4, "cm")
  )

jpeg(
  filename = file.path(dir.path, "/epoch100yx3vaeM.jpeg"),
  width = 15, height = 4, units = "in", res = 300, pointsize = 1
)
grid.draw(p1, recording = T)
dev.off()



# miae --------------------------------------------------------------------

#with he normal, large variance



torch_manual_seed(2023)
epoch100vae<-mivae(data=withNA.df, m = 5,
                    categorical.encoding = "onehot", device = "cpu",
                    epochs = 100, batch.size = 50,
                    early.stopping.epochs = 1, subsample = 1,
                    vae.params = list(shuffle = TRUE, drop.last = FALSE,
                                      input.dropout = 0, hidden.dropout = 0,
                                      optimizer = "adamW", learning.rate = 0.0001, weight.decay = 0.002, momentum = 0, dampening = 0, eps = 1e-08, rho = 0.9, alpha = 0.99, learning.rate.decay = 0,
                                      encoder.structure = c(128, 64, 32), latent.dim = 16, decoder.structure = c(32, 64, 128),
                                      act = "elu", init.weight = "he.normal.elu", scaler = "none",initial.imp = "sample", lower=0.25, upper=0.75),
                    pmm.params = list(pmm.type = NULL, pmm.k = 5, pmm.link = "prob", pmm.save.vars = NULL),
                    loss.na.scale = FALSE,
                    verbose = TRUE,print.every.n = 1,
                    save.model = FALSE,
                    path = file.path(model.dir, "mivaemodel.pt"))

p1<-plot_hist(imputation.list=epoch100vae, var.name="X1", original.data = withNA.df,true.data=full.df,
              color.pal = c("black","gray50","#4E72FF", "springgreen3", "orange","orchid2","burlywood4"))+
  theme(
    plot.title = element_blank(),
    plot.subtitle = element_blank(),
    strip.text = element_text(size = 23, face = "plain"),
    axis.title.x = element_text(size = 26, margin = margin(t = 10, r = 0, b = 0, l = 0), ),
    axis.title.y = element_text(size = 26, margin = margin(0, r = 5, 0, l = 0)),
    axis.text.x = element_text(size = 21),
    axis.text.y = element_text(size = 21),
    panel.spacing.x = unit(0.4, "cm")
  )

jpeg(
  filename = file.path(dir.path, "/epoch100x1vae.jpeg"),
  width = 15, height = 4, units = "in", res = 300, pointsize = 1
)
grid.draw(p1, recording = T)
dev.off()

p1<-plot_2num(imputation.list=epoch100vae, var.x="X3", var.y="Y", original.data = withNA.df,true.data=full.df,
              color.pal = c("black","gray50","#4E72FF", "springgreen3", "orange","orchid2","burlywood4"))+
  theme(
    plot.title = element_blank(),
    plot.subtitle = element_blank(),
    strip.text = element_text(size = 23, face = "plain"),
    axis.title.x = element_text(size = 26, margin = margin(t = 10, r = 0, b = 0, l = 0), ),
    axis.title.y = element_text(size = 26, margin = margin(0, r = 5, 0, l = 0)),
    axis.text.x = element_text(size = 21),
    axis.text.y = element_text(size = 21),
    panel.spacing.x = unit(0.4, "cm")
  )

jpeg(
  filename = file.path(dir.path, "/epoch100yx3vae.jpeg"),
  width = 15, height = 4, units = "in", res = 300, pointsize = 1
)
grid.draw(p1, recording = T)
dev.off()


