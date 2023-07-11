library(MASS)
set.seed(2023)


x1<-rnorm(1000,mean=-100,sd=40)
# Generate random numbers from the log-normal distribution
x2 <- rlnorm(1000, meanlog = 5, sdlog = 0.8)
#log(x2)~normal(5,0.8^2)

x3<-runif(1000,600,1000)

mean(x2)

x.df<-data.frame(x=c(x1,x2,x3),variable=rep(c("x1","x2","x3"),each=1000))


ggplot(data=x.df,aes(x=x,color=variable))+
  geom_density()


x1.s<-standard(x1)
x2.s<-standard(x2)
x3.s<-standard(x3)



x1.m<-minmax(x1)
x2.m<-minmax(x2)
x3.m<-minmax(x3)


x1.d<-decile(x1)
x2.d<-decile(x2)
x3.d<-decile(x3)



all.df<-data.frame(x=c(x1,x2,x3,
                       x1.s,x2.s,x3.s,
                       x1.m,x2.m,x3.m,
                       x1.d,x2.d,x3.d),
                   variable=rep(rep(c("x1","x2","x3"),each=1000),times=4),
                   scaler=rep(c("None","Standardization","Min-Max","Decile"),each=3000))


all.df$scaler<-factor(all.df$scaler,levels=c("None","Standardization","Min-Max","Decile"))

ggplot(data=all.df,aes(x=x,color=variable))+
  geom_density()+
  facet_wrap(~scaler,scales = "free")


dir.path <- "C:/Users/agnes/Desktop/phd-thesis/my-projects/miae-paper/preprint/figures"

p1<-ggplot(data=all.df,aes(x=x,color=variable))+
  geom_density()+
  facet_wrap(~scaler,scales = "free")+
  theme(
    strip.text = element_text(size = 22,face = "plain"),
    axis.title.x = element_text(size = 22, margin = margin(t = 10, r = 0, b = 0, l = 0), ),
    axis.title.y = element_text(size = 22, margin = margin(0, r = 5, 0, l = 0)),
    axis.text.x = element_text(size = 18),
    axis.text.y = element_text(size = 18),
    #legend.position = "bottom",
    #legend.box = "horizontal",
    legend.spacing.x = unit(0.17, "in"),
    legend.title = element_text(face = "bold", size = 22),
    legend.text = element_text(face = "bold", size = 18),
    legend.key.width = unit(0.4, "in"),
    legend.key.height = unit(0.4, "in"),
    legend.spacing.y = unit(0.1, "in"),
    legend.text.align = 0
  )


jpeg(
  filename = file.path(dir.path, "/scalers.jpeg"),
  width = 16, height = 8, units = "in", res = 300, pointsize = 1
)
grid.draw(p1, recording = T)
dev.off()






range(x1)
range(x1.s)
range(x1.m)
range(x1.d)

xs.df<-data.frame(x=c(x1.s,x2.s,x3.s),scaled.variable=rep(c("x1","x2","x3"),each=1000))

xm.df<-data.frame(x=c(x1.m,x2.m,x3.m),scaled.variable=rep(c("x1","x2","x3"),each=1000))


xd.df<-data.frame(x=c(x1.d,x2.d,x3.d),scaled.variable=rep(c("x1","x2","x3"),each=1000))


dir.path <- "C:/Users/agnes/Desktop/phd-thesis/my-projects/miae-paper/preprint/figures"

p1<-ggplot(data=x.df,aes(x=x,color=variable))+
  geom_density()+
  labs(title = "None")+
  guides(color = "none")

jpeg(
  filename = file.path(dir.path, "/scalernone.jpeg"),
  width = 4, height = 4, units = "in", res = 300, pointsize = 1
)
grid.draw(p1, recording = T)
dev.off()



p2<-ggplot(data=xs.df,aes(x=x,color=scaled.variable))+
  geom_density()+
  labs(title = "Standardization")+
  guides(color = "none")

jpeg(
  filename = file.path(dir.path, "/scalerstandard.jpeg"),
  width = 4, height = 4, units = "in", res = 300, pointsize = 1
)
grid.draw(p2, recording = T)
dev.off()



p3<-ggplot(data=xm.df,aes(x=x,color=scaled.variable))+
  geom_density()+
  labs(title = "Min-Max")+
  guides(color = "none")

jpeg(
  filename = file.path(dir.path, "/scalerminmax.jpeg"),
  width = 4, height = 4, units = "in", res = 300, pointsize = 1
)
grid.draw(p3, recording = T)
dev.off()


p4<-ggplot(data=xd.df,aes(x=x,color=scaled.variable))+
  geom_density()+
  labs(title = "Decile")+
  guides(color = "none")

jpeg(
  filename = file.path(dir.path, "/scalerdecile.jpeg"),
  width = 4, height = 4, units = "in", res = 300, pointsize = 1
)
grid.draw(p4, recording = T)
dev.off()

library(ggpubr)
library(grid)
combinescaler <-ggarrange(p1, p2, p3, p4)
jpeg(
  filename = file.path(dir.path, "/combinescaler.jpeg"),
  width = 15, height = 15, units = "in", res = 300, pointsize = 1
)
grid.draw(combinescaler, recording = T)
dev.off()



x1<-rnorm(1000,mean=0,sd=10)
# Generate random numbers from the log-normal distribution
x2<-rnorm(1000,mean=100,sd=30)

x2<-rnorm(1000,mean=200,sd=50)


x.df<-data.frame(x=c(x1,x2,x3),variable=rep(c("x1","x2","x3"),each=1000))


ggplot(data=x.df,aes(x=x,color=variable))+
  geom_density()


x1.s<-standard(x1)
x2.s<-standard(x2)
x3.s<-standard(x3)

xs.df<-data.frame(x=c(x1.s,x2.s,x3.s),scaled.variable=rep(c("x1","x2","x3"),each=1000))



ggplot(data=xs.df,aes(x=x,color=scaled.variable))+
  geom_density()




library(grid)
setwd("C:/Users/agnes/Desktop/phd-thesis/my-packages/miae")

devtools::load_all()
devtools::document()


setwd("C:/Users/agnes/Desktop/phd-thesis/my-packages/vismi")

devtools::load_all()
devtools::document()



# normal distribution ---------------------------------------------------

set.seed(2023)
x<-rnorm(1000,mean=100,sd=30)
x.df<-data.frame(x=x)
withNA.df<-createNA(x.df,p=0.5)

str(withNA.df)



torch::torch_manual_seed(2023)
vae.none=list(encoder.structure = c(2), latent.dim = 2, decoder.structure = c(2),scaler="none")
vae.standard=list(encoder.structure = c(2), latent.dim = 2, decoder.structure = c(2),scaler="standard")
vae.minmax=list(encoder.structure = c(2), latent.dim = 2, decoder.structure = c(2),scaler="minmax")
vae.decile=list(encoder.structure = c(2), latent.dim = 2, decoder.structure = c(2),scaler="decile")

data.none<-mivae(data = withNA.df,m=5,epochs = 10, batch.size = 100,vae.params=vae.none,
          pmm.params=list(pmm.type=NULL),path="C:/Users/agnes/Desktop/phd-thesis/Thesis/R/temp/mivaemodel.pt")



data.standard<-mivae(data = withNA.df,m=5,epochs = 10, batch.size = 100,vae.params=vae.standard,
                pmm.params=list(pmm.type=NULL),path="C:/Users/agnes/Desktop/phd-thesis/Thesis/R/temp/mivaemodel.pt")


data.minmax<-mivae(data = withNA.df,m=5,epochs = 10, batch.size = 100,vae.params=vae.minmax,
                     pmm.params=list(pmm.type=NULL),path="C:/Users/agnes/Desktop/phd-thesis/Thesis/R/temp/mivaemodel.pt")

data.decile<-mivae(data = withNA.df,m=5,epochs = 10, batch.size = 100,vae.params=vae.decile,
                     pmm.params=list(pmm.type=NULL),path="C:/Users/agnes/Desktop/phd-thesis/Thesis/R/temp/mivaemodel.pt")


setwd("C:/Users/agnes/Desktop/phd-thesis/my-packages/vismi")

devtools::load_all()
devtools::document()

plot_density_facet(imputation.lists<-list(data.none,data.standard,data.minmax,data.decile),
                   methods<-c("None","Standard","Minmax","Decile"),
                   var.name = "x",original.data = withNA.df,true.data=x.df)


# save figures
dir.path <- "C:/Users/agnes/Desktop/phd-thesis/my-projects/miae-paper/preprint/figures"
p1 <- plot_density_facet(imputation.lists<-list(data.none,data.standard,data.minmax,data.decile),
                         methods<-c("None","Standard","Minmax","Decile"),
                         var.name = "x",original.data = withNA.df,true.data=x.df)

p1<-p1+  theme(
  plot.title = element_blank(),
  plot.subtitle = element_blank(),
  strip.text = element_text(size = 18,face = "plain"),
  axis.title.x = element_text(size = 22, margin = margin(t = 10, r = 0, b = 0, l = 0), ),
  axis.title.y = element_text(size = 22, margin = margin(0, r = 5, 0, l = 0)),
  axis.text.x = element_text(size = 18),
  axis.text.y = element_text(size = 18),
  #legend.position = "bottom",
  #legend.box = "horizontal",
  legend.spacing.x = unit(0.17, "in"),
  legend.title = element_text(face = "bold", size = 18),
  legend.text = element_text(face = "bold", size = 14),
  legend.key.width = unit(0.5, "in"),
  legend.key.height = unit(0.3, "in"),
  legend.text.align = 0
)
jpeg(
  filename = file.path(dir.path, "/scaler.jpeg"),
  width = 15, height = 6, units = "in", res = 300, pointsize = 1
)
grid.draw(p1, recording = T)
dev.off()





# 2-------------------------------------------------------------------------

set.seed(2023)
x<-runif(1000,0,100)
x.df<-data.frame(x=x)
withNA.df<-createNA(x.df,p=0.5)

str(withNA.df)



torch::torch_manual_seed(2023)
vae.none=list(encoder.structure = c(2), latent.dim = 2, decoder.structure = c(2),scaler="none")
vae.standard=list(encoder.structure = c(2), latent.dim = 2, decoder.structure = c(2),scaler="standard")
vae.minmax=list(encoder.structure = c(2), latent.dim = 2, decoder.structure = c(2),scaler="minmax")
vae.decile=list(encoder.structure = c(2), latent.dim = 2, decoder.structure = c(2),scaler="decile")

data.none<-mivae(data = withNA.df,m=5,epochs = 10, batch.size = 100,vae.params=vae.none,
                 pmm.params=list(pmm.type=NULL),path="C:/Users/agnes/Desktop/phd-thesis/Thesis/R/temp/mivaemodel.pt")



data.standard<-mivae(data = withNA.df,m=5,epochs = 10, batch.size = 100,vae.params=vae.standard,
                     pmm.params=list(pmm.type=NULL),path="C:/Users/agnes/Desktop/phd-thesis/Thesis/R/temp/mivaemodel.pt")


data.minmax<-mivae(data = withNA.df,m=5,epochs = 10, batch.size = 100,vae.params=vae.minmax,
                   pmm.params=list(pmm.type=NULL),path="C:/Users/agnes/Desktop/phd-thesis/Thesis/R/temp/mivaemodel.pt")

data.decile<-mivae(data = withNA.df,m=5,epochs = 10, batch.size = 100,vae.params=vae.decile,
                   pmm.params=list(pmm.type=NULL),path="C:/Users/agnes/Desktop/phd-thesis/Thesis/R/temp/mivaemodel.pt")



plot_density_facet(imputation.lists<-list(data.none,data.standard,data.minmax,data.decile),
                   methods<-c("None","Standard","Minmax","Decile"),
                   var.name = "x",original.data = withNA.df,true.data=x.df)


# save figures
dir.path <- "C:/Users/agnes/Desktop/phd-thesis/my-projects/miae-paper/preprint/figures"
p1 <- plot_density_facet(imputation.lists<-list(data.none,data.standard,data.minmax,data.decile),
                         methods<-c("None","Standard","Minmax","Decile"),
                         var.name = "x",original.data = withNA.df,true.data=x.df)

p1<-p1+  theme(
  plot.title = element_blank(),
  plot.subtitle = element_blank(),
  strip.text = element_text(size = 18,face = "plain"),
  axis.title.x = element_text(size = 22, margin = margin(t = 10, r = 0, b = 0, l = 0), ),
  axis.title.y = element_text(size = 22, margin = margin(0, r = 5, 0, l = 0)),
  axis.text.x = element_text(size = 18),
  axis.text.y = element_text(size = 18),
  #legend.position = "bottom",
  #legend.box = "horizontal",
  legend.spacing.x = unit(0.17, "in"),
  legend.title = element_text(face = "bold", size = 18),
  legend.text = element_text(face = "bold", size = 14),
  legend.key.width = unit(0.5, "in"),
  legend.key.height = unit(0.3, "in"),
  legend.text.align = 0
)
jpeg(
  filename = file.path(dir.path, "/scaler2.jpeg"),
  width = 15, height = 6, units = "in", res = 300, pointsize = 1
)
grid.draw(p1, recording = T)
dev.off()



# 3 -----------------------------------------------------------------------

set.seed(2023)
x<-rnorm(1000,mean=0,sd=1)
x.df<-data.frame(x=x)
withNA.df<-createNA(x.df,p=0.5)

str(withNA.df)



torch::torch_manual_seed(2023)
vae.none=list(encoder.structure = c(2), latent.dim = 2, decoder.structure = c(2),scaler="none")
vae.standard=list(encoder.structure = c(2), latent.dim = 2, decoder.structure = c(2),scaler="standard")
vae.minmax=list(encoder.structure = c(2), latent.dim = 2, decoder.structure = c(2),scaler="minmax")
vae.decile=list(encoder.structure = c(2), latent.dim = 2, decoder.structure = c(2),scaler="decile")

data.none<-mivae(data = withNA.df,m=5,epochs = 10, batch.size = 100,vae.params=vae.none,
                 pmm.params=list(pmm.type=NULL),path="C:/Users/agnes/Desktop/phd-thesis/Thesis/R/temp/mivaemodel.pt")



data.standard<-mivae(data = withNA.df,m=5,epochs = 10, batch.size = 100,vae.params=vae.standard,
                     pmm.params=list(pmm.type=NULL),path="C:/Users/agnes/Desktop/phd-thesis/Thesis/R/temp/mivaemodel.pt")


data.minmax<-mivae(data = withNA.df,m=5,epochs = 10, batch.size = 100,vae.params=vae.minmax,
                   pmm.params=list(pmm.type=NULL),path="C:/Users/agnes/Desktop/phd-thesis/Thesis/R/temp/mivaemodel.pt")

data.decile<-mivae(data = withNA.df,m=5,epochs = 10, batch.size = 100,vae.params=vae.decile,
                   pmm.params=list(pmm.type=NULL),path="C:/Users/agnes/Desktop/phd-thesis/Thesis/R/temp/mivaemodel.pt")



plot_density_facet(imputation.lists<-list(data.none,data.standard,data.minmax,data.decile),
                   methods<-c("None","Standard","Minmax","Decile"),
                   var.name = "x",original.data = withNA.df,true.data=x.df)


# save figures
dir.path <- "C:/Users/agnes/Desktop/phd-thesis/my-projects/miae-paper/preprint/figures"
p1 <- plot_density_facet(imputation.lists<-list(data.none,data.standard,data.minmax,data.decile),
                         methods<-c("None","Standard","Minmax","Decile"),
                         var.name = "x",original.data = withNA.df,true.data=x.df)

p1<-p1+  theme(
  plot.title = element_blank(),
  plot.subtitle = element_blank(),
  strip.text = element_text(size = 18,face = "plain"),
  axis.title.x = element_text(size = 22, margin = margin(t = 10, r = 0, b = 0, l = 0), ),
  axis.title.y = element_text(size = 22, margin = margin(0, r = 5, 0, l = 0)),
  axis.text.x = element_text(size = 18),
  axis.text.y = element_text(size = 18),
  #legend.position = "bottom",
  #legend.box = "horizontal",
  legend.spacing.x = unit(0.17, "in"),
  legend.title = element_text(face = "bold", size = 18),
  legend.text = element_text(face = "bold", size = 14),
  legend.key.width = unit(0.5, "in"),
  legend.key.height = unit(0.3, "in"),
  legend.text.align = 0
)
jpeg(
  filename = file.path(dir.path, "/scaler3.jpeg"),
  width = 15, height = 6, units = "in", res = 300, pointsize = 1
)
grid.draw(p1, recording = T)
dev.off()


# 4 -----------------------------------------------------------------------


set.seed(2023)
x<-runif(1000,0,255)
x.df<-data.frame(x=x)
withNA.df<-createNA(x.df,p=0.5)

str(withNA.df)



torch::torch_manual_seed(2023)
vae.none=list(encoder.structure = c(2), latent.dim = 2, decoder.structure = c(2),scaler="none")
vae.standard=list(encoder.structure = c(2), latent.dim = 2, decoder.structure = c(2),scaler="standard")
vae.minmax=list(encoder.structure = c(2), latent.dim = 2, decoder.structure = c(2),scaler="minmax")
vae.decile=list(encoder.structure = c(2), latent.dim = 2, decoder.structure = c(2),scaler="decile")

data.none<-mivae(data = withNA.df,m=5,epochs = 10, batch.size = 100,vae.params=vae.none,
                 pmm.params=list(pmm.type=NULL),path="C:/Users/agnes/Desktop/phd-thesis/Thesis/R/temp/mivaemodel.pt")



data.standard<-mivae(data = withNA.df,m=5,epochs = 10, batch.size = 100,vae.params=vae.standard,
                     pmm.params=list(pmm.type=NULL),path="C:/Users/agnes/Desktop/phd-thesis/Thesis/R/temp/mivaemodel.pt")


data.minmax<-mivae(data = withNA.df,m=5,epochs = 10, batch.size = 100,vae.params=vae.minmax,
                   pmm.params=list(pmm.type=NULL),path="C:/Users/agnes/Desktop/phd-thesis/Thesis/R/temp/mivaemodel.pt")

data.decile<-mivae(data = withNA.df,m=5,epochs = 10, batch.size = 100,vae.params=vae.decile,
                   pmm.params=list(pmm.type=NULL),path="C:/Users/agnes/Desktop/phd-thesis/Thesis/R/temp/mivaemodel.pt")



plot_density_facet(imputation.lists<-list(data.none,data.standard,data.minmax,data.decile),
                   methods<-c("None","Standard","Minmax","Decile"),
                   var.name = "x",original.data = withNA.df,true.data=x.df)


# save figures
dir.path <- "C:/Users/agnes/Desktop/phd-thesis/my-projects/miae-paper/preprint/figures"
p1 <- plot_density_facet(imputation.lists<-list(data.none,data.standard,data.minmax,data.decile),
                         methods<-c("None","Standard","Minmax","Decile"),
                         var.name = "x",original.data = withNA.df,true.data=x.df)

p1<-p1+  theme(
  plot.title = element_blank(),
  plot.subtitle = element_blank(),
  strip.text = element_text(size = 18,face = "plain"),
  axis.title.x = element_text(size = 22, margin = margin(t = 10, r = 0, b = 0, l = 0), ),
  axis.title.y = element_text(size = 22, margin = margin(0, r = 5, 0, l = 0)),
  axis.text.x = element_text(size = 18),
  axis.text.y = element_text(size = 18),
  #legend.position = "bottom",
  #legend.box = "horizontal",
  legend.spacing.x = unit(0.17, "in"),
  legend.title = element_text(face = "bold", size = 18),
  legend.text = element_text(face = "bold", size = 14),
  legend.key.width = unit(0.5, "in"),
  legend.key.height = unit(0.3, "in"),
  legend.text.align = 0
)
jpeg(
  filename = file.path(dir.path, "/scaler3.jpeg"),
  width = 15, height = 6, units = "in", res = 300, pointsize = 1
)
grid.draw(p1, recording = T)
dev.off()

#



# 5 -----------------------------------------------------------------------


# Generate random numbers from the log-normal distribution
x<-rbeta(n=1000, shape1=2, shape2=5, ncp = 0)
x.df<-data.frame(x=x)
withNA.df<-createNA(x.df,p=0.5)

torch::torch_manual_seed(2023)
vae.none=list(encoder.structure = c(2), latent.dim = 2, decoder.structure = c(2),scaler="none")
vae.standard=list(encoder.structure = c(2), latent.dim = 2, decoder.structure = c(2),scaler="standard")
vae.minmax=list(encoder.structure = c(2), latent.dim = 2, decoder.structure = c(2),scaler="minmax")
vae.decile=list(encoder.structure = c(2), latent.dim = 2, decoder.structure = c(2),scaler="decile")

data.none<-mivae(data = withNA.df,m=5,epochs = 10, batch.size = 100,vae.params=vae.none,
                 pmm.params=list(pmm.type=NULL),path="C:/Users/agnes/Desktop/phd-thesis/Thesis/R/temp/mivaemodel.pt")



data.standard<-mivae(data = withNA.df,m=5,epochs = 10, batch.size = 100,vae.params=vae.standard,
                     pmm.params=list(pmm.type=NULL),path="C:/Users/agnes/Desktop/phd-thesis/Thesis/R/temp/mivaemodel.pt")


data.minmax<-mivae(data = withNA.df,m=5,epochs = 10, batch.size = 100,vae.params=vae.minmax,
                   pmm.params=list(pmm.type=NULL),path="C:/Users/agnes/Desktop/phd-thesis/Thesis/R/temp/mivaemodel.pt")

data.decile<-mivae(data = withNA.df,m=5,epochs = 10, batch.size = 100,vae.params=vae.decile,
                   pmm.params=list(pmm.type=NULL),path="C:/Users/agnes/Desktop/phd-thesis/Thesis/R/temp/mivaemodel.pt")



plot_density_facet(imputation.lists<-list(data.none,data.standard,data.minmax,data.decile),
                   methods<-c("None","Standard","Minmax","Decile"),
                   var.name = "x",original.data = withNA.df,true.data=x.df)


# save figures
dir.path <- "C:/Users/agnes/Desktop/phd-thesis/my-projects/miae-paper/preprint/figures"
p1 <- plot_density_facet(imputation.lists<-list(data.none,data.standard,data.minmax,data.decile),
                         methods<-c("None","Standard","Minmax","Decile"),
                         var.name = "x",original.data = withNA.df,true.data=x.df)

p1<-p1+  theme(
  plot.title = element_blank(),
  plot.subtitle = element_blank(),
  strip.text = element_text(size = 18,face = "plain"),
  axis.title.x = element_text(size = 22, margin = margin(t = 10, r = 0, b = 0, l = 0), ),
  axis.title.y = element_text(size = 22, margin = margin(0, r = 5, 0, l = 0)),
  axis.text.x = element_text(size = 18),
  axis.text.y = element_text(size = 18),
  #legend.position = "bottom",
  #legend.box = "horizontal",
  legend.spacing.x = unit(0.17, "in"),
  legend.title = element_text(face = "bold", size = 18),
  legend.text = element_text(face = "bold", size = 14),
  legend.key.width = unit(0.5, "in"),
  legend.key.height = unit(0.3, "in"),
  legend.text.align = 0
)
jpeg(
  filename = file.path(dir.path, "/scaler4.jpeg"),
  width = 15, height = 6, units = "in", res = 300, pointsize = 1
)
grid.draw(p1, recording = T)
dev.off()

library(MASS)

# Set parameters for the log-normal distribution
meanlog <- 10     # Mean of the underlying normal distribution in logarithmic scale
sdlog <- 0.5      # Standard deviation of the underlying normal distribution in logarithmic scale
n <- 1000         # Number of data points to generate

# Generate random numbers from the log-normal distribution
x <- rlnorm(1000, meanlog = 10, sdlog = 0.5)


set.seed(2023)
x<-rbeta(n=1000, shape1=2, shape2=5, ncp = 0)
x.df<-data.frame(x=x)

x.d<-decile(x)
x.s<-standard(x)


x.d<-decile(x)
x.df<-data.frame(x=x)
withNA.df<-createNA(x.df,p=0.5)

str(withNA.df)
range(x)
range(x.d)
range(x.s)
x[581]
which(x.d< -0.19)

range(data.none[[1]])
0.18363328

quantile(x,  probs = c(0.1))
quantile(x,  probs = c(0.9))
(quantile(x,  probs = c(0.9)) - quantile(x,  probs = c(0.1)))

(x[581] - quantile(x,  probs = c(0.1)))/(quantile(x,  probs = c(0.9)) - quantile(x,  probs = c(0.1)))

x.d<-decile_scaler(data = x.df)
x2<-rev_decile_scaler(scaled.data =x.d$decile.mat,num.names = "x",decile1=x.d$decile1, decile9=x.d$decile9 )


decile9=quantile(x,  probs = c(0.9))
decile1=quantile(x,  probs = c(0.1))
x.t<-data.none[[1]]$x * (decile9 - decile1) + decile1
cbind(x,x.d,x2,data.none[[1]]$x,x.t )

hist(x)
hist(data.none[[1]]$x[is.na(withNA.df$x)])

hist(x[is.na(withNA.df$x)])


hist(x.t)

class(data.none[[1]]$x)

torch::torch_manual_seed(2023)
vae.none=list(encoder.structure = c(2), latent.dim = 2, decoder.structure = c(2),scaler="none")
vae.standard=list(encoder.structure = c(2), latent.dim = 2, decoder.structure = c(2),scaler="standard")
vae.minmax=list(encoder.structure = c(2), latent.dim = 2, decoder.structure = c(2),scaler="minmax")
vae.decile=list(encoder.structure = c(2), latent.dim = 2, decoder.structure = c(2),scaler="decile")

data.none<-mivae(data = withNA.df,m=5,epochs = 30, batch.size = 100,vae.params=vae.none,
                 pmm.params=list(pmm.type=NULL),path="C:/Users/agnes/Desktop/phd-thesis/Thesis/R/temp/mivaemodel.pt")



data.standard<-mivae(data = withNA.df,m=5,epochs = 30, batch.size = 100,vae.params=vae.standard,
                     pmm.params=list(pmm.type=NULL),path="C:/Users/agnes/Desktop/phd-thesis/Thesis/R/temp/mivaemodel.pt")


data.minmax<-mivae(data = withNA.df,m=5,epochs = 30, batch.size = 100,vae.params=vae.minmax,
                   pmm.params=list(pmm.type=NULL),path="C:/Users/agnes/Desktop/phd-thesis/Thesis/R/temp/mivaemodel.pt")

data.decile<-mivae(data = withNA.df,m=5,epochs = 10, batch.size = 100,vae.params=vae.decile,
                   pmm.params=list(pmm.type=NULL),path="C:/Users/agnes/Desktop/phd-thesis/Thesis/R/temp/mivaemodel.pt")



plot_density_facet(imputation.lists<-list(data.none,data.standard,data.minmax,data.decile),
                   methods<-c("None","Standard","Minmax","Decile"),
                   var.name = "x",original.data = withNA.df,true.data=x.df)


# save figures
dir.path <- "C:/Users/agnes/Desktop/phd-thesis/my-projects/miae-paper/preprint/figures"
p1 <- plot_density_facet(imputation.lists<-list(data.none,data.standard,data.minmax,data.decile),
                         methods<-c("None","Standard","Minmax","Decile"),
                         var.name = "x",original.data = withNA.df,true.data=x.df)

p1<-p1+  theme(
  plot.title = element_blank(),
  plot.subtitle = element_blank(),
  strip.text = element_text(size = 18,face = "plain"),
  axis.title.x = element_text(size = 22, margin = margin(t = 10, r = 0, b = 0, l = 0), ),
  axis.title.y = element_text(size = 22, margin = margin(0, r = 5, 0, l = 0)),
  axis.text.x = element_text(size = 18),
  axis.text.y = element_text(size = 18),
  #legend.position = "bottom",
  #legend.box = "horizontal",
  legend.spacing.x = unit(0.17, "in"),
  legend.title = element_text(face = "bold", size = 18),
  legend.text = element_text(face = "bold", size = 14),
  legend.key.width = unit(0.5, "in"),
  legend.key.height = unit(0.3, "in"),
  legend.text.align = 0
)
jpeg(
  filename = file.path(dir.path, "/scaler3.jpeg"),
  width = 15, height = 6, units = "in", res = 300, pointsize = 1
)
grid.draw(p1, recording = T)
dev.off()

#

imputation.list = data.none
var.name = "x"
original.data = withNA.df
true.data=x.df




set.seed(2023)
x<-runif(1000,0,1)
x.df<-data.frame(x=x)
withNA.df<-createNA(x.df,p=0.5)

str(withNA.df)



torch::torch_manual_seed(2023)
vae.none=list(encoder.structure = c(2), latent.dim = 2, decoder.structure = c(2),scaler="none",act="identity")
vae.standard=list(encoder.structure = c(2), latent.dim = 2, decoder.structure = c(2),scaler="standard")
vae.minmax=list(encoder.structure = c(2), latent.dim = 2, decoder.structure = c(2),scaler="minmax")
vae.decile=list(encoder.structure = c(2), latent.dim = 2, decoder.structure = c(2),scaler="decile")

data.none<-mivae(data = withNA.df,m=5,epochs = 20, batch.size = 100,vae.params=vae.none,
                 pmm.params=list(pmm.type=NULL),path="C:/Users/agnes/Desktop/phd-thesis/Thesis/R/temp/mivaemodel.pt")

range(x)
range(data.none[[1]]$x)

original.df<-data.frame(x=c(10,30,50,70,90))
minmax.obj<-minmax_scaler(original.df)

rev_minmax_scaler(scaled.data = minmax.obj$minmax.mat,num.names = "x", colmin = minmax.obj$colmin,colmax = minmax.obj$colmax)

trained.mat<-minmax.obj$minmax.mat
trained.mat[,1]<-c(0,0.1,0.4,0.9,1)
rev_minmax_scaler(scaled.data = trained.mat,num.names = "x", colmin = minmax.obj$colmin,colmax = minmax.obj$colmax)


trained.mat[,1]<-c(0,0.1,0.4,0.9,1)
rev_minmax_scaler(scaled.data = trained.mat,num.names = "x", colmin = minmax.obj$colmin,colmax = minmax.obj$colmax)




range(data.minmax[[1]]$x)

range(x.df$x)
range(withNA.df$x,na.rm=T)


data = withNA.df
m=5
epochs = 100
batch.size = 100
vae.params=list(shuffle = TRUE, drop.last = FALSE,
                beta = 1, input.dropout = 0, hidden.dropout = 0,
                optimizer = "adamW", learning.rate = 0.0001, weight.decay = 0.002, momentum = 0, eps = 1e-07,
                encoder.structure = c(128, 64, 32), latent.dim = 16, decoder.structure = c(32, 64, 128),
                act = "elu", init.weight = "xavier.normal",
                scaler = "none")

pmm.params=list(pmm.type = NULL, pmm.k = 5, pmm.link = "prob", pmm.save.vars = NULL)

path="C:/Users/agnes/Desktop/phd-thesis/Thesis/R/temp/mivaemodel.pt"

categorical.encoding = "embeddings"
device = "cpu"
subsample = 1
early.stopping.epochs = 1
loss.na.scale = FALSE
verbose = TRUE
print.every.n = 1
save.model = FALSE
