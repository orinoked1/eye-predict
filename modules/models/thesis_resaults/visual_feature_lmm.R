rm(list=ls())
graphics.off()
pardefault <- par()
options(warn=1)


#load libraries
library(pacman)
pacman::p_load(rstudioapi,lme4,ggplot2,reshape2,dplyr,data.table,stargazer)
library(lme4)
library(MuMIn)


# define path for the dir of current file
rstudioapi::getActiveDocumentContext
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

fileName=paste0('snack_visual_features.csv')
alldata<- as.data.frame(read.csv(fileName,header = T,sep = ','))



varnames <- c('avg_bid') # fixed Xs

# scale data by subject
for (c in c('bid',varnames)){
  print(c)
  for (sub in unique(alldata$subjectID)){
    alldata[alldata$subjectID%in%sub,paste(c)]<-scale(alldata[alldata$subjectID%in%sub,c])
    
  }
}
regComp= paste0(varnames, collapse= " + ")
# regRand= paste0("(" ,varnames, " | subjectID)", collapse= " + ")
regRand = '(1|subjectID) + (1|stimName)'
fmla <- as.formula(paste("bid ~ ", regRand, " + ", regComp ))
model= lme4::lmer( fmla , data=alldata)
print(model)
#print(anova(model))
print(summary(model))
stargazer(model)
#Determine R2:
print(r.squaredGLMM(model))
