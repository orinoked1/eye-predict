rm(list=ls())
graphics.off()
pardefault <- par()
options(warn=1)


#load libraries
library(pacman)
pacman::p_load(rstudioapi,lme4,ggplot2,reshape2,dplyr,data.table,stargazer)

# define path for the dir of current file
rstudioapi::getActiveDocumentContext
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

fileName_1=paste0('snack_visual_features.csv')
alldata_1<- as.data.frame(read.csv(fileName_1,header = T,sep = ','))

fileName_2=paste0('snack_ss_features.csv')
alldata_2<- as.data.frame(read.csv(fileName_2,header = T,sep = ','))

fileName_3=paste0('snack_hmm_features_not_scaled.csv')
alldata_3<- as.data.frame(read.csv(fileName_3,header = T,sep = ','))

alldata<-merge(x=alldata_1,y=alldata_2,by="sampleId",all.x=TRUE)
alldata<-merge(x=alldata,y=alldata_3,by="sampleId",all.x=TRUE)

varnames <- c('avg_bid', 
              'avg_sacc_duration',
              #'fix_count', 
              #'p_s1_to_s1', 'p_s1_to_s2', 'p_s1_to_s3', 
              'mean_s1_x', 'mean_s2_x', 'mean_s3_x') # fixed Xs

# scale data by subject
for (c in c('bid',varnames)){
  print(c)
  for (sub in unique(alldata$subjectID)){
    alldata[alldata$subjectID%in%sub,paste(c,'_scaled')]<-scale(alldata[alldata$subjectID%in%sub,c])
    
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
