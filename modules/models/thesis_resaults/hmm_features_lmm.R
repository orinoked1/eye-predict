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

fileName=paste0('face_hmm_features.csv')
alldata<- as.data.frame(read.csv(fileName,header = T,sep = ','))



varnames <- c('praior_s1', 'praior_s2', 'praior_s3',
              'p_s1_to_s1', 'p_s1_to_s2', 'p_s1_to_s3',
              'p_s2_to_s1', 'p_s2_to_s2', 'p_s2_to_s3',
              'p_s3_to_s1', 'p_s3_to_s2', 'p_s3_to_s3',
              'mean_s1_x', 'mean_s1_y',
              'mean_s2_x', 'mean_s2_y',
              'mean_s3_x', 'mean_s3_y',
              'var_s1_x', 'var_s1_y',
              'var_s2_x', 'var_s2_y',
              'var_s3_x', 'var_s3_y') # fixed Xs


regComp= paste0(varnames, collapse= " + ")
# regRand= paste0("(" ,varnames, " | subjectID)", collapse= " + ")
regRand = '(1|subjectID) + (1|stimName)'
fmla <- as.formula(paste("bid ~ ", regRand, " + ", regComp ))
model= lme4::lmer( fmla , data=alldata)
print(model)
#print(anova(model))
print(summary(model))
stargazer(model)
