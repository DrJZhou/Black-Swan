library(ggplot2)
library(gridExtra)
library(data.table)

source("StackHelper.R")
##########################################################
### Load the data
##########################################################
data35  <- ensembleDPFunc(fread("2017-05-28extendBaseline.csv"))
dataB1  <- ensembleDPFunc(fread("2017-05-29OBOB1.csv"))
dataC1  <- ensembleDPFunc(fread("2017-05-29OBOC1.csv"))
dataC3  <- ensembleDPFunc(fread("2017-05-29OBOC3.csv"))
OBOData <- rbind(dataB1,dataC1,dataC3)
setnames(OBOData,'pred', "C3Pred")

setkeyv(data35,c("intersection_id", "tollgate_id","time_window"))
setkeyv(OBOData, c("intersection_id", "tollgate_id","time_window"))
joinData    <- OBOData[data35]

joinData[!is.na(C3Pred),pred:=(0.5*pred+0.5*C3Pred)]

finalResult <- copy(joinData)
finalResult <- subset(finalResult, select=c("intersection_id", "tollgate_id", "time_window", "pred"))
finalResult[,time_window:=paste("\"", time_window, "\"", sep="")]
write.table(finalResult, file=paste(Sys.Date(), "baseline1.csv", sep=""),  row.names=F,col.names=F, quote=F, sep=",")

