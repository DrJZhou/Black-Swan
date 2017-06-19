library(lubridate)
library(data.table)

source("ATTHelper.R")
load("advancedPreprocess.rda")
#################################################################
###The NA Data Imputation 
#################################################################
totalData   <- totalData[ATT60NumLag1>0,]
totalData   <- totalData[!(mark=='test' & extendFlag !='main'), ]

totalData[,intersectionTollgate:=as.factor(intersectionTollgate)]
totalData[, hourWindow:=as.factor(hourWindow)]

###Add the wetherData
finalWeatherData    <- finalWeatherData[,.(curDate,curHour,rel_humidity,precipitation)]
finalWeatherData[,precipitation:=log(precipitation+1)]
finalWeatherData[,precipitation:=ifelse(precipitation>0.1,1,0)]
###### 
setkeyv(finalWeatherData, c("curDate","curHour"))
setkeyv(totalData, c("curDate", "curHour"))
extendData        <- finalWeatherData[totalData]
###Add the holiday
extendData[,isHoliday:=0];  extendData[curDate %in% holidayList, isHoliday:=1]
extendData[curDate==as.Date("2016-09-30") & hourWindow=='[18,17]',isHoliday:=1]
###The weekindex
maxDate         <- extendData[, max(curDate)]
extendData[, weekIndex:=as.numeric(floor((maxDate-curDate)/7)+1)]

#################################################################
###The main +-3, +-5
#################################################################
mainData    <- extendData[extendFlag=='main',]
YName       <- "ATT";
XNames      <- c("intersectionTollgate" ,"hourWindow","MWInterval","ATTLag6","ATTLag5","ATTLag4","ATTLag3", "ATTLag2", "ATTLag1", "isHoliday")
mainData    <- ATTImputationFunc(mainData, XNames=XNames, YName=YName, ntrees=50)

extendLead3 <- extendData[extendFlag=='extendLead3',]
extendLead3 <- extendImputationFunc(extendLead3, XNames=XNames, YName=YName, ntrees=50) 
### The Lag3 extend data imputation
extendLag3  <- extendData[extendFlag=='extendLag3',]
extendLag3  <- extendImputationFunc(extendLag3, XNames=XNames, YName=YName, ntrees=50)

extendLead5 <- extendData[extendFlag=='extendLead5',]
extendLead5 <- extendImputationFunc(extendLead5, XNames=XNames, YName=YName, ntrees=50) 
### The Lag5 extend data imputation
extendLag5  <- extendData[extendFlag=='extendLag5',]
extendLag5  <- extendImputationFunc(extendLag5, XNames=XNames, YName=YName, ntrees=50)
#################################################################
###the Lead2 and Lag2
#################################################################
extendLead2 <- extendData[extendFlag=='extendLead2',]
extendLead2 <- extendImputationFunc(extendLead2, XNames=XNames, YName=YName, ntrees=50)
### The Lag2 extend data imputation
extendLag2  <- extendData[extendFlag=='extendLag2',]
extendLag2  <- extendImputationFunc(extendLag2, XNames=XNames, YName=YName, ntrees=50)

sapply(extendLag2, function(x){sum(is.na(x))})
sapply(extendLead2, function(x){sum(is.na(x))})
#################################################################
###the Lead4 and Lag4
#################################################################
extendLead4 <- extendData[extendFlag=='extendLead4',]
extendLead4 <- extendImputationFunc(extendLead4, XNames=XNames, YName=YName, ntrees=50)
### The Lag4 extend data imputation
extendLag4  <- extendData[extendFlag=='extendLag4',]
extendLag4  <- extendImputationFunc(extendLag4, XNames=XNames, YName=YName, ntrees=50)

sapply(extendLag4, function(x){sum(is.na(x))})
sapply(extendLead4, function(x){sum(is.na(x))})

#################################################################
###the Lead6 and Lag6
#################################################################
extendLead6 <- extendData[extendFlag=='extendLead6',]
extendLead6 <- extendImputationFunc(extendLead6, XNames=XNames, YName=YName, ntrees=50)
### The Lag6 extend data imputation
extendLag6  <- extendData[extendFlag=='extendLag6',]
extendLag6  <- extendImputationFunc(extendLag6, XNames=XNames, YName=YName, ntrees=50)

sapply(extendLag6, function(x){sum(is.na(x))})
sapply(extendLead6, function(x){sum(is.na(x))})
#################################################################
###the Lead8 and Lag8
#################################################################
extendLead8 <- extendData[extendFlag=='extendLead8',]
extendLead8 <- extendImputationFunc(extendLead8, XNames=XNames, YName=YName, ntrees=50)
### The Lag8 extend data imputation
extendLag8  <- extendData[extendFlag=='extendLag8',]
extendLag8  <- extendImputationFunc(extendLag8, XNames=XNames, YName=YName, ntrees=50)

sapply(extendLag8, function(x){sum(is.na(x))})
sapply(extendLead8, function(x){sum(is.na(x))})

#################################################################
###the Lead10 and Lag10
#################################################################
extendLead10 <- extendData[extendFlag=='extendLead10',]
extendLead10 <- extendImputationFunc(extendLead10, XNames=XNames, YName=YName, ntrees=50)
### The Lag10 extend data imputation
extendLag10  <- extendData[extendFlag=='extendLag10',]
extendLag10  <- extendImputationFunc(extendLag10, XNames=XNames, YName=YName, ntrees=50)

sapply(extendLag10, function(x){sum(is.na(x))})
sapply(extendLead10, function(x){sum(is.na(x))})

#################################################################
###the combined data
#################################################################
extendData	<- rbind(extendLead2, extendLag2,extendLead3,extendLag3, extendLead4, extendLag4,extendLead5, extendLag5,
                           extendLead6, extendLag6, extendLead8, extendLag8,extendLead10, extendLag10)
extendData	<- subset(extendData, select=colnames(mainData))
finalData	<- rbind(mainData, extendData)

print(sapply(finalData, function(x){sum(is.na(x))}))

save(finalData, file='baseImputationData.rda')

################################################################
###Revised those too large data
#################################################################
finalData[is.na(ATT40MeanLag1), ATT40MeanLag1:=ATT60MeanLag1]

ATT120SdLagFill    <- apply(subset(finalData, select=paste("ATTLag",1:6,sep="")),1, sd)
finalData$ATT120SdLagFill    <- ATT120SdLagFill
finalData[ATT120NumLag<3, ATT120SdLag:=ATT120SdLagFill]
finalData[ATT60NumLag1<3, ATT60SdLag1:=ATT120SdLag]
finalData[ATT60NumLag2<3, ATT60SdLag2:=ATT120SdLag]

sapply(finalData, function(x){sum(is.na(x))})
save(finalData, file="dataImputation.rda")

