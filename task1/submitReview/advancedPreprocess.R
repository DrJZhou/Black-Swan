library(lubridate)
library(data.table)

source("ATTHelper.R")
load("basicPreprocess.rda")
##############################################################
####The ATTData
##############################################################
trajectoriesData[, c("curDate", "curHour", "curMinute"):=list(as.Date(starting_time), hour(starting_time), minute(starting_time))]
trajectoriesData[,curMinute20:=floor(curMinute/20)*20]
trajectoriesData[, timeStamp:=paste(curDate, paste(curHour, curMinute20,"00", sep=":"), sep=" ")]
trajectoriesData[, timeStamp:=ymd_hms(timeStamp)]

#########################################################
####Analysis trajectories Data and remove outlier
#########################################################
trajectoriesData[,outlierThreshold:=450]
trajectoriesData[curDate==as.Date("2016-10-15") & intersectionTollgate %in% c('C1','A3'), outlierThreshold:=600]
trajectoriesData[curDate==as.Date("2016-10-21") & intersectionTollgate %in% c('C1'), outlierThreshold:=550]
trajectoriesData[curDate==as.Date("2016-10-28") & intersectionTollgate %in% c('A3'), outlierThreshold:=350]
trajectoriesData[curDate==as.Date("2016-10-28") & intersectionTollgate %in% c('C1'), outlierThreshold:=330]
trajectoriesData    <- trajectoriesData[travel_time<outlierThreshold,]
trajectoriesData[,outlierThreshold:=NULL]

ATTData     <- trajectoriesData[curHour %in% dataHourList, ]
#################################################################
##### add submit data
#################################################################
ATTSubmit   <- unique(subset(ATTData, select=c("intersectionTollgate", "intersection_id", "tollgate_id", "linkCount")))
##### The expand metrics
intersectionTollgateList    <- unique(ATTData$intersectionTollgate)

###########################################################################
### !!!!!!!!!!!!!Warning!!! in the second period, you must change the day list
###########################################################################
submitDateList              <- seq(as.Date("2016-10-25"),as.Date("2016-10-31"),1)
curHourList                 <- c(8,9,17,18)
curMinuteList               <- c(0, 20, 40)
ATTExpand   <- data.table(expand.grid(intersectionTollgateList, submitDateList, curHourList, curMinuteList))
setnames(ATTExpand, c("intersectionTollgate", "curDate", "curHour", "curMinute20"))

##### Join the data
setkey(ATTExpand, "intersectionTollgate");  setkey(ATTSubmit, "intersectionTollgate")
ATTSubmit   <- ATTSubmit[ATTExpand]

ATTSubmit[ ,timeStamp:=paste(curDate, paste(curHour, curMinute20, "00",sep=":"), sep=" ")]
ATTSubmit[ ,timeStamp:=ymd_hms(timeStamp)]
ATTSubmit[,starting_time:=timeStamp]
ATTSubmit[ ,mark:='test']
ATTSubmit[ ,c("ATT", "lastTT", "prevTT", "vehicleNum","vehicle_id","travel_time","curMinute"):=NA]
ATTSubmit   <- subset(ATTSubmit, select=colnames(ATTData))
ATTData     <- rbind(ATTData, ATTSubmit)

#######################################################
### Create Data for Machine learning
#####################################################
mainData    <- ATTMLDataFunc(ATTData);  
mainData[,extendFlag:='main']

###+- 2
extendLag2  <- extendMLDataFunc(trajectoriesData,2)
extendLag2[,extendFlag:='extendLag2']
extendLead2 <- extendMLDataFunc(trajectoriesData,-2)
extendLead2[,extendFlag:='extendLead2']

###+-3
extendLag3  <- extendMLDataFunc(trajectoriesData,3)
extendLag3[,extendFlag:='extendLag3']
extendLead3 <- extendMLDataFunc(trajectoriesData,-3)
extendLead3[,extendFlag:='extendLead3']

###+-4
extendLag4  <- extendMLDataFunc(trajectoriesData,4)
extendLag4[,extendFlag:='extendLag4']
extendLead4 <- extendMLDataFunc(trajectoriesData,-4)
extendLead4[,extendFlag:='extendLead4']

###+-5 minutes
extendLag5  <- extendMLDataFunc(trajectoriesData,5)
extendLag5[,extendFlag:='extendLag5']
extendLead5 <- extendMLDataFunc(trajectoriesData,-5)
extendLead5[,extendFlag:='extendLead5']

###+-6 minutes
extendLag6  <- extendMLDataFunc(trajectoriesData,6)
extendLag6[,extendFlag:='extendLag6']
extendLead6 <- extendMLDataFunc(trajectoriesData,-6)
extendLead6[,extendFlag:='extendLead6']

###+-8
extendLag8  <- extendMLDataFunc(trajectoriesData,8)
extendLag8[,extendFlag:='extendLag8']
extendLead8 <- extendMLDataFunc(trajectoriesData,-8)
extendLead8[,extendFlag:='extendLead8']
###+-10
extendLag10  <- extendMLDataFunc(trajectoriesData,10)
extendLag10[,extendFlag:='extendLag10']
extendLead10 <- extendMLDataFunc(trajectoriesData,-10)
extendLead10[,extendFlag:='extendLead10']

totalData   <- rbind(mainData,extendLag2,extendLead2,extendLag3,extendLead3,extendLag4,extendLead4,extendLag5,extendLead5,extendLag6,extendLead6,extendLag8,extendLead8,extendLag10, extendLead10)
save(mainData,totalData,finalWeatherData,file="advancedPreprocess.rda")

