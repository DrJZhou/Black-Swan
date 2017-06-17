library(ggplot2)
library(lubridate)
library(data.table)

source("ATVHelper.R")
load("basicPreprocess.rda")
########################################################################
###The ATV Data  Data preprocessing
###Create the moving window data,for every 2hours, general the previous 6 time window
########################################################################
volumeData[, has_etc:=as.numeric(has_etc)]
ATVData     <- volumeData[,list(tollgate_id=unique(tollgate_id), direction=unique(direction),
                                  ATV=length(time),etcV=sum(has_etc),VM01=sum(aggVehicleModel==0|aggVehicleModel==1),
                                 VM1=sum(aggVehicleModel==1), VM012=sum(aggVehicleModel<=2),
                                  curDate=unique(curDate),curHour=unique(curHour),curMinute20=unique(curMinute20),
                                  mark=unique(mark)), by=c("tollgateDirection", "timeStamp")]
#################################################################
##### add submit data
#################################################################
ATVSubmit   <- unique(subset(ATVData, select=c("tollgateDirection","tollgate_id","direction")))

##### The expand metrics
tollgateDirectionList   <- unique(ATVData$tollgateDirection)
submitDateList          <- seq(as.Date("2016-10-25"),as.Date("2016-10-31"),1)
curHourList             <- c(8,9,17,18)
curMinuteList           <- c(0, 20, 40)
ATVExpand   <- data.table(expand.grid(tollgateDirectionList, submitDateList, curHourList, curMinuteList))
setnames(ATVExpand, c("tollgateDirection", "curDate", "curHour", "curMinute20"))

##### Join the data
setkey(ATVExpand, "tollgateDirection");  setkey(ATVSubmit, "tollgateDirection")
ATVSubmit   <- ATVSubmit[ATVExpand]

ATVSubmit[ ,timeStamp:=paste(curDate, paste(curHour, curMinute20, "00",sep=":"), sep=" ")]
ATVSubmit[ ,timeStamp:=ymd_hms(timeStamp)]
ATVSubmit[ ,mark:='submit']
ATVSubmit[ ,c("ATV", "etcV", "VM1", "VM01", "VM012"):=NA]
ATVSubmit   <- subset(ATVSubmit, select=colnames(ATVData))
ATVData     <- rbind(ATVData, ATVSubmit)

####Getting the data from the prediction period
ATVData     <- ATVData[curHour %in%dataHourList, ]
####Create the data for machine learning
mainData    <- ATVMLDataFunc(ATVData)
mainData[, extendFlag:='main']

########################################################################
###The ATV Data  Data preprocessing
###Create the moving window data,for every 2hours, general the previous 6 time window
########################################################################
ATVDataLead2   <- extendMLDataFunc(volumeData, -2)
ATVDataLead2[,extendFlag:='lead2']

ATVDataLead3   <- extendMLDataFunc(volumeData, -3)
ATVDataLead3[,extendFlag:='lead3']

ATVDataLead4   <- extendMLDataFunc(volumeData, -4)
ATVDataLead4[,extendFlag:='lead4']

ATVDataLead5    <- extendMLDataFunc(volumeData, -5)
ATVDataLead5[,extendFlag:='lead5']

ATVDataLead6    <- extendMLDataFunc(volumeData, -6)
ATVDataLead6[,extendFlag:='lead6']

ATVDataLead8    <- extendMLDataFunc(volumeData, -8) 
ATVDataLead8[,extendFlag:='lead8']
ATVDataLead10    <- extendMLDataFunc(volumeData, -10) 
ATVDataLead10[,extendFlag:='lead10']

ATVDataLag2    <- extendMLDataFunc(volumeData, 2)
ATVDataLag2[,extendFlag:='lag2']
ATVDataLag3    <- extendMLDataFunc(volumeData, 3)
ATVDataLag3[,extendFlag:='lag3']

ATVDataLag4    <- extendMLDataFunc(volumeData, 4)
ATVDataLag4[,extendFlag:='lag4']
ATVDataLag5    <- extendMLDataFunc(volumeData, 5) 
ATVDataLag5[,extendFlag:='lag5']

ATVDataLag6    <- extendMLDataFunc(volumeData, 6)  
ATVDataLag6[,extendFlag:='lag6']

ATVDataLag8    <- extendMLDataFunc(volumeData, 8) 
ATVDataLag8[,extendFlag:='lag8']
ATVDataLag10    <- extendMLDataFunc(volumeData, 10) 
ATVDataLag10[,extendFlag:='lag10']

extendData  <- rbind(mainData, ATVDataLag2,ATVDataLead2, ATVDataLag3,ATVDataLead3,
                     ATVDataLag4,ATVDataLead4, ATVDataLag5,ATVDataLead5, ATVDataLag6,ATVDataLead6,
                     ATVDataLead8,ATVDataLag8,ATVDataLead10,ATVDataLag10)
save(volumeData, mainData, finalWeatherData,extendData, file="advancedPreprocess.rda")


