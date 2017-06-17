####################################################
### Anomaly detection
####################################################
library(caret)
library(reshape2)
library(lubridate)
library(data.table)

source("ATVHelper.R")
load("advancedPreprocess.rda")
#########################################################
### Preprocess based on the EDA
###(1)Remove the outlier
###(2) Do data point imputation
#########################################################
##The remove meta key
extendData[, tollgateDirectionDate:=paste(tollgateDirection, curDate, sep="-")]
extendData[, tollgateDirectionDateHW:=paste(tollgateDirection, curDate,hourWindow, sep="-")]
extendData[, tollgateDirectionDateHWMWInterval:=paste(tollgateDirection, curDate,hourWindow,MWInterval, sep="-")]
##Remove the tollgateDirectionDate outlier
##
outlierList1    <- c(paste("2-0", c(as.Date("2016-09-28"), seq(as.Date("2016-10-01"),as.Date("2016-10-07"),1)),sep="-"))
outlierList2    <- c("1-0-2016-09-30-[18,17]","1-0-2016-10-01-[9,8]", 
                     "1-1-2016-9-30-[18,17]","1-1-2016-10-01-[9,8]","1-1-2016-10-02-[9,8]",
                     "2-0-2016-09-21-[9,8]","2-0-2016-09-27-[18,17]","2-0-2016-09-30-[18,17]",
                     "3-1-2016-10-02-[18,17]")

outlierList3    <- c(paste("2-0-2016-09-24-[18,17]",4:6,sep="-"),
                     paste("3-0-2016-09-29-[18,17]",4:6,sep="-"),
                     paste("3-1-2016-09-30-[18,17]",4:6,sep="-")
                     )
extendData      <- extendData[!tollgateDirectionDate %in% outlierList1,]
extendData      <- extendData[!tollgateDirectionDateHW %in% outlierList2,]
extendData      <- extendData[!tollgateDirectionDateHWMWInterval %in% outlierList3,]

###Do imputation 
imputationList  <- c("2-0-2016-10-15-[9,8]")
extendData      <- extendData[!(tollgateDirectionDateHW %in% imputationList & extendFlag != 'main'),]

extendData[tollgateDirectionDateHW %in% imputationList, ATVLag4:=0.2*ATVLag1+0.8*ATVLag5]
extendData[tollgateDirectionDateHW %in% imputationList, ATVLag3:=0.4*ATVLag1+0.6*ATVLag5]
extendData[tollgateDirectionDateHW %in% imputationList, ATVLag2:=0.6*ATVLag1+0.4*ATVLag5]

extendData[tollgateDirectionDateHW %in% imputationList, VM012Lag4:=0.2*VM012Lag1+0.8*VM012Lag5]
extendData[tollgateDirectionDateHW %in% imputationList, VM012Lag3:=0.4*VM012Lag1+0.6*VM012Lag5]
extendData[tollgateDirectionDateHW %in% imputationList, VM012Lag2:=0.6*VM012Lag1+0.4*VM012Lag5]

extendData[tollgateDirectionDateHW %in% imputationList, VM01Lag4:=0.2*VM01Lag1+0.8*VM01Lag5]
extendData[tollgateDirectionDateHW %in% imputationList, VM01Lag3:=0.4*VM01Lag1+0.6*VM01Lag5]
extendData[tollgateDirectionDateHW %in% imputationList, VM01Lag2:=0.6*VM01Lag1+0.4*VM01Lag5]

extendData[tollgateDirectionDateHW %in% imputationList, etcVLag4:=0.2*etcVLag1+0.8*etcVLag5]
extendData[tollgateDirectionDateHW %in% imputationList, etcVLag3:=0.4*etcVLag1+0.6*etcVLag5]
extendData[tollgateDirectionDateHW %in% imputationList, etcVLag2:=0.6*etcVLag1+0.4*etcVLag5]

extendData[, c("tollgateDirectionDate", "tollgateDirectionDateHW","tollgateDirectionDateHWMWInterval"):=NULL]

##add holiday
extendData[, isHoliday:=0]
extendData[curDate %in% nationDayList, isHoliday:=1]
extendData[curDate==as.Date("2016-09-30") & hourWindow=='[18,17]',isHoliday:=1]
#########################################################
### concentrate to 40 minutes a data point
#########################################################
extendData[, twoATVLag1:=ATVLag1+ATVLag2]; extendData[, twoVM01Lag1:=VM01Lag1+VM01Lag2]; extendData[, twoVM012Lag1:=VM012Lag1+VM012Lag2];extendData[, twoEtcVLag1:=etcVLag1+etcVLag2];
extendData[, twoATVLag2:=ATVLag3+ATVLag4]; extendData[, twoVM01Lag2:=VM01Lag3+VM01Lag4]; extendData[, twoVM012Lag2:=VM012Lag3+VM012Lag4];extendData[, twoEtcVLag2:=etcVLag3+etcVLag4];
extendData[, twoATVLag3:=ATVLag5+ATVLag6]; extendData[, twoVM01Lag3:=VM01Lag5+VM01Lag6]; extendData[, twoVM012Lag3:=VM012Lag5+VM012Lag6];extendData[, twoEtcVLag3:=etcVLag5+etcVLag6];

finalWeatherData    <- finalWeatherData[,.(date, hour,rel_humidity,precipitation)]
finalWeatherData[,precipitation:=log(precipitation+1)]
finalWeatherData[,precipitation:=ifelse(precipitation>0.1,1,0)]
setnames(finalWeatherData, c("curDate", "curHour", "rel_humidity", "precipitation"))

###The extend Data
setkeyv(finalWeatherData, c("curDate", "curHour"))
setkeyv(extendData, c("curDate","curHour"))
extendData      <- finalWeatherData[extendData]
##add weekIndex
maxDate         <- extendData[,max(curDate)]
extendData[, weekIndex:=as.numeric(floor((maxDate-curDate)/7)+1)]
extendData      <- extendData[mark !='test',]
mainData        <- extendData[extendFlag %in% c('main'),]


save(mainData, extendData,file="cleanFeatureEngine.rda")

