###################################################
#####The data List
###################################################
predictHourWindow   <-  c("[9,8]", "[18,17]")
dataHourList        <-  c(6,7,8,9,15,16,17,18)
nationDayList       <-  seq(as.Date("2016-10-01"),as.Date("2016-10-07"),1)

##############################################################
#### Data preprocessing
##############################################################
ATVMLDataFunc   <- function(data){
    data[, hourWindow:=ifelse(curHour<=12, "[9,8]", "[18,17]")]
    data[, keyIndex:=paste(tollgateDirection, curDate,hourWindow,sep="-")]
    keyIndexList     <- unique(data$keyIndex)
    ATVMWList  <- lapply(keyIndexList, function(x){
        print(paste("tollgateDirection-curDate-hourWindow:", x))
        subData     <- data[keyIndex==x,  ]
        
        y           <- ifelse(unique(subData$hourWindow)=='[9,8]',9,18)
        ###The MWInterval
        subMain     <- subData[curHour %in% c(y,y-1)]
        subMain[,MWInterval:=(curHour-y+1)*3+curMinute20/20+1]
        subFeature  <- subData[curHour %in% c(y-2,y-3),]
            ####The AVTLag
        subMain$ATVLag6  <- subFeature[curHour==y-3 & curMinute20==0, ]$ATV
        subMain$ATVLag5  <- subFeature[curHour==y-3 & curMinute20==20, ]$ATV
        subMain$ATVLag4  <- subFeature[curHour==y-3 & curMinute20==40, ]$ATV
        subMain$ATVLag3  <- subFeature[curHour==y-2 & curMinute20==0, ]$ATV
        subMain$ATVLag2  <- subFeature[curHour==y-2 & curMinute20==20, ]$ATV
        subMain$ATVLag1  <- subFeature[curHour==y-2 & curMinute20==40, ]$ATV

	####The etc-Lag
        subMain$etcVLag6  <- subFeature[curHour==y-3 & curMinute20==0, ]$etcV
        subMain$etcVLag5  <- subFeature[curHour==y-3 & curMinute20==20, ]$etcV
        subMain$etcVLag4  <- subFeature[curHour==y-3 & curMinute20==40, ]$etcV
        subMain$etcVLag3  <- subFeature[curHour==y-2 & curMinute20==0, ]$etcV
        subMain$etcVLag2  <- subFeature[curHour==y-2 & curMinute20==20, ]$etcV
        subMain$etcVLag1  <- subFeature[curHour==y-2 & curMinute20==40, ]$etcV

        ####The VM01
        subMain$VM01Lag6  <- subFeature[curHour==y-3 & curMinute20==0, ]$VM01
        subMain$VM01Lag5  <- subFeature[curHour==y-3 & curMinute20==20, ]$VM01
        subMain$VM01Lag4  <- subFeature[curHour==y-3 & curMinute20==40, ]$VM01
        subMain$VM01Lag3  <- subFeature[curHour==y-2 & curMinute20==0, ]$VM01
        subMain$VM01Lag2  <- subFeature[curHour==y-2 & curMinute20==20, ]$VM01
        subMain$VM01Lag1  <- subFeature[curHour==y-2 & curMinute20==40, ]$VM01

	####The VM012
        subMain$VM012Lag6  <- subFeature[curHour==y-3 & curMinute20==0, ]$VM012
        subMain$VM012Lag5  <- subFeature[curHour==y-3 & curMinute20==20, ]$VM012
        subMain$VM012Lag4  <- subFeature[curHour==y-3 & curMinute20==40, ]$VM012
        subMain$VM012Lag3  <- subFeature[curHour==y-2 & curMinute20==0, ]$VM012
        subMain$VM012Lag2  <- subFeature[curHour==y-2 & curMinute20==20, ]$VM012
        subMain$VM012Lag1  <- subFeature[curHour==y-2 & curMinute20==40, ]$VM012
        subMain[,keyIndex:=NULL]
    })
    ATVMWData  <- rbindlist(ATVMWList)
    return(ATVMWData)
}


##############################################################
#### Data preprocessing
##############################################################
extendMLDataFunc    <- function(data, minuteGap=5){
        newData     <- copy(data)
        newData[, time:=time+dminutes(minuteGap)]
        newData[, c("curDate", "curHour", "curMinute"):=list(as.Date(time), hour(time), minute(time))]
        newData[,curMinute20:=floor(curMinute/20)*20]
        newData[, timeStamp:=paste(curDate, paste(curHour, curMinute20,"00", sep=":"), sep=" ")]
        newData[, timeStamp:=ymd_hms(timeStamp)]
        ##Subset 
        newData     <- newData[curHour %in% dataHourList,]
        newATVData  <- newData[,list(tollgate_id=unique(tollgate_id), direction=unique(direction),
                                     ATV=length(time),etcV=sum(has_etc),VM01=sum(aggVehicleModel==0|aggVehicleModel==1),
                                 VM1=sum(aggVehicleModel==1), VM012=sum(aggVehicleModel<=2),
                                curDate=unique(curDate),curHour=unique(curHour),curMinute20=unique(curMinute20),
                                 mark=unique(mark)),
                                    by=c("tollgateDirection", "timeStamp")]

        newATVData[, hourWindow:=ifelse(curHour<=12, "[9,8]", "[18,17]")]
        newATVData[, keyIndex:=paste(tollgateDirection, curDate,hourWindow,sep="-")]
        keyIndexList    <- unique(newATVData$keyIndex)
        ATVMWList       <- lapply(keyIndexList, function(x){
                print(paste("tollgateDirection-curDate-hourWindow:", x))
                subData     <- newATVData[keyIndex==x,  ]
                y           <- ifelse(unique(subData$hourWindow)=='[9,8]',9,18)
                ###The MWInterval
                subMain     <- subData[curHour %in% c(y,y-1)]
                subMain[,MWInterval:=(curHour-y+1)*3+curMinute20/20+1]
                subFeature  <- subData[curHour %in% c(y-2,y-3),]
                ####The AVTLag
                subMain$ATVLag6  <- subFeature[curHour==y-3 & curMinute20==0, ]$ATV
                subMain$ATVLag5  <- subFeature[curHour==y-3 & curMinute20==20, ]$ATV
                subMain$ATVLag4  <- subFeature[curHour==y-3 & curMinute20==40, ]$ATV
                subMain$ATVLag3  <- subFeature[curHour==y-2 & curMinute20==0, ]$ATV
                subMain$ATVLag2  <- subFeature[curHour==y-2 & curMinute20==20, ]$ATV
                subMain$ATVLag1  <- subFeature[curHour==y-2 & curMinute20==40, ]$ATV

        ####The etc-Lag
        subMain$etcVLag6  <- subFeature[curHour==y-3 & curMinute20==0, ]$etcV
        subMain$etcVLag5  <- subFeature[curHour==y-3 & curMinute20==20, ]$etcV
        subMain$etcVLag4  <- subFeature[curHour==y-3 & curMinute20==40, ]$etcV
        subMain$etcVLag3  <- subFeature[curHour==y-2 & curMinute20==0, ]$etcV
        subMain$etcVLag2  <- subFeature[curHour==y-2 & curMinute20==20, ]$etcV
        subMain$etcVLag1  <- subFeature[curHour==y-2 & curMinute20==40, ]$etcV

        ####The VM01
        subMain$VM01Lag6  <- subFeature[curHour==y-3 & curMinute20==0, ]$VM01
        subMain$VM01Lag5  <- subFeature[curHour==y-3 & curMinute20==20, ]$VM01
        subMain$VM01Lag4  <- subFeature[curHour==y-3 & curMinute20==40, ]$VM01
        subMain$VM01Lag3  <- subFeature[curHour==y-2 & curMinute20==0, ]$VM01
        subMain$VM01Lag2  <- subFeature[curHour==y-2 & curMinute20==20, ]$VM01
        subMain$VM01Lag1  <- subFeature[curHour==y-2 & curMinute20==40, ]$VM01

        ####The VM012
        subMain$VM012Lag6  <- subFeature[curHour==y-3 & curMinute20==0, ]$VM012
        subMain$VM012Lag5  <- subFeature[curHour==y-3 & curMinute20==20, ]$VM012
        subMain$VM012Lag4  <- subFeature[curHour==y-3 & curMinute20==40, ]$VM012
        subMain$VM012Lag3  <- subFeature[curHour==y-2 & curMinute20==0, ]$VM012
        subMain$VM012Lag2  <- subFeature[curHour==y-2 & curMinute20==20, ]$VM012
        subMain$VM012Lag1  <- subFeature[curHour==y-2 & curMinute20==40, ]$VM012
        subMain[,keyIndex:=NULL]
    })
    ATVMWData  <- rbindlist(ATVMWList)
    return(ATVMWData)

}

########################################################
### The time series decomposition
########################################################
extendTSFunc   <- function(data, minuteGap=5){
        newData     <- copy(data)
        newData[, time:=time+dminutes(minuteGap)]
        newData[, c("curDate", "curHour", "curMinute"):=list(as.Date(time), hour(time), minute(time))]
        newData[,curMinute20:=floor(curMinute/20)*20]
        newData[, timeStamp:=paste(curDate, paste(curHour, curMinute20,"00", sep=":"), sep=" ")]
        newData[, timeStamp:=ymd_hms(timeStamp)]
        ##Subset 
        newData     <- newData[curHour %in% dataHourList,]
        newATVData  <- newData[,list(tollgate_id=unique(tollgate_id), direction=unique(direction),
                                     ATV=length(time),etcV=sum(has_etc),VM01=sum(aggVehicleModel==0|aggVehicleModel==1),
                                 VM1=sum(aggVehicleModel==1), VM012=sum(aggVehicleModel<=2),
                                curDate=unique(curDate),curHour=unique(curHour),curMinute20=unique(curMinute20),
                                 mark=unique(mark)),
                                    by=c("tollgateDirection", "timeStamp")]
        newATVData[, hourWindow:=ifelse(curHour<=12, "[9,8]", "[18,17]")]
        newATVData      <- newATVData[order(tollgateDirection,timeStamp),]
        return(newATVData)
}




