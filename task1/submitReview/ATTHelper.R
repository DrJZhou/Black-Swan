library(mice)
library(randomForest)
###################################################
#####The Holiday list
###################################################
holidayList         <- c(seq(as.Date("2016-10-01"),as.Date("2016-10-07"),1), seq(as.Date("2016-09-15"),as.Date("2016-09-17"),1))
predictHourWindow   <-  c("[9,8]", "[18,17]")
dataHourList        <-  c(6,7,8,9,15,16,17,18)
extendDataHourList        <-  c(5, 6,7,8,9,10,14,15,16,17,18,19)

##############################################################
#### Data preprocessing
##############################################################
ATTMLDataFunc       <- function(data){
    newData     <- copy(data)
    ##Only get the given hour data
    newData         <- newData[curHour %in% dataHourList, ]
    newData[, hourWindow:=ifelse(curHour<=12, "[9,8]", "[18,17]")]
    newData[, keyIndex:=paste(intersectionTollgate, curDate,hourWindow,sep="-")]
    keyIndexList    <- unique(newData$keyIndex)
    ATTMWList       <- lapply(keyIndexList, function(x){
        print(paste("intersectionTollgate-curDate-hourWindow:", x))
        subData         <- newData[keyIndex==x,]
        y               <- ifelse(unique(subData$hourWindow)=='[9,8]',9,18)
        hourIndexList	<- unique(subData$curHour)
        flag1           <- sum(c(y,y-1) %in% hourIndexList)>0   #Make sure theres is supervised value
        flag2           <- sum(c(y-2,y-3) %in% hourIndexList)>0
        if(flag1 & flag2){
            ###The MWInterval
            subMain     <- subData[curHour %in% c(y,y-1),]
            subMain[,MWInterval:=(curHour-y+1)*3+curMinute20/20+1]
            subFeature  <- subData[curHour %in% c(y-2,y-3),]
            ###The supervised data
            subResult   <-  subMain[,list(intersection_id=unique(intersection_id),tollgate_id=unique(tollgate_id),linkCount=unique(linkCount),
                            ATT=mean(travel_time),lastTT=mean(lastTT), prevTT=mean(prevTT),vehicleNum=length(vehicle_id),
                            curDate=unique(curDate),curHour=unique(curHour), curMinute20=unique(curMinute20),
                            mark=unique(mark)), by=c("intersectionTollgate", "timeStamp", "hourWindow")]
            ##The MWInterval
            subResult[,MWInterval:=(curHour-y+1)*3+curMinute20/20+1]
	 ###The statistics feature of the last 2 hours
            subResult$ATT120MeanLag         <- mean(subFeature$travel_time)
            subResult$ATT120SdLag           <- sd(subFeature$travel_time)
            subResult$ATT120NumLag          <- length(subFeature$travel_time)
	    ###The statistics feature of the last hour
            subResult$ATT60MeanLag1         <- mean(subFeature[curHour==y-2,]$travel_time)
            subResult$ATT60SdLag1           <- sd(subFeature[curHour==y-2,]$travel_time)
            subResult$ATT60NumLag1          <- length(subFeature[curHour==y-2,]$travel_time)
            ### The statistics feature of the first hour
            subResult$ATT60MeanLag2         <- mean(subFeature[curHour==y-3,]$travel_time)
            subResult$ATT60SdLag2           <- sd(subFeature[curHour==y-3,]$travel_time)
            subResult$ATT60NumLag2          <- length(subFeature[curHour==y-3,]$travel_time)
            ### The stats of the last 40 minutes
            subResult$ATT40MeanLag1         <- mean(subFeature[curHour==y-2 & curMinute20>0,]$travel_time)
            subResult$ATT40NumLag1          <- length(subFeature[curHour==y-2 & curMinute20>0,]$travel_time)
            
        ###The AVTLag length
            subResult$ATTNumLag6  <- length(subFeature[curHour==y-3 & curMinute20==0, ]$travel_time)
            subResult$ATTNumLag5  <- length(subFeature[curHour==y-3 & curMinute20==20, ]$travel_time)
            subResult$ATTNumLag4  <- length(subFeature[curHour==y-3 & curMinute20==40, ]$travel_time)
            subResult$ATTNumLag3  <- length(subFeature[curHour==y-2 & curMinute20==0, ]$travel_time)
            subResult$ATTNumLag2  <- length(subFeature[curHour==y-2 & curMinute20==20, ]$travel_time)
            subResult$ATTNumLag1  <- length(subFeature[curHour==y-2 & curMinute20==40, ]$travel_time)
            ####The AVTLag mean
            subResult$ATTLag6  <- mean(subFeature[curHour==y-3 & curMinute20==0, ]$travel_time)
            subResult$ATTLag5  <- mean(subFeature[curHour==y-3 & curMinute20==20, ]$travel_time)
            subResult$ATTLag4  <- mean(subFeature[curHour==y-3 & curMinute20==40, ]$travel_time)
            subResult$ATTLag3  <- mean(subFeature[curHour==y-2 & curMinute20==0, ]$travel_time)
            subResult$ATTLag2  <- mean(subFeature[curHour==y-2 & curMinute20==20, ]$travel_time)
            subResult$ATTLag1  <- mean(subFeature[curHour==y-2 & curMinute20==40, ]$travel_time)    
            subResult}
		})
	ATTMWData  <- rbindlist(ATTMWList)
    	return(ATTMWData)
}

extendMLDataFunc    <- function(data,minuteGap=0){
    newData         <- copy(data)
    newData[, starting_time:=starting_time+dminutes(minuteGap)]
    newData[, c("curDate", "curHour", "curMinute"):=list(as.Date(starting_time), hour(starting_time), minute(starting_time))]
    ##The timestamp feature
    newData[,curMinute20:=floor(curMinute/20)*20]
    newData[, timeStamp:=paste(curDate, paste(curHour, curMinute20,"00", sep=":"), sep=" ")]
    newData[, timeStamp:=ymd_hms(timeStamp)]
    ##Only get the given hour data
    newData         <- newData[curHour %in% dataHourList, ]
    newData[, hourWindow:=ifelse(curHour<=12, "[9,8]", "[18,17]")]
    newData[, keyIndex:=paste(intersectionTollgate, curDate,hourWindow,sep="-")]
    keyIndexList    <- unique(newData$keyIndex)
    ATTMWList       <- lapply(keyIndexList, function(x){
        print(paste("intersectionTollgate-curDate-hourWindow:", x))
        subData     <- newData[keyIndex==x,]
        y           <- ifelse(unique(subData$hourWindow)=='[9,8]',9,18)
        hourIndexList	<- unique(subData$curHour)
        flag1           <- sum(c(y,y-1) %in% hourIndexList)>0   #Make sure theres is supervised value
        flag2           <- sum(c(y-2,y-3) %in% hourIndexList)>0
        if((flag1 & flag2)){
            ###The MWInterval
            subMain     <- subData[curHour %in% c(y,y-1),]
            subMain[,MWInterval:=(curHour-y+1)*3+curMinute20/20+1]
            subFeature  <- subData[curHour %in% c(y-2,y-3),]

            ###The supervised data
            subResult   <-  subMain[,list(intersection_id=unique(intersection_id),tollgate_id=unique(tollgate_id),linkCount=unique(linkCount),
                            ATT=mean(travel_time),lastTT=mean(lastTT), prevTT=mean(prevTT),vehicleNum=length(vehicle_id),
                            curDate=unique(curDate),curHour=unique(curHour), curMinute20=unique(curMinute20),
                            mark=unique(mark)), by=c("intersectionTollgate", "timeStamp","hourWindow")]
	    
            ##The MWInterval and hourwindow
            subResult[,MWInterval:=(curHour-y+1)*3+curMinute20/20+1]
            ###The statistics feature of the last 2 hours
            subResult$ATT120MeanLag         <- mean(subFeature$travel_time)
            subResult$ATT120SdLag           <- sd(subFeature$travel_time)
            subResult$ATT120NumLag          <- length(subFeature$travel_time)
            ###The statistics feature of the last hour
            subResult$ATT60MeanLag1         <- mean(subFeature[curHour==y-2,]$travel_time)
            subResult$ATT60SdLag1           <- sd(subFeature[curHour==y-2,]$travel_time)
            subResult$ATT60NumLag1          <- length(subFeature[curHour==y-2,]$travel_time)
	    ### The statistics feature of the first hour
            subResult$ATT60MeanLag2         <- mean(subFeature[curHour==y-3,]$travel_time)
            subResult$ATT60SdLag2           <- sd(subFeature[curHour==y-3,]$travel_time)
            subResult$ATT60NumLag2          <- length(subFeature[curHour==y-3,]$travel_time)
            ### The stats of the last 40 minutes
            subResult$ATT40MeanLag1         <- mean(subFeature[curHour==y-2 & curMinute20>0,]$travel_time)
            subResult$ATT40NumLag1          <- length(subFeature[curHour==y-2 & curMinute20>0,]$travel_time)
            
	###The AVTLag length
            subResult$ATTNumLag6  <- length(subFeature[curHour==y-3 & curMinute20==0, ]$travel_time)
            subResult$ATTNumLag5  <- length(subFeature[curHour==y-3 & curMinute20==20, ]$travel_time)
            subResult$ATTNumLag4  <- length(subFeature[curHour==y-3 & curMinute20==40, ]$travel_time)
            subResult$ATTNumLag3  <- length(subFeature[curHour==y-2 & curMinute20==0, ]$travel_time)
            subResult$ATTNumLag2  <- length(subFeature[curHour==y-2 & curMinute20==20, ]$travel_time)
            subResult$ATTNumLag1  <- length(subFeature[curHour==y-2 & curMinute20==40, ]$travel_time)
            ####The AVTLag mean
            subResult$ATTLag6  <- mean(subFeature[curHour==y-3 & curMinute20==0, ]$travel_time)
            subResult$ATTLag5  <- mean(subFeature[curHour==y-3 & curMinute20==20, ]$travel_time)
            subResult$ATTLag4  <- mean(subFeature[curHour==y-3 & curMinute20==40, ]$travel_time)
            subResult$ATTLag3  <- mean(subFeature[curHour==y-2 & curMinute20==0, ]$travel_time)
            subResult$ATTLag2  <- mean(subFeature[curHour==y-2 & curMinute20==20, ]$travel_time)
            subResult$ATTLag1  <- mean(subFeature[curHour==y-2 & curMinute20==40, ]$travel_time)                
            subResult}
		})
	ATTMWData  <- rbindlist(ATTMWList)
    	return(ATTMWData)
}

ATTImputationFunc  <- function(data,XNames, YName, ntrees=100){
	# data	<- copy(mainData);  YName   <- "ATT"
        #  XNames      <- c("intersectionTollgate" ,"hourWindow","MWInterval", "ATTLag6","ATTLag5","ATTLag4","ATTLag3", "ATTLag2", "ATTLag1")
        dataNames       <- colnames(data)
        formulaStr      <- paste(YName, "~", paste(XNames, collapse='+' ), sep="")
        preData         <- data[!is.na(get(YName)),]
        remainderData   <- data[is.na(get(YName)),]
        ###The new rfImp data for train data filling
        rfImp           <- data.table(rfImpute(as.formula(formulaStr), preData,  ntree=ntrees))
        preData         <- subset(preData, select=dataNames[!dataNames %in% colnames(rfImp)])
        preData         <- cbind(preData, rfImp)
        preData         <- subset(preData, select=dataNames)
        data            <- rbind(preData,remainderData)
        
        ### The total data by mice
        miceData        <- subset(data, select=c(XNames, YName))
        imputeData      <- data.table(complete(mice(miceData,m=10,method='rf')))
        imputeData      <- subset(imputeData, select=XNames)
        data            <- subset(data, select=dataNames[!dataNames %in% XNames])
        data            <- cbind(data,imputeData)
        return(data)
}

extendImputationFunc  <- function(data,XNames, YName, ntrees=100){
        dataNames       <- colnames(data)
        formulaStr      <- paste(YName, "~", paste(XNames, collapse='+' ), sep="")
        preData         <- data[!is.na(get(YName)),]
        remainderData   <- data[is.na(get(YName)),]
        ###The new rfImp data for train data filling
        rfImp           <- data.table(rfImpute(as.formula(formulaStr), preData,  ntree=ntrees))
        preData         <- subset(preData, select=dataNames[!dataNames %in% colnames(rfImp)])
        preData         <- cbind(preData, rfImp)
        preData         <- subset(preData, select=dataNames)
        data            <- rbind(preData,remainderData)
        return(data)
}


