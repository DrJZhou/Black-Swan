###################################################
#####The Holiday list
###################################################
holidayList         <- c(seq(as.Date("2016-10-01"),as.Date("2016-10-07"),1), seq(as.Date("2016-09-15"),as.Date("2016-09-17"),1))
predictHourWindow   <-  c("[9,8]", "[18,17]")
dataHourList        <-  c(6,7,8,9,15,16,17,18)


ensembleDPFunc  <- function(data){
    setnames(data, c("intersection_id", "tollgate_id", "time_window", "pred"))
    data        <- data[order(intersection_id,tollgate_id,time_window),]
    return(data)
}

combinePredFunc	<- function(data,newATTData){
	data	<- data[order(intersection_id,tollgate_id,time_window),]
	data[,intersectionTollgate:=paste(intersection_id,tollgate_id,sep='')]
	data[,timeStamp:=tstrsplit(time_window,",")[1]]
	data[,timeStamp:=ymd_hms(substr(timeStamp,2,20))]
	data[, c("curDate", "curHour", "curMinute"):=list(as.Date(timeStamp), hour(timeStamp), minute(timeStamp))]
	data[,curMinute20:=floor(curMinute/20)*20]
	data[, hourWindow:=ifelse(curHour<=12, "[9,8]", "[18,17]")]
	data	<- data[,.(intersectionTollgate,curDate,hourWindow,curHour,curMinute20,timeStamp,pred,tag)]
	setnames(data,'pred',"ATT")
	newATTData$tag	<- unique(data$tag)
	newATTData	<- subset(newATTData, select=colnames(data))
	resultData	<- rbind(newATTData, data)
	return(resultData)
}

