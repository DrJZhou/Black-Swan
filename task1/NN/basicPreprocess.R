library(lubridate)
library(data.table)
#################################################
### load the data
#################################################
##### The training data
dataPath                <- "../data/dataset"
volumeTrainPath         <- file.path(dataPath,  'volume(table 6)_training.csv')
weatherTrainPath        <- file.path(dataPath,  'weather (table 7)_training_update.csv')

#####The new week dataset
volumeUpdatePath        <- file.path(dataPath,  'volume(table 6)_training2.csv')
weatherUpdatePath       <- file.path(dataPath,  'weather (table 7)_test1.csv')

####The submit data
volumeTestPath          <- file.path(dataPath,  'volume(table 6)_test2.csv')
weatherTestPath	        <- file.path(dataPath,  'weather (table 7)_2.csv')

####The  traffic volumne Data
volumeTrainData         <- fread(volumeTrainPath,integer64="numeric", na.strings=c("NULL","NA", "", "\\N"))
volumeUpdateData        <- fread(volumeUpdatePath, integer64="numeric", na.strings=c("NULL","NA", "", "\\N"))
colnames(volumeUpdateData)  <- colnames(volumeTrainData)
volumeTestData          <- fread(volumeTestPath,integer64="numeric", na.strings=c("NULL","NA", "", "\\N"))
colnames(volumeTestData)    <- colnames(volumeTrainData)
volumeTrainData[,mark:='train']
volumeUpdateData[,mark:='update']
volumeTestData[, mark:='test']
volumeData              <- rbind(volumeTrainData,volumeUpdateData, volumeTestData)

###The format transfer
volumeData[ ,time:=ymd_hms(time)]
volumeData[,vehicle_model:=as.numeric(vehicle_model)]

volumeData[, c("curDate", "curHour", "curMinute"):=list(as.Date(time), hour(time), minute(time))]
volumeData[,curMinute20:=floor(curMinute/20)*20]

volumeData[, timeStamp:=paste(curDate, paste(curHour, curMinute20,"00", sep=":"), sep=" ")]
volumeData[, timeStamp:=ymd_hms(timeStamp)]

##The aggVehicleModel
volumeData[, aggVehicleModel:=vehicle_model]
volumeData[aggVehicleModel>=3, aggVehicleModel:=3]
volumeData[, tollgateDirection:=paste(tollgate_id, direction, sep="-")]

#####################################################################
### The weather data
#####################################################################
weatherTrainData        <- fread(weatherTrainPath,integer64="numeric", na.strings=c("NULL","NA", "", "\\N"))
weatherUpdateData       <- fread(weatherUpdatePath,integer64="numeric", na.strings=c("NULL","NA", "", "\\N"))
weatherTestData         <- fread(weatherTestPath,integer64="numeric", na.strings=c("NULL","NA", "", "\\N"))

weatherTrainData[,mark:='train']
weatherUpdateData[,mark:='update']
weatherTestData[, mark:='test']

weatherData             <- rbind(weatherTrainData,weatherUpdateData, weatherTestData)
weatherData[, c("date", "hour", "pressure", "sea_pressure", "wind_direction", "wind_speed", "temperature", "rel_humidity","precipitation"):=list(as.Date(date),as.numeric(hour), as.numeric(pressure), as.numeric(sea_pressure), as.numeric(wind_direction), as.numeric(wind_speed), as.numeric(temperature), as.numeric(rel_humidity), as.numeric(precipitation))]

weatherData[,sea_pressure:=NULL]
weatherData[, dayHour:=ymd_hms(paste(date, paste(hour, "00", "00", sep=":"),sep=" "))]
weatherData <- weatherData[order(dayHour),]

###The outlier of the wind_direction
weatherData[,wdOutlier:=0]; weatherData[wind_direction>360, wdOutlier:=1]
weatherData[,wdLag:=shift(wind_direction, n=1, fill=NA, type="lag")]
weatherData[,wdLead:=shift(wind_direction, n=1, fill=NA, type="lead")]
weatherData[wind_direction>360, wind_direction:=round(wdLag+wdLead)/2]
weatherData[, c("wdOutlier", "wdLag", "wdLead"):=NULL]
#################################################################
### Add the weather data
#################################################################
weatherFill1    <- weatherData[date ==as.Date("2016-09-29") & hour==18,]
weatherFill1[, dayHour:=dayHour+dhours(3)]; weatherFill1[, hour:=hour(dayHour)]

weatherFill2    <- weatherData[date ==as.Date("2016-09-30") & hour==3,]
weatherFill2[, dayHour:=dayHour-dhours(3)]; weatherFill2[, hour:=hour(dayHour)]
weatherData     <- rbind(weatherData, weatherFill1, weatherFill2)

####Add the weather of date 2016.10.10
weatherData1010 <- weatherData[date==as.Date("2016-10-09"),]
weatherData1010[,dayHour:=dayHour+dhours(24)]
weatherData1010[, date:=as.Date(dayHour)]
weatherData     <- rbind(weatherData, weatherData1010)

weatherDataLag  <- copy(weatherData)
weatherDataLead <- copy(weatherData)
###The Lag1
weatherDataLag[, dayHour:=dayHour-dhours(1)];
weatherDataLag[,date:=as.Date(dayHour)];
weatherDataLag[, hour:=hour(dayHour)]
###The Lead1
weatherDataLead[, dayHour:=dayHour+dhours(1)];
weatherDataLead[,date:=as.Date(dayHour)];
weatherDataLead[, hour:=hour(dayHour)]

finalWeatherData    <- rbind(weatherDataLag, weatherDataLead, weatherData)
finalWeatherData    <- finalWeatherData[date>=as.Date("2016-07-01"),]

lastRecord          <- finalWeatherData[date==as.Date("2016-10-31") & hour==22,]
lastRecord[,dayHour:=dayHour+dhours(1)]; lastRecord[,hour:=hour(dayHour)]

finalWeatherData    <- rbind(finalWeatherData, lastRecord)

save(finalWeatherData, volumeData, file="basicPreprocess.rda")

