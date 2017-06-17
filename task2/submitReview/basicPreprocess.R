library(lubridate)
library(data.table)
#################################################
### load the data
#################################################
##### The training data
dataPath                <- "./newDataSets"
linkPath                <- file.path(dataPath, 'training', 'links (table 3).csv')
routePath               <- file.path(dataPath, 'training', 'routes (table 4).csv')
trajectoriesTrainPath   <- file.path(dataPath, 'training', 'trajectories(table 5)_training.csv')
weatherTrainPath        <- file.path(dataPath, 'training', 'weather (table 7)_training_update.csv')

#####The new week dataset
trajectoriesUpdatePath  <- file.path(dataPath, 'dataSetPhase2', 'trajectories_training2.csv')
weatherUpdatePath       <- file.path(dataPath, "testing_phase1", 'weather (table 7)_test1.csv')
####The submit data
trajectoriesTestPath    <- file.path(dataPath, 'dataSetPhase2', 'trajectories_test2.csv')
weatherTestPath         <- file.path(dataPath, "dataSetPhase2", 'weather2.csv')

####The link and route data
linkData        <- fread(linkPath,integer64="numeric", na.strings=c("NULL","NA", "", "\\N"))
linkData[, c("length", "width", "lanes", "lane_width"):=list(as.numeric(length),as.numeric(width), as.integer(lanes), as.numeric(lane_width))]
routeData       <- fread(routePath,integer64="numeric", na.strings=c("NULL","NA", "", "\\N"))

#####trajectories data
trajectoriesTrainData   <- fread(trajectoriesTrainPath,integer64="numeric", na.strings=c("NULL","NA", "", "\\N"))
trajectoriesUpdateData  <- fread(trajectoriesUpdatePath,integer64="numeric", na.strings=c("NULL","NA", "", "\\N"))
trajectoriesTestData    <- fread(trajectoriesTestPath,integer64="numeric", na.strings=c("NULL","NA", "", "\\N"))
trajectoriesTrainData[,mark:='train']
trajectoriesUpdateData[,mark:='update']
trajectoriesTestData[, mark:='test'] 
trajectoriesData        <- rbind(trajectoriesTrainData,trajectoriesUpdateData ,trajectoriesTestData)

## construct the submit data
trajectoriesData[ ,starting_time:=ymd_hms(starting_time)]
trajectoriesData[ ,travel_time:=as.numeric(travel_time)]
trajectoriesData[, intersectionTollgate:=paste(intersection_id, tollgate_id, sep="")]
trajectoriesData    <- trajectoriesData[order(intersectionTollgate, starting_time), ]

########################################################################
### ATT Data preprocessing
########################################################################
#The route data
routeData$linkCount     <- sapply(strsplit(routeData$link_seq,split=','), function(x){length(x)})

setkeyv(trajectoriesData, c("intersection_id", "tollgate_id"))
setkeyv(routeData, c("intersection_id", "tollgate_id"))
trajectoriesData        <- routeData[trajectoriesData]

lastTT  <- sapply(strsplit(trajectoriesData$travel_seq, split=";|#"),tail, 1)
trajectoriesData[,travel_time:=as.numeric(travel_time)]
trajectoriesData[,lastTT:=as.numeric(lastTT)]
trajectoriesData[,prevTT:=(travel_time-lastTT)/(linkCount-1)]

trajectoriesData[,c("travel_seq", "link_seq"):=NULL]

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

setnames(finalWeatherData, c("date","hour"), c("curDate", "curHour"))
save(finalWeatherData, trajectoriesData, file="basicPreprocess.rda")
