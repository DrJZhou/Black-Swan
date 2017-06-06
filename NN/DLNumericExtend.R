library(h2o)
library(reshape2)
library(lubridate)
library(data.table)

source("DLHelper.R")
load("cleanFeatureEngine.rda")
#########################################################
### Data for model
#########################################################
extendData[,c("tollgateDirection", "hourWindow"):=list(as.factor(tollgateDirection),as.factor(hourWindow))]
extendData[,target:=log(ATV)]

### Take the last 4 days as validation
validDateList   <- seq(as.Date("2016-10-14"), as.Date("2016-10-17"),1)
### training data
subTrainData    <- extendData[mark !='submit' & extendFlag=='main' & !(curDate %in% validDateList),]
subTrainData2   <- extendData[mark  !='submit'  & !(curDate %in% validDateList) & extendFlag %in% c("main","lag5","lead5"),]
###Validation data
subValidData    <- extendData[mark  !='submit' & (curDate %in% validDateList) ,]
subValidData2   <- extendData[mark  !='submit' & (curDate %in% validDateList) & extendFlag %in% c("main","lag5","lead5"),]
subValidData3   <- extendData[mark  !='submit' & (curDate %in% validDateList) & extendFlag %in% c("lag10","lead10","lead8","lag8"),]
### Train and submit
trainData       <- extendData[mark  !='submit' ,]
submitData      <- extendData[mark=='submit',]

### The h2o data
totalFeatureList         <- colnames(extendData)[c(3:5,9,14:57)]
h2o.init(nthreads=-1,min_mem_size='12G')
h2oSubmitData       <- as.h2o(subset(submitData, select=totalFeatureList), destination_frame="h2oSubmitData")
h2oTrainData        <- as.h2o(subset(trainData, select=totalFeatureList), destination_frame="h2oTrainData")
### The h2o subtrain
h2oSubTrainData     <- as.h2o(subset(subTrainData, select=totalFeatureList), destination_frame="h2oSubTrainData")
h2oSubTrainData2    <- as.h2o(subset(subTrainData2, select=totalFeatureList), destination_frame="h2oSubTrainData2")

h2oSubValidData     <- as.h2o(subset(subValidData, select=totalFeatureList), destination_frame="h2oSubValidData")
h2oSubValidData2    <- as.h2o(subset(subValidData2, select=totalFeatureList), destination_frame="h2oSubValidData2")
h2oSubValidData3    <- as.h2o(subset(subValidData3, select=totalFeatureList), destination_frame="h2oSubValidData3")

######################################################################
###The basic deeplearning training
######################################################################
basicFeatureList    <-c("tollgateDirection", "hourWindow","MWInterval","isHoliday")
featureList         <- c(basicFeatureList,"ATVLag1","ATVLag2", "twoATVLag2", "twoATVLag3","twoEtcVLag1")
hidden_opt     <- c(1024)
XNames          <- featureList; YName   <- "ATV"
h2oDLFit        <- h2o.deeplearning(XNames, YName,training_frame=h2oSubTrainData2, validation_frame=h2oSubValidData2,
                                activation="Rectifier",distribution="laplace", hidden=hidden_opt,
                                epochs=5000,stopping_rounds = 3, stopping_tolerance = 1e-5, stopping_metric = "deviance",
                                l1=1e-5, l2=1e-5,           # Regularization for overfitting
                                input_dropout_ratio=0.2,    # Input dropout ratio for overfitting
                                max_w2 = 3.4028235e+38,     # The default max square weight of neuron
                                adaptive_rate=T,            # Enables adaptive learning rate
                                variable_importances=T,
                                seed=123456)
print(DLEvalFunc(h2oDLFit,subValidData,h2oSubValidData))
print(DLEvalFunc(h2oDLFit,subValidData2,h2oSubValidData2))
print(DLEvalFunc(h2oDLFit,subValidData3,h2oSubValidData3))

VI  <- h2o.varimp(h2oDLFit);    print(VI)

######################################################################
###The main data training
######################################################################
search_criteria = list(strategy = "RandomDiscrete", max_models = 50, max_runtime_secs = 4200, seed=123456)
baseNeuronList  <- c(64, 128,256,512,768,1024)
hidden_opt      <- c(as.list(baseNeuronList))
distribution_opt    <-c("laplace","huber")

validDataPair  <- list(subValidData,h2oSubValidData)
validDataPair2  <- list(subValidData2,h2oSubValidData2)
validDataPair3  <- list(subValidData3,h2oSubValidData3)
validDataList   <- list(validDataPair, validDataPair2,validDataPair3)

activation_opt  <- c("Rectifier");
###The parameters grid
hyperParams     <- list(activation=activation_opt,distribution=distribution_opt,
                            hidden = hidden_opt,
                            max_w2=max_w2_opt,
                            l1=l1_opt,l2=l2_opt,input_dropout_ratio=input_dropout_opt,
                            epsilon=epsilon_opt, rho=rho_opt)

mainGrid    <- h2o.grid("deeplearning", grid_id = "mainGrid",
                    x = XNames, y = YName,
                    training_frame = h2oSubTrainData2,validation_frame = h2oSubValidData2,
                    hyper_params = hyperParams,
                    stopping_rounds = 3, stopping_tolerance = 1e-5, stopping_metric = "deviance", epochs=5000,##convenge based early stopping
                    max_runtime_secs=240,
                    search_criteria=search_criteria)
mainResult  <- adaGridEvalFunc(mainGrid, validDataList,17)

activation_opt  <- c("RectifierWithDropout");
###The parameters grid
hyperParams     <- list(activation=activation_opt,distribution=distribution_opt,
                            hidden = hidden_opt,hidden_dropout_ratios=one_hidden_dropout_opt,
                            max_w2=max_w2_opt,
                            l1=l1_opt,l2=l2_opt,input_dropout_ratio=input_dropout_opt,
                            epsilon=epsilon_opt, rho=rho_opt)

mainGrid2    <- h2o.grid("deeplearning", grid_id = "mainGrid2",
                    x = XNames, y = YName,
                    training_frame = h2oSubTrainData2,validation_frame = h2oSubValidData2,
                    hyper_params = hyperParams,
                    stopping_rounds = 3, stopping_tolerance = 1e-5, stopping_metric = "deviance", epochs=5000,##convenge based early stopping
                    max_runtime_secs=240,
                    search_criteria=search_criteria)
mainResult2  <- adaGridEvalFunc(mainGrid2, validDataList,17)

##########################################################
### The model result
##########################################################
###manually select the best model list
modelResult     <- head(mainResult,10)
#modelResult     <- modelResult[-c(5,6,10)]
modelResult2    <- head(mainResult2,10)
modelSummary1   <- DLModelSummary(modelResult)
modelSummary2   <- DLModelSummary(modelResult2)

##########################################################
###The submit 
##########################################################
bestK1	<- 5;   bestK2  <- 5
submitPredList      <- sapply(modelResult$modelId, function(id){
        subModel    <- h2o.getModel(id)
        subPred     <- as.vector(predict(subModel,h2oSubmitData))
        subPred})
submitPredList2      <- sapply(modelResult2$modelId, function(id){
        subModel    <- h2o.getModel(id)
        subPred     <- as.vector(predict(subModel,h2oSubmitData))
        subPred})
pred1	<- rowMeans(submitPredList[,1:bestK1])
pred2	<- rowMeans(submitPredList2[,1:bestK2])

#####NN submit
submitData$pred1     	<- pred1
submitData$pred2	<- pred2
submitData[,pred:=pred1]
submitData[pred2<pred,pred:=pred2]
submitData[, time_window:=paste("\"[", timeStamp, ",", timeStamp+dminutes(20), ")\"", sep="")]
submitResult   <- subset(submitData, select=c("tollgate_id","time_window","direction", "pred"))
write.table(submitResult,file=paste(Sys.Date(), "../answer/ATVNumeric5.csv", sep=""),  row.names=F,col.names=F, quote=F, sep=",")
