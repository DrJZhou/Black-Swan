library(h2o)
library(reshape2)
library(lubridate)
library(data.table)
source("DLHelper.R")

load("dataImputation.rda")
sapply(finalData, function(x){sum(is.na(x))})
######################################################################
### The data preprocess
######################################################################
subTrainExtendList  <- c("main","extendLag5","extendLead5","extendLag3","extendLead3")
subTrainData    <- finalData[mark=='train' & weekIndex>2 & extendFlag %in% subTrainExtendList, ]

validExtendList <- c("main","extendLag5","extendLead5")
subValidData    <- finalData[mark !='test' & weekIndex==2 & extendFlag=='main', ]
subValidData2   <- finalData[mark != 'test' & weekIndex==2 & extendFlag %in% validExtendList, ]
subValidData3   <- finalData[mark !='test' & weekIndex==2 & extendFlag %in% c("extendLag10","extendLead10","extendLag8","extendLead8"), ]
###The submit data
submitData      <- finalData[weekIndex==1 & extendFlag=='main', ]

######################################################################
### The h2o data preprocess
######################################################################
totalFeatureList    <- c("intersectionTollgate","hourWindow","MWInterval", "ATTLag6", "ATTLag5", "ATTLag4", "ATTLag3", "ATTLag2", "ATTLag1","rel_humidity", "precipitation", "isHoliday", "ATT120SdLag", "ATT60SdLag1", "ATTNumLag1", "ATT60NumLag1", "ATT60MeanLag1","ATT40NumLag1", "ATT40MeanLag1", "ATT120MeanLag")
h2o.init(nthreads=-1, min_mem_size='32G')
h2oSubmitData       <- as.h2o(subset(submitData, select=c(totalFeatureList,"ATT")), destination_frame="h2oSubmitData")
h2oSubTrainData     <- as.h2o(subset(subTrainData, select=c(totalFeatureList,"ATT")), destination_frame="h2oSubTrainData")

h2oSubValidData     <- as.h2o(subset(subValidData,select=c(totalFeatureList,"ATT")), destination_frame="h2oSubValidData")
h2oSubValidData2    <- as.h2o(subset(subValidData2,select=c(totalFeatureList,"ATT")), destination_frame="h2oSubValidData2")
h2oSubValidData3    <- as.h2o(subset(subValidData3,select=c(totalFeatureList,"ATT")), destination_frame="h2oSubValidData3")

###################################################
#### The h2o machine learning
#################################################3
featureList         <- c("intersectionTollgate","hourWindow","MWInterval", "ATTLag6", "ATTLag5", "ATTLag4", "ATTLag3", "ATTLag2", "ATTLag1","rel_humidity", "precipitation","isHoliday","ATT120SdLag","ATT60NumLag1","ATT60MeanLag1")
###The first model
XNames          <- featureList; YName   <- "ATT"
h2oDLFit        <- h2o.deeplearning(XNames, YName,training_frame=h2oSubTrainData, validation_frame=h2oSubValidData,
                                    activation="Rectifier",
                                    distribution="laplace", hidden=c(64,64,64),
                                    l1=1e-7, input_dropout_ratio = 0.2, epochs=2000,
                                    initial_weight_distribution="UniformAdaptive",
                                    stopping_rounds = 5, stopping_tolerance = 0.001, stopping_metric = "MAE",
                                    seed=123456, variable_importance=T)
dlPred1         <- as.vector(predict(h2oDLFit, h2oSubValidData))
##The validation error
mapeFunc(subValidData$ATT, dlPred1);
##The variable importance
VI  <- h2o.varimp(h2oDLFit);    print(VI)
##################################################################
###The randomized grid search
##################################################################
search_criteria = list(strategy = "RandomDiscrete", max_models = 150, max_runtime_secs = 12000, seed=123456)
neuronList  <- c(16,32,64,96,128)
neuronList3 <- lapply(1:200, function(x)sample(neuronList, 3, replace=T))
neuronList4 <- lapply(1:200, function(x)sample(neuronList, 4, replace=T))
hidden_opt  <- c(unique(neuronList3),  unique(neuronList4))

activation_opt    <- c("Rectifier","Tanh")
distribution_opt    <- c("laplace")
hyper_params        <- list(l1=l1_opt,l2=l2_opt,distribution=distribution_opt,activation=activation_opt,
                            hidden=hidden_opt,input_dropout_ratio=input_dropout_opt,
                            epsilon=epsilon_opt, rho=rho_opt,max_w2=max_w2_opt)
####The first model
DLGrid  <- h2o.grid("deeplearning", grid_id = "DLGrid",
            hyper_params = hyper_params,
            x = XNames, y = YName,
            training_frame = h2oSubTrainData,
            validation_frame = h2oSubValidData2,
            score_validation_samples=round(nrow(subValidData2)*0.8),
            epochs = 3000,
            stopping_rounds = 3,
            stopping_tolerance = 1e-4,
            stopping_metric = "deviance",
            max_runtime_secs=300,
            search_criteria=search_criteria)

###The model evaluation
validDataPair   <- list(subValidData,h2oSubValidData)
validDataPair2  <- list(subValidData2,h2oSubValidData2)
validDataPair3  <- list(subValidData3,h2oSubValidData3)
validDataList   <- list(validDataPair,validDataPair2,validDataPair3)

DLGridResult    <- adaGridEvalFunc(DLGrid,validDataList,23)
DLGridResult    <- DLGridResult[order(validMAPE1),]

###################################################
### The validation result of model1
###################################################
###manually select the best model list
modelResult 	<- head(DLGridResult,15);
#modelResult     <- modelResult[-c(5,12:15),]
modelSummary    <- DLSummaryFunc(modelResult)

###############################################################3
### The Pred of Submit
###############################################################3
bestK		<- 7
submitModelList <- head(modelResult,bestK)$modelId

###The mean pred
submitPredList       <- sapply(submitModelList, function(id){
        subModel    <- h2o.getModel(id)
        subPred     <- as.vector(predict(subModel,h2oSubmitData))
        subPred
})
submitPred          <- rowMeans(submitPredList)

ATTNNSubmit        <- copy(submitData)
ATTNNSubmit$pred   <- submitPred

ATTNNSubmit[, time_window:=paste("\"[", timeStamp, ",", timeStamp+dminutes(20), ")\"", sep="")]
ATTNNSubmit   <- subset(ATTNNSubmit, select=c("intersection_id", "tollgate_id", "time_window", "pred"))
write.table(ATTNNSubmit,file=paste(Sys.Date(), "extendBaseline.csv", sep=""),  row.names=F,col.names=F, quote=F, sep=",")

save(ATTNNSubmit,DLGridResult,modelResult,modelSummary,file=paste(Sys.Date(), "extendBaseline.Rdata", sep=""))


