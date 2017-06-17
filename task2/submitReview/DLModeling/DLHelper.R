#########################################################################
### Basic parameter list for deep learning
#########################################################################
###Regularization
l1_opt      <- c(0, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7)
l2_opt      <- c(0, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7)
input_dropout_opt   <- c(0,0.05,0.1,0.2)
###Max square weight of one neuron
max_w2_opt  <-  seq(10,50, by=10)

### The ada-delta learning
epsilon_opt <- c(1e-4,1e-6,1e-8,1e-10)
rho_opt     <- c(0.9,0.95,0.99,0.999)

### The momentum learning
rate_opt        <- c(1e-4, 1e-3, 1e-2); rate_annealing_opt  <- c(1e-5,1e-6,1e-7,1e-8,1e-9)
momentum_start  <- seq(0.5,0.9,by=0.1); momentum_stable     <- 0.99

###The neuron structure for h2o training
baseList        <- c(16,32,64,128,256,512,768,1024,2048)

#####NeuronList for one layer
oneLayerNeuronList      <- as.list(baseList)
twoLayerNeuronList      <- lapply(baseList, function(x)c(x,x))
threeLayerNeuronList    <- lapply(baseList, function(x)c(x,x,x))
fourLayerNeuronList     <- lapply(baseList, function(x)c(x,x,x,x))
fiveLayerNeuronList     <- lapply(baseList, function(x)c(x,x,x,x,x))
sixLayerNeuronList      <- lapply(baseList, function(x)c(x,x,x,x,x,x))

##The  hidden dropout
one_hidden_dropout_opt	    <- list(c(0.5), c(0.4), c(0.3))
two_hidden_dropout_opt	    <- list(c(0.5,0.5), c(0.4,0.4), c(0.3,0.3))
three_hidden_dropout_opt    <- list(c(0.5,0.5,0.5), c(0.4,0.4,0.4), c(0.3,0.3,0.3))
four_hidden_dropout_opt     <- list(c(0.5,0.5,0.5,0.5), c(0.4,0.4,0.4,0.4), c(0.3,0.3,0.3,0.3))
five_hidden_dropout_opt     <- list(c(0.5,0.5,0.5,0.5,0.5), c(0.4,0.4,0.4,0.4,0.4), c(0.3,0.3,0.3,0.3,0.3))
six_hidden_dropout_opt      <- list(c(0.5,0.5,0.5,0.5,0.5,0.5), c(0.4,0.4,0.4,0.4,0.4,0.4), c(0.3,0.3,0.3,0.3,0.3,0.3))

#################################################################################
### The Evaluation For Deep Learning
#################################################################################
mapeFunc <- function(y, yhat){mean(abs((y - yhat)/y))}

DLEvalFunc	<-  function(DLModel, validData, h2oValidData){
    dlPred          <- as.vector(predict(DLModel,h2oValidData))
    hidden          <- paste(DLModel@allparameters$hidden, collapse=",")
    ##The real epochs
    epochsList      <-  DLModel@model$scoring_history$epochs
    bestEpochIndex  <-  which.min(DLModel@model$scoring_history$validation_deviance)
    bestEpoch       <-  epochsList[bestEpochIndex]
    ##Evaluation Metrics 
    trainMAE    <- DLModel@model$training_metrics@metrics$mae
    validMAE    <- DLModel@model$validation_metrics@metrics$mae
    validMAPE   <- mapeFunc(validData$ATT,dlPred)
    modelMetrics        <- c(bestEpoch,trainMAE,validMAE,validMAPE)
    names(modelMetrics) <- c("bestEpoch","trainMAE","validMAE","validMAPE")
    return(modelMetrics)
}

manualDLEvalFunc    <- function(DLGrid, validData, h2oValidData){
    modelIds        <- DLGrid@model_ids
    subModelList    <- lapply(modelIds, function(id){
        subModel    <- h2o.getModel(id)
        allparameters   <- subModel@allparameters
        activation     <- allparameters$activation
        distribution    <- allparameters$distribution

        dlPred      <- as.vector(predict(subModel,h2oValidData))
        hidden      <- paste(subModel@allparameters$hidden, collapse=",")
        singleNeuron    <- subModel@allparameters$hidden[1]
        layerNum    <- length(subModel@allparameters$hidden)
        ##The real epochs
        epochsList      <-  subModel@model$scoring_history$epochs
        bestEpochIndex  <-  which.min(subModel@model$scoring_history$validation_deviance)
        bestEpoch       <-  epochsList[bestEpochIndex]
        ##Evaluation Metrics 
        trainMAE    <- subModel@model$training_metrics@metrics$mae
        validMAE    <- subModel@model$validation_metrics@metrics$mae
        validMAPE   <- mapeFunc(validData$ATT,dlPred)
        modelMetrics    <- data.table(activation,distribution,singleNeuron,layerNum,hidden,bestEpoch,trainMAE,validMAE,validMAPE)
    })
    result	<- rbindlist(subModelList)
    result      <- result[order(singleNeuron),]
}

adaGridEvalFunc    <- function(DLGrid, validDataList, P){
    modelIds        <- DLGrid@model_ids
    subModelList    <- lapply(modelIds, function(id){
        subModel        <- h2o.getModel(id)
        ##The real epochs
        epochsList      <-  subModel@model$scoring_history$epochs
        bestEpochIndex  <-  which.min(subModel@model$scoring_history$validation_deviance)
        bestEpoch       <-  epochsList[bestEpochIndex]
        ##The parameters
        allparameters   <- subModel@allparameters
        activation      <- allparameters$activation
        distribution    <- allparameters$distribution
        max_w2          <- allparameters$max_w2
        hidden          <- paste(allparameters$hidden, collapse=",")
        layerNum        <- length(subModel@allparameters$hidden)
        l1              <- allparameters$l1
        l2              <- allparameters$l2
        inputDropoutRatio   <- allparameters$input_dropout_ratio
        epsilon         <- allparameters$epsilon
        rho             <- allparameters$rho
        ##The weight of the model
        weightList      <- c(P, allparameters$hidden)
        weight          <- 0
        for(i in 1:layerNum){weight      <- weight + weightList[i]*weightList[i+1]}
         ##The train residaul deviance   
        trainMAE        <- subModel@model$training_metrics@metrics$mae
        validMAE        <- subModel@model$validation_metrics@metrics$mae
        resDeivance     <- subModel@model$validation_metrics@metrics$mean_residual_deviance
        ###The MAPE in the validation
        validMAPEList   <- sapply(1:length(validDataList), function(i){
        validData       <- validDataList[[i]][[1]]
        h2oValidData    <- validDataList[[i]][[2]]
        subPred         <- as.vector(predict(subModel, h2oValidData))
        validMAPE       <- mapeFunc(validData$ATT,subPred)
        validMAPE})     
        validMAPE1      <- validMAPEList[1]; validMAPE2 <- validMAPEList[2];    validMAPE3 <- validMAPEList[3]
        data.table(modelId=id, activation, distribution,bestEpoch,max_w2,layerNum, hidden,weight, l1,l2, inputDropoutRatio, epsilon, rho,trainMAE,validMAE,resDeivance, validMAPE1, validMAPE2,validMAPE3)
    })
    result  <- rbindlist(subModelList);
    result  <- result[order(validMAPE3),];
    return(result)
}   
LogAdaGridEvalFunc    <- function(DLGrid, validDataList, P){
    modelIds        <- DLGrid@model_ids
    subModelList    <- lapply(modelIds, function(id){
        subModel        <- h2o.getModel(id)
        ##The real epochs
        epochsList      <-  subModel@model$scoring_history$epochs
        bestEpochIndex  <-  which.min(subModel@model$scoring_history$validation_deviance)
        bestEpoch       <-  epochsList[bestEpochIndex]
        ##The parameters
        allparameters   <- subModel@allparameters
        activation      <- allparameters$activation
        distribution    <- allparameters$distribution
        max_w2          <- allparameters$max_w2
        hidden          <- paste(allparameters$hidden, collapse=",")
        layerNum        <- length(subModel@allparameters$hidden)
        l1              <- allparameters$l1
        l2              <- allparameters$l2
        inputDropoutRatio   <- allparameters$input_dropout_ratio
        epsilon         <- allparameters$epsilon
        rho             <- allparameters$rho
        ##The weight of the model
        weightList      <- c(P, allparameters$hidden)
        weight          <- 0
        for(i in 1:layerNum){weight      <- weight + weightList[i]*weightList[i+1]}
         ##The train residaul deviance   
        trainMAE        <- subModel@model$training_metrics@metrics$mae
        validMAE        <- subModel@model$validation_metrics@metrics$mae
        resDeivance     <- subModel@model$validation_metrics@metrics$mean_residual_deviance
        ###The MAPE in the validation
        validMAPEList   <- sapply(1:length(validDataList), function(i){
        validData       <- validDataList[[i]][[1]]
        h2oValidData    <- validDataList[[i]][[2]]
        subPred         <- exp(as.vector(predict(subModel, h2oValidData)))
        validMAPE       <- mapeFunc(validData$ATT,subPred)
        validMAPE})
        validMAPE1      <- validMAPEList[1]; validMAPE2 <- validMAPEList[2];    validMAPE3 <- validMAPEList[3]
        data.table(modelId=id, activation, distribution,bestEpoch,max_w2,layerNum, hidden,weight, l1,l2, inputDropoutRatio, epsilon, rho,trainMAE,validMAE,resDeivance, validMAPE1, validMAPE2,validMAPE3)
    })
    result  <- rbindlist(subModelList);
    result  <- result[order(validMAPE3),];
    return(result)
}



adaDeltaDropoutEvalFunc   <- function(DLGrid, validData, h2oValidData){
    modelIds        <- DLGrid@model_ids
    subModelList    <- lapply(modelIds, function(id){
            subModel <- h2o.getModel(id)
            ##The parameters
            allparameters   <- subModel@allparameters
            activation     <- allparameters$activation
            distribution    <- allparameters$distribution
            max_w2          <- allparameters$max_w2
            hidden          <- paste(allparameters$hidden, collapse=",")
            l1              <- allparameters$l1
            l2              <- allparameters$l2
            inputDropoutRatio   <- allparameters$input_dropout_ratio
            hiddenDropoutRatio  <- paste(allparameters$hidden_dropout_ratios,collapse=",")

            epsilon         <- allparameters$epsilon
            rho             <- allparameters$rho
            ##The train residaul deviance
            trainMAE        <- subModel@model$training_metrics@metrics$mae
            validMAE        <- subModel@model$validation_metrics@metrics$mae

            subPred         <- as.vector(predict(subModel, h2oValidData))
            mapeErr    <- mapeFunc(validData$ATT,subPred)
            data.table(modelId=id, activation, distribution,max_w2, hidden, l1,l2, inputDropoutRatio,hiddenDropoutRatio, 
epsilon, rho,trainMAE,validMAE, mapeErr)
    })
    result  <- rbindlist(subModelList);
    result  <- result[order(mapeErr),];
    return(result)
}

momentumEvalFunc    <- function(DLGrid, validData, h2oValidData){
    modelIds        <- DLGrid@model_ids
    subModelList    <- lapply(modelIds, function(id){
            subModel <- h2o.getModel(id)
            ##The parameters
            allparameters   <- subModel@allparameters
            activation     <- allparameters$activation
            distribution    <- allparameters$distribution
            max_w2          <- allparameters$max_w2
            hidden          <- paste(allparameters$hidden, collapse=",")
            l1              <- allparameters$l1
            l2              <- allparameters$l2
            inputDropoutRatio   <- allparameters$input_dropout_ratio
	    #momentu parameters
            rate            <- allparameters$rate
            rate_annealing  <- allparameters$rate_annealing
            momentum_start  <- allparameters$momentum_start
            momentum_ramp   <- allparameters$momentum_ramp
            momentum_stable <- allparameters$momentum_stable
            ##The train residaul deviance
            trainMAE        <- subModel@model$training_metrics@metrics$mae
            validMAE        <- subModel@model$validation_metrics@metrics$mae

            subPred         <- as.vector(predict(subModel, h2oValidData))
            mapeErr    <- mapeFunc(validData$ATT,subPred)
            data.table(modelId=id, activation, distribution,max_w2, hidden, l1,l2, inputDropoutRatio,
                       rate, rate_annealing,momentum_start,momentum_ramp,momentum_stable,trainMAE,validMAE, mapeErr)
    })
    result  <- rbindlist(subModelList);
    result  <- result[order(mapeErr),];
    return(result)
}

#################################################
###The model summary
#################################################
DLSummaryFunc    <- function(modelResult){
mainPredList   <- sapply(modelResult$modelId, function(id){
    subModel    <- h2o.getModel(id)
    subPred    <- as.vector(predict(subModel,h2oSubValidData))
    subPred
})
mainPredList2   <- sapply(modelResult$modelId, function(id){
    subModel    <- h2o.getModel(id)
    subPred2    <- as.vector(predict(subModel,h2oSubValidData2))
    subPred2
})
mainPredList3   <- sapply(modelResult$modelId, function(id){
    subModel    <- h2o.getModel(id)
    subPred3    <- as.vector(predict(subModel,h2oSubValidData3))
    subPred3
})
mainSubmitList   <- sapply(modelResult$modelId, function(id){
    subModel    <- h2o.getModel(id)
    subPred    <- as.vector(predict(subModel,h2oSubmitData))
    subPred
})
###Determine the best K=5
KList           <- 2:nrow(modelResult)
validPredList   <- lapply(KList, function(x){
        print(x)
    validPred   <- rowMeans(mainPredList[,1:x])
    validPred2  <- rowMeans(mainPredList2[,1:x])
    validPred3  <- rowMeans(mainPredList3[,1:x])
    validMAPE   <- mapeFunc(subValidData$ATT, validPred)
    validMAPE2  <- mapeFunc(subValidData2$ATT, validPred2)
    validMAPE3  <- mapeFunc(subValidData3$ATT, validPred3)
    meanPred    <- mean(rowMeans(mainSubmitList[,1:x]))
    data.table(K=x,validMAPE, validMAPE2,validMAPE3,meanPred)
})
validPredResult <- rbindlist(validPredList)
print(colMeans(mainSubmitList));  print(validPredResult)
return(validPredResult)
}
