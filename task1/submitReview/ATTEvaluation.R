###################################################
#####The model evaluation
###################################################
mapeFunc <- function(y, yhat){mean(abs((y - yhat)/y))}

### The function for caret turning
mapeSummary  <- function (data,lev = NULL,model = NULL) {
	out	<- mean(abs((data$obs-data$pred)/data$obs))
      names(out) <- "MAPE"
	out 
}

### The function for xgboost 
mapeXgb <- function(preds, dtrain) {
    labels  <- getinfo(dtrain, "label")
    err		<- mean(abs((labels-preds)/labels))	
    return(list(metric = "madNew", value = err))
}

### The function for lightgbm
mapelgb <- function(preds, dtrain) {
    labels  <- getinfo(dtrain, "label")
    err     <- mean(abs((labels-preds)/labels))
    return(list(name = "error", value = err, higher_better=FALSE))
}
