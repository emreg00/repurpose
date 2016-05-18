library(caret)
library(ROCR)
#library(ggplot2)
#library(reshape2)
#color.palette <- c("blue", "orange", "grey20", "red", "green", "yellow") 

data.dir = "/home/eguney/data/gottlieb/"
output.dir = data.dir

main<-function() {
    #set.seed(142341) 
    data.source = "zhang" #"oh"
    d = get.data.and.features(data.source)
    features = d$features
    d = d$data
    mod = train.model(d, features)
}

get.data.and.features<function(data.source=c("oh","zhan")) {
    data.source = match.arg(data.source)
    # Get class info and values 
    if(data.source == "oh") {
	file.name = paste0(data.dir, "oh.csv")
	d = read.csv(file.name, sep = "\t")
	d = d[,-c(1,2)]
	n = ncol(d)
	d[, n] = factor(d[, n])
	#colnames(d)[ncol(d)] = "flag"
	colnames(d) = c(paste0("X", 1:(n-1)), "flag")
	d$flag = factor(ifelse(d$flag == T, "true", "false"), levels=c("true", "false")) # to make true as the positive class
	# Choose features to use
	#features = c("d.ABI", "p.ABI", "c.ABI", "flag")
	#features = c("X1", "X2", "X3", "flag")
    } else if(data.source == "zhang") {
	d=read.csv("drug_disease.csv",row.names=1)
	e=read.csv("drug_sider.csv",row.names=1)
	f=cor(t(e))
	#!
    }
    features = colnames(d)
    indices = which(colnames(d) %in% features)
    d = d[,indices]
    print(summary(d))
    return(list(data=d, features=features);
}


train.model<-function(d, features) {
    #ctrl = trainControl(method = "repeatedcv", number = 10, repeats = 3)
    #ml.method = "rf" #"ctree" #"J48" #"glm" #"rf" #"gbm" 
    #ml.method = "glm" 
    ml.method = "ctree" 
    ctrl = trainControl(method = "cv", number = 10, classProbs=T, summaryFunction=twoClassSummary,  savePredictions = T)
    use.subset = NULL
 
    ## Scale all the columns
    ##e = d[,-which(colnames(d) == "flag")]
    ##e = predict(preProcess(e, method = c("center", "scale")), e) 
    ##e$flag = d$flag
    
    values = c()
    for(i in c(1:10)) {
    #for(i in c(1)) {
	# Balance classes in the data 
	indices.positive = which(d$flag == "true")
	# Choose random subset of the data
	if(!is.null(use.subset)) {
	    indices.positive = sample(indices.positive, use.subset) 
	}
	indices = which(d$flag == "false")
	indices = sample(indices, length(indices.positive))
	indices = c(indices.positive, indices)
	d = d[indices,]
	print(length(indices.positive))
	print(dim(d))

	# Create train - test partition
	#inTrain = createDataPartition(y = d$flag, p = 0.7, list = F)
	inTrain = createDataPartition(y = d$flag, p = 0.5, list = F) #!
	training = d[inTrain, ] 
	testing = d[-inTrain, ]

	#modFit = train(flag ~ ., data = training, method='ctree', tuneLength=10,
	#trControl = ctrl) #, classProbs=F, summaryFunction=twoClassSummary))
	modFit = train(flag  ~ ., data = training, method = ml.method, trControl = ctrl) #, verbose=F) 
	print(sprintf("(%d) AUC (train): %.3f", i, max(modFit$results[,"ROC"])))
	pred = predict(modFit, testing)
	#print(predictors(modFit))
	print(modFit)
	# Test set performance metrics
	a = confusionMatrix(pred, testing$flag)
	print(a)
	#rocValues = roc(pred, testing$flag)
	#a = aucRoc(rocValues)
	#print(c(i, a))
	if(ml.method == "rf") {
	    indices = modFit$pred$mtry == 2
	    score = modFit$pred$M[indices]
	    label = modFit$pred$obs[indices]
	} else if(ml.method == "ctree") {
	    score = modFit$pred[,"true"]
	    label = ifelse(modFit$pred[,"obs"] == "true", 1, 0)
	}
	pred = prediction(score, label)
	perf = performance(pred, "auc")
	a = perf@y.values[[1]]
	print(sprintf("(%d) AUC (test): %.3f", i, a))
	values = c(values, a)
    }
    print(summary(values))
}


main()

