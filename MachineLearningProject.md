---
title: "MachineLearningProject"
author: "Li"
date: "3/27/2020"
output: html_document
keep_md: true
---
## Get data and do explarotary analysis to clean up the data by removing all empty values and plit the training data into .75 as TrainSet for training the model and .25 as ValidSet for cross validation.

```r
# get data
    url_train="https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
    url_test="https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
    download.file(url_train, "pml-training.csv", method="curl", mode="wb")
    download.file(url_test, "pml-testing.csv", method="curl", mode="wb")
    # read in data and convert all empty values to NA
    training = read.csv("pml-training.csv",na.strings=c("NA","NaN"," ",""))
    testing = read.csv("pml-testing.csv",na.strings=c("NA","NaN"," ",""))
    library(dplyr)
    # only keep needed variables
    colsel = grep("belt|forearm|arm|dumbell|classe",names(training))
    trainsel=select(training, names(training)[colsel])
    testsel=select(testing, names(testing)[colsel])

# remove all NA data   
    trainclean=trainsel[,!colSums(is.na(trainsel)) > 0.1*nrow(trainsel)] 
    testclean=testsel[,!colSums(is.na(testsel)) > 0.1*nrow(testsel)]
    testclean=testclean[,-40] # remove the last column
    
# split clean train data to train set and validation
    train = sample(nrow(trainclean),0.75*nrow(trainclean), replace=FALSE)
    TrainSet = trainclean[train,]
    ValidSet = trainclean[-train,]

# import libraries
    library(caret)
    library(rattle)
    library(forecast)
    library(e1071)
```

## Modeling with Random Forest, and cross-validate with valid data set then predict the test data set. Random forest modeling took quiet some time even with parallel implementation. But accuracy is pretty higher (>99%). As expected, the in sample accuracy (which is 1 here) will be higher than the out of sample accuracy (99.23%).

```r
## Parallel Implementation of Random Forest
    set.seed(1234)    
    library(randomForest)
    
    # set up training run for x / y syntax because model format performs poorly
    x = TrainSet[,-40]
    y = TrainSet[,40]
    
    # configure parallel processing
    library(parallel)
    library(doParallel)
    cluster = makeCluster(detectCores()-1) # convention to leave 1 core for OS
    registerDoParallel(cluster) # open the cluster
    modrfCtrl = trainControl(method ="cv",number=5,allowParallel=TRUE) # Configure trainControl object
    modrf = train(x,y,method="rf",data=TrainSet,trCtrl=modrfCtrl) # develop training model
    stopCluster(cluster) # shut down the cluster
    registerDoSEQ() # force R to return to single treaded processing
    
    pred.train.rf = predict(modrf,TrainSet) # in sample predict
    rf.in = confusionMatrix(pred.train.rf, TrainSet$classe)$overall['Accuracy'] # in sample accuracy
    predrf = predict(modrf,ValidSet) # cross validate      
    rf.out = confusionMatrix(predrf, ValidSet$classe)$overall[1] # out of sample accuracy
    predrf.test = predict(modrf, testclean) # predict test data set 
    predrf.test # print the results for test data set
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

## Modeling with Extreme Gradient Boosting (xgboost), which has bulit-in cross-validation then predict the test data set. XGBoost modeling is very fast compared with rf modeling, and provide very high accuracy (~100%), the best modeling. xgb.cv provides the out of sample errors.

```r
## XGBoost Multinomial Classification
    library(xgboost)
  # convert the response factor to an integer class starting at 0
    trainlabel = as.integer(trainclean$classe)-1
  # convert data frame to matrix
    TrainMatrix=xgb.DMatrix(data=as.matrix(trainclean[,1:39]),label=as.matrix(trainlabel))
    TestMatrix=xgb.DMatrix(data=as.matrix(testclean))
  # Set parameters(default)
    params = list(booster ="gbtree",objective="multi:softprob",num_class=5,eval_metric="merror")
  # train model using full training set
    modxgb = xgb.train(params =params,data=TrainMatrix,nrounds=1000)
  # Predict outcomes with the test data
    predxgb = as.data.frame(predict(modxgb,newdata=TestMatrix,reshape=T))
    colnames(predxgb) = levels(TrainSet$classe)
  # label the highest probability with classe levels
    predxgb$predict = apply(predxgb,1,function(x) colnames(predxgb)[which.max(x)])
    predxgb$predict # print the predition result for test data set
```

```
##  [1] "B" "A" "B" "A" "A" "E" "D" "B" "A" "A" "B" "C" "B" "A" "E" "E" "A" "B" "B" "B"
```

```r
  # modeling with build-in cross validation and checking for out of sample error
    modxgbcv = xgb.cv(params=params,data=TrainMatrix,nrounds=1000,nfold=10,showsd=TRUE,
            stratified=TRUE,print_every_n=100,early_stop_round=20,maximize=FALSE,prediction=TRUE)
```

```
## [1]	train-merror:0.226905+0.015292	test-merror:0.238045+0.016671 
## [101]	train-merror:0.000000+0.000000	test-merror:0.004281+0.001599 
## [201]	train-merror:0.000000+0.000000	test-merror:0.003262+0.001122 
## [301]	train-merror:0.000000+0.000000	test-merror:0.003109+0.001081 
## [401]	train-merror:0.000000+0.000000	test-merror:0.003007+0.000896 
## [501]	train-merror:0.000000+0.000000	test-merror:0.002854+0.000889 
## [601]	train-merror:0.000000+0.000000	test-merror:0.002752+0.000797 
## [701]	train-merror:0.000000+0.000000	test-merror:0.002803+0.000830 
## [801]	train-merror:0.000000+0.000000	test-merror:0.002752+0.000889 
## [901]	train-merror:0.000000+0.000000	test-merror:0.002803+0.000947 
## [1000]	train-merror:0.000000+0.000000	test-merror:0.002803+0.000947
```

```r
    modxgbcv # print accuracy
```

```
## ##### xgb.cv 10-folds
##     iter train_merror_mean train_merror_std test_merror_mean test_merror_std
##        1         0.2269054      0.015291913        0.2380452    0.0166713673
##        2         0.1534164      0.010155163        0.1678701    0.0112953816
##        3         0.1177023      0.005869483        0.1347970    0.0079189760
##        4         0.0929514      0.004846219        0.1111490    0.0088771524
##        5         0.0803804      0.003510415        0.0968796    0.0079966122
## ---                                                                         
##      996         0.0000000      0.000000000        0.0028032    0.0009472302
##      997         0.0000000      0.000000000        0.0028032    0.0009472302
##      998         0.0000000      0.000000000        0.0028032    0.0009472302
##      999         0.0000000      0.000000000        0.0028032    0.0009472302
##     1000         0.0000000      0.000000000        0.0028032    0.0009472302
```

## Other modelings: Decision tree modeling (rpart) is fast but accuracy is only 53%; the basic gradient boosting took sometime with accuracy 93%; Linear Discriminant Analysis (lda) is fast, but accuracy is only 56%


```r
## other modelings
    # decision tree modeling
    modrpart=train(classe~.,method="rpart",data=TrainSet) # decision tree
    pred.in.rpart <- predict(modrpart, TrainSet) # predict in sample
    rpart.in = confusionMatrix(pred.in.rpart, TrainSet$classe)$overall[1] # in sample accuracy
    predrpart <- predict(modrpart, ValidSet) # predict with ValidSet
    rpart.out = confusionMatrix(predrpart, ValidSet$classe)$overall[1] # Accuracy check
    
    # gradient boosting modeling
    modgbm =train(classe~.,method="gbm",data=TrainSet,verbose=FALSE) # gradient boosting
    pred.in.gbm <- predict(modgbm, TrainSet) # predict in sample
    gbm.in = confusionMatrix(pred.in.gbm, TrainSet$classe)$overall[1] # in sample accuracy
    predgbm <- predict(modgbm, ValidSet) # predict with ValidSet 
    gbm.out = confusionMatrix(predgbm, ValidSet$classe)$overall[1] # Accuracy check
    
    #  Linear Discriminant Analysis
    modlda=train(classe~.,method="lda",data=TrainSet) 
    pred.in.lda <- predict(modlda, TrainSet) # predict in sample
    lda.in = confusionMatrix(pred.in.lda, TrainSet$classe)$overall[1] # in sample accuracy
    predlda <- predict(modlda, ValidSet) # predict with ValidSet
    lda.out = confusionMatrix(predlda, ValidSet$classe)$overall[1]
```

## Cross validation

```r
InAccuracy = c(rf.in,rpart.in,gbm.in,lda.in)
OutAccuracy =c(rf.out,rpart.out,gbm.out,lda.out)
knitr::kable(data.frame(Model=names(OutAccuracy),InSampleAccuracy=paste0(round(InAccuracy*100,2),"%"),OutofSampleAccuracy=paste0(round(OoutAccuracy*100,2),"%"), OutofSampleError=paste0(round((1-OutAccuracy)*100,2),"%")))
```

```
## Error in paste0(round(OoutAccuracy * 100, 2), "%"): object 'OoutAccuracy' not found
```

## Inclusion, both XGBoost and Random Forest modelig provide higher in sample and out of sample accuracy than other modeling (decision tree, basic gradient boosting, Linear Discriminant Analysis, etc.), but XGBoost modeling is way faster. So I'll choose XBGoost modeling to predict the test data set. The prediction result is listed below, which passed the final quiz with 100% score.
Predicted results of test data set:
"B" "A" "B" "A" "A" "E" "D" "B" "A" "A" "B" "C" "B" "A" "E" "E" "A" "B" "B" "B"
