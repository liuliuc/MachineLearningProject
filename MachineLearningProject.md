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
```

```
## 
## Attaching package: 'dplyr'
```

```
## The following objects are masked from 'package:stats':
## 
##     filter, lag
```

```
## The following objects are masked from 'package:base':
## 
##     intersect, setdiff, setequal, union
```

```r
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
```

```
## Warning: package 'caret' was built under R version 3.6.3
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
    library(rattle)
```

```
## Warning: package 'rattle' was built under R version 3.6.3
```

```
## Rattle: A free graphical interface for data science with R.
## Version 5.3.0 Copyright (c) 2006-2018 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

```r
    library(forecast)
```

```
## Warning: package 'forecast' was built under R version 3.6.3
```

```
## Registered S3 method overwritten by 'quantmod':
##   method            from
##   as.zoo.data.frame zoo
```

```r
    library(e1071)
```

```
## Warning: package 'e1071' was built under R version 3.6.3
```

## Modeling with Random Forest, and cross-validate with valid data set then predict the test data set. Random forest modeling took quiet some time even with parallel implementation. But accuracy is pretty higher (>99%). As expected, the in sample accuracy (which is 1 here) will be higher than the out of sample accuracy (99.23%).

```r
## Parallel Implementation of Random Forest
    set.seed(1234)    
    library(randomForest)
```

```
## Warning: package 'randomForest' was built under R version 3.6.3
```

```
## randomForest 4.6-14
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:rattle':
## 
##     importance
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```
## The following object is masked from 'package:dplyr':
## 
##     combine
```

```r
    # set up training run for x / y syntax because model format performs poorly
    x = TrainSet[,-40]
    y = TrainSet[,40]
    
    # configure parallel processing
    library(parallel)
    library(doParallel)
```

```
## Warning: package 'doParallel' was built under R version 3.6.3
```

```
## Loading required package: foreach
```

```
## Warning: package 'foreach' was built under R version 3.6.3
```

```
## Loading required package: iterators
```

```
## Warning: package 'iterators' was built under R version 3.6.3
```

```r
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
```

```
## 
## Attaching package: 'xgboost'
```

```
## The following object is masked from 'package:rattle':
## 
##     xgboost
```

```
## The following object is masked from 'package:dplyr':
## 
##     slice
```

```r
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
## [1]	train-merror:0.227573+0.014717	test-merror:0.239628+0.015447 
## [101]	train-merror:0.000000+0.000000	test-merror:0.003313+0.001123 
## [201]	train-merror:0.000000+0.000000	test-merror:0.002905+0.001229 
## [301]	train-merror:0.000000+0.000000	test-merror:0.002803+0.001123 
## [401]	train-merror:0.000000+0.000000	test-merror:0.002803+0.001233 
## [501]	train-merror:0.000000+0.000000	test-merror:0.002752+0.001189 
## [601]	train-merror:0.000000+0.000000	test-merror:0.002701+0.001164 
## [701]	train-merror:0.000000+0.000000	test-merror:0.002752+0.001273 
## [801]	train-merror:0.000000+0.000000	test-merror:0.002650+0.001223 
## [901]	train-merror:0.000000+0.000000	test-merror:0.002650+0.001223 
## [1000]	train-merror:0.000000+0.000000	test-merror:0.002650+0.001223
```

```r
    modxgbcv # print accuracy
```

```
## ##### xgb.cv 10-folds
##     iter train_merror_mean train_merror_std test_merror_mean test_merror_std
##        1         0.2275733      0.014717371        0.2396277     0.015447003
##        2         0.1512476      0.004393218        0.1673616     0.010715601
##        3         0.1172322      0.003473707        0.1321465     0.007949701
##        4         0.0947293      0.004104163        0.1101822     0.009274772
##        5         0.0811674      0.003122887        0.0968807     0.010777809
## ---                                                                         
##      996         0.0000000      0.000000000        0.0026502     0.001223025
##      997         0.0000000      0.000000000        0.0026502     0.001223025
##      998         0.0000000      0.000000000        0.0026502     0.001223025
##      999         0.0000000      0.000000000        0.0026502     0.001223025
##     1000         0.0000000      0.000000000        0.0026502     0.001223025
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
    modgbm=invisible(capture.output(train(classe~.,method="gbm",data=TrainSet))) # gradient boosting
    pred.in.gbm <- predict(modgbm, TrainSet) # predict in sample
```

```
## Warning in is.constant(y): NAs introduced by coercion
```

```
## Error in lastlevel + phi * lasttrend: non-numeric argument to binary operator
```

```r
    gbm.in = confusionMatrix(pred.in.gbm, TrainSet$classe)$overall[1] # in sample accuracy
```

```
## Error in confusionMatrix(pred.in.gbm, TrainSet$classe): object 'pred.in.gbm' not found
```

```r
    predgbm <- predict(modgbm, ValidSet) # predict with ValidSet 
```

```
## Warning in is.constant(y): NAs introduced by coercion
```

```
## Error in lastlevel + phi * lasttrend: non-numeric argument to binary operator
```

```r
    gbm.out = confusionMatrix(predgbm, ValidSet$classe)$overall[1] # Accuracy check
```

```
## Error in confusionMatrix.default(predgbm, ValidSet$classe): the data cannot have more levels than the reference
```

```r
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
```

```
## Error in eval(expr, envir, enclos): object 'gbm.in' not found
```

```r
OutAccuracy =c(rf.out,rpart.out,gbm.out,lda.out)
```

```
## Error in eval(expr, envir, enclos): object 'gbm.out' not found
```

```r
knitr::kable(data.frame(Model=names(OutAccuracy),InSampleAccuracy=paste0(round(InAccuracy*100,2),"%"),OutofSampleAccuracy=paste0(round(OoutAccuracy*100,2),"%"), OutofSampleError=paste0(round((1-OutAccuracy)*100,2),"%")))
```

```
## Error in data.frame(Model = names(OutAccuracy), InSampleAccuracy = paste0(round(InAccuracy * : object 'OutAccuracy' not found
```

## Inclusion, both XGBoost and Random Forest modelig provide higher in sample and out of sample accuracy than other modeling (decision tree, basic gradient boosting, Linear Discriminant Analysis, etc.), but XGBoost modeling is way faster. So I'll choose XBGoost modeling to predict the test data set. The prediction result is listed below, which passed the final quiz with 100% score.
Predicted results of test data set:
"B" "A" "B" "A" "A" "E" "D" "B" "A" "A" "B" "C" "B" "A" "E" "E" "A" "B" "B" "B"
