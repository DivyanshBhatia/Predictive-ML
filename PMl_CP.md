Predictive Machine Learning Course Project
========================================================
# Executive Summary
This report Presents a write up on a Machine Learning Algorithm to predict correct class of exercises based on given data. The Attribute Classe is the outcome variable while there are 158 predictor variables to choose from. 


### cross validation done in svm
using parameter cross =10

```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.1.1
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```
## Warning: package 'ggplot2' was built under R version 3.1.1
```

```r
library(kernlab)
```

```
## Warning: package 'kernlab' was built under R version 3.1.1
```

```r
training_data <- read.csv("pml-training_new.csv",header=TRUE,strip.white=TRUE,na.strings=c("",".","NA"))
training_data <- as.data.frame(training_data)
training_data[is.na(training_data)]<-0.0
```

```
## Warning: invalid factor level, NA generated
## Warning: invalid factor level, NA generated
## Warning: invalid factor level, NA generated
## Warning: invalid factor level, NA generated
## Warning: invalid factor level, NA generated
## Warning: invalid factor level, NA generated
## Warning: invalid factor level, NA generated
## Warning: invalid factor level, NA generated
## Warning: invalid factor level, NA generated
## Warning: invalid factor level, NA generated
## Warning: invalid factor level, NA generated
## Warning: invalid factor level, NA generated
## Warning: invalid factor level, NA generated
## Warning: invalid factor level, NA generated
## Warning: invalid factor level, NA generated
## Warning: invalid factor level, NA generated
## Warning: invalid factor level, NA generated
## Warning: invalid factor level, NA generated
## Warning: invalid factor level, NA generated
## Warning: invalid factor level, NA generated
## Warning: invalid factor level, NA generated
## Warning: invalid factor level, NA generated
## Warning: invalid factor level, NA generated
## Warning: invalid factor level, NA generated
## Warning: invalid factor level, NA generated
## Warning: invalid factor level, NA generated
## Warning: invalid factor level, NA generated
## Warning: invalid factor level, NA generated
## Warning: invalid factor level, NA generated
## Warning: invalid factor level, NA generated
## Warning: invalid factor level, NA generated
## Warning: invalid factor level, NA generated
## Warning: invalid factor level, NA generated
```

```r
intrain <- createDataPartition(y=training_data$classe,p=0.75,list=FALSE)
training <- training_data[intrain,]
testing <- training_data[-intrain,]
ncol(training)
```

```
## [1] 160
```

```r
nums <- sapply(training, is.numeric)
test_nums <- sapply(testing,is.numeric)
indx <- which(is.na(training[,])==TRUE) 
test_indx <- which(is.na(testing[,])==TRUE) 
training = data.frame(training)
testing = data.frame(testing)
data_num <- training[,nums]

data_test_nums <- testing[,test_nums]
M <- abs(cor(data_num))
diag(M) <- 0
new_data_num <- data_num[,unique(which(M>0.9,arr.ind=T)[1:ncol(M)])]
ncol(data_num)
```

```
## [1] 123
```

```r
new_data_test_nums <- data_test_nums[,colnames(new_data_num)]
new_data_test_nums <- cbind(new_data_test_nums,testing$classe)
ncol(data_test_nums[,colnames(new_data_num)])
```

```
## [1] 48
```

```r
new_data_num  <- cbind(new_data_num ,training$classe)
colnames(new_data_num)[ncol(new_data_num)] <- "classe"
ncol(data_test_nums)
```

```
## [1] 123
```

```r
data_num[13000,123]
```

```
## [1] 19
```

```r
preproc <- preProcess(new_data_num[,-ncol(new_data_num)],method="pca",pcaComp=5)
trainPC <- predict(preproc,new_data_num[,-ncol(new_data_num)])
trainPC <- cbind(trainPC,training$classe)
colnames(trainPC)[ncol(trainPC)]<-("classe")
library(e1071)
```

```
## Warning: package 'e1071' was built under R version 3.1.1
```

```r
modelfit <- svm(training$classe~.,kernel = "linear", C=0.1, gamma=1, cross=10, type="C-classification", data=trainPC)
testPC <- predict(preproc,new_data_test_nums[,-ncol(new_data_test_nums)])
testPC <- cbind(testPC,testing$classe)
colnames(testPC)[ncol(testPC)]<-("classe")
confusionMatrix(testing$classe,predict(modelfit,newdata=testPC))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1381   10    0    1    3
##          B  935   10    0    1    3
##          C  850    5    0    0    0
##          D  794    7    0    1    2
##          E  891    2    0    0    8
## 
## Overall Statistics
##                                         
##                Accuracy : 0.285         
##                  95% CI : (0.273, 0.298)
##     No Information Rate : 0.989         
##     P-Value [Acc > NIR] : 1             
##                                         
##                   Kappa : 0.003         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.2847  0.29412       NA 0.333333  0.50000
## Specificity            0.7358  0.80719    0.826 0.836156  0.81731
## Pos Pred Value         0.9900  0.01054       NA 0.001244  0.00888
## Neg Pred Value         0.0111  0.99393       NA 0.999512  0.99800
## Prevalence             0.9892  0.00693    0.000 0.000612  0.00326
## Detection Rate         0.2816  0.00204    0.000 0.000204  0.00163
## Detection Prevalence   0.2845  0.19352    0.174 0.163948  0.18373
## Balanced Accuracy      0.5103  0.55065       NA 0.584745  0.65865
```


For Other Methods

```r
#tc <- trainControl("cv",10,savePred=T)
#fit <- train(classe~.,data=new_data_num,method="rpart",trControl=tc,family=poisson(link = "log"))
```
