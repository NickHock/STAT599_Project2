library(jpeg)
library(IM)
library(randomForest)
library(sampling)
library(klaR)

################################################################
### Set working directory
setwd("/Users/nicholashockensmith/Desktop/Big Data/Project 2/train")
train.class.folders = substring(list.dirs(),3)[-1]
train.data = NULL
response = NULL

### Read in Training Data
for(i in 1:length(train.class.folders)) {
  setwd(paste("/Users/nicholashockensmith/Desktop/Big Data/Project 2/train/", train.class.folders[i], sep = ""))
  data.file.names = list.files()
  train.data = c(train.data, sapply(data.file.names, function(x) {readJPEG(x, native=F)}))
  response = c(response, rep(train.class.folders[i], length(data.file.names)))
}

### Calculates Image Moments via the IM package
## See http://cran.r-project.org/web/packages/IM/IM.pdf for 
## alternative moment kernels! 
N<-10
train.moment = matrix(NA,N*N,length(train.data))
for(i in 1:length(train.data)) { 
#   #--- Zeros and Ones ---#
#   temp.train<-log(-1*train.data[[i]][])
#   temp.train[temp.train<1]<-0
#   temp.train[temp.train>1]<-1
#   train.moment[,i] = as.vector(momentObj(temp.train, type="krawt", order=N+1, 0.5)@moments[1:N,1:N])
  #--- Training Set ---#
#   #---- Native set to True!
#   train.moment[,i] = as.vector(momentObj(log(-1*train.data[[i]][]), type="krawt", order=N+1, 0.5)@moments[1:N,1:N])
  #---- Native set to False!
  train.moment[,i] = as.vector(momentObj(train.data[[i]][], type="krawt", order=N, 0.5)@moments[1:N,1:N])
}
train.moment <- t(train.moment)
dim(train.moment)

setwd("/Users/nicholashockensmith/Desktop/Big Data/Project 2/test")
test.data = NULL
data.file.names.test = list.files()
test.data = sapply(data.file.names.test, function(x) {readJPEG(x, native=F)})
#-- ii. More Image Moments via the IM package
test.moment = matrix(NA,N*N,length(test.data))
for(i in 1:length(test.data)) { 
  #   #---- Native set to True!
  #   test.moment[,i] = as.vector(momentObj(log(-1*train.data[[i]][]), type="krawt", order=N+1, 0.5)@moments[1:N,1:N])
  #---- Native set to False!
  test.moment[,i] = as.vector(momentObj(test.data[[i]][], type="krawt", order=N, 0.5)@moments[1:N,1:N])
}
test.moment<-t(test.moment)
dim(test.moment)

### Kevin's Histogram Features! ###
binfeat.train <- read.csv("~/Desktop/Big Data Repository/ST599_Project2/ST599_Project2/BinCountDataSet/bincount_train.txt")
binfeat.test <- read.csv("~/Desktop/Big Data Repository/ST599_Project2/ST599_Project2/BinCountDataSet/bincount_test.txt")
kevfeat.train<-binfeat.train[,-11]
kevfeat.test<-binfeat.test[,-11]
ardvark.train<-cbind(train.moment,kevfeat.train)
ardvark.test<-cbind(test.moment,kevfeat.test)

################################################################ 
###---Build Random Forest model for Kaggle Submission
setwd("/Users/nicholashockensmith/Desktop/Big Data/Project 2")
#--- k=1: Random Forest
#--- k=2: Naive Bayes
#--- k=3: Combined Features
j=3
if(j==1){
  ## 1a. Random Forest on Whole Data Set
  krawt.rf<-randomForest(x=train.moment,y=as.factor(response),ntree=500)
  ## 1b. Correct Classification/ Confusion Matrix/ OOB
  confucius<-krawt.rf$confusion
  sum(diag(confucius))/sum(confucius)
  ## 2. Write the prediction model to .CSV 3. Write test .CSV file for Kaggle Submission
  write.csv(cbind(data.file.names.test,predict(krawt.rf,newdata=test.moment,type="prob")),file="KaggleSubmissionKrawtchoukKojack.csv")
  }else{
    if(j==2){
      ## 1. Naive Bayes on Whole Data Set
      krawt.nbay<-NaiveBayes(train.moment,as.factor(response))
      ## 2. Write the prediction model to .CSV 3. Write test .CSV file for Kaggle Submission
      write.csv(predict(krawt.nbay,newdata=test.moment),file="KaggleSubmissionKrawtchoukNBayFire.csv")
    }else{
      ## 1a. Random Forest on Whole Data Set
      krawt.rf<-randomForest(x=ardvark.train,y=as.factor(response),ntree=500)
      ## 1b. Correct Classification/ Confusion Matrix/ OOB
      confucius<-krawt.rf$confusion
      sum(diag(confucius))/sum(confucius)
      ## 2. Write the prediction model to .CSV 3. Write test .CSV file for Kaggle Submission
      write.csv(cbind(data.file.names.test,predict(krawt.rf,newdata=ardvark.test,type="prob")),file="KaggleSubmissionArdvarkKojack.csv")   
    }
}

################################################################