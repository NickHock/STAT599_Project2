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
  #--- Training Set ---#
  #   #---- Native set to True!
  #   train.moment[,i] = as.vector(momentObj(log(-1*train.data[[i]][]), type="krawt", order=N+1, 0.5)@moments[1:N,1:N])
  #---- Native set to False!
  train.moment[,i] = as.vector(momentObj(train.data[[i]][], type="krawt", order=N, 0.5)@moments[1:N,1:N])
}
train.moment <- t(train.moment)
dim(train.moment)

################################################################
###---Run in the case you want to measure accuracy in the
###---training set! Otherwise, skip ahead to the 
###---Monopoly reference!
## 2a. Random Forest on trainging set
# Random Sample for Training and Validation Sets
z<-sample(1:30336,20000)
# trainging set
t.t<-train.moment[z,]
t.v<-train.moment[-z,]
# Validation set
r.t<-response[z]
r.v<-response[-z]
# Training and Validation Data Frames
train<-data.frame(t.t)
#--- k=1: Random Forest
#--- k=2: Naive Bayes
k=1
if(k==1){
  # Random Forrest #
  krawt.rf.p<-randomForest(x=t.t,y=as.factor(r.t),ntree=500)
  ## 2b. Correct Classification from Prediction Function
  pred.krawt.rf.p<-predict(krawt.rf.p,newdata=t.v)
  tabby.krawt<-table(observed=r.v,predicted=pred.krawt.rf.p)
  sum(diag(tabby.krawt))/sum(tabby.krawt)
  ## 3. Write test .CSV file for Kaggle Submission
  #setwd("/Users/nicholashockensmith/Desktop/Big Data/Project 2")
  #write.csv(cbind(r.v,predict(krawt.rf.p,newdata=t.v,type="prob")),file="krawt.rf.p.csv")
}else{
  # Naive Bayes #
  nbay<-NaiveBayes(t.t,as.factor(r.t))
  ## 2b. Correct Classification from Prediction Function
  pred.nbay<-predict(nbay,newdata=t.v)
  ## 3. Write test .CSV file for Kaggle Submission
  #setwd("/Users/nicholashockensmith/Desktop/Big Data/Project 2")
  #write.csv(pred.nbay,file="nbay.csv")   
}
################################################################