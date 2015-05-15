# Binarizing an image with k-means clsutering 
# and extracting bin count. 

library(jpeg)

main_dir <- "C:/Users/kevar_000/Dropbox/Oregon State University/ST 599 - Big Data/Project2"
## First import the data 
setwd("C:/Users/kevar_000/Dropbox/Oregon State University/ST 599 - Big Data/Project2/train")
train.class.folders = substring(list.dirs(),3)[-1]

# train_dir <- "C:/Users/kevar_000/Dropbox/Oregon State University/ST 599 - Big Data/Project2/train/";
test_dir <-"C:/Users/kevar_000/Dropbox/Oregon State University/ST 599 - Big Data/Project2/test";

for (folder_num in 1:num_of_folders){

      data.file.names = list.files(pattern=".jpg")
      ## We choose native = FALSE because we want the grayscale to be between 0 and 1 that is on a normalized scale 
      num_images <- length(data.file.names) ## number of images 
      
      bin_count <-data.frame(bin1=0,bin2=0,bin3=0,bin4=0,bin5=0,bin6=0,bin7=0,bin8=0,bin9=0,bin10=0);
      pb <- winProgressBar(title="Extraction Progress", label="0% done", min=0, max=100, initial=0)
      prt <- proc.time()
      for (i in 1:num_images)
      {

            df_tmp <- readJPEG(data.file.names[i],native=FALSE)
            # Show the Original image 
            # image(df_tmp,col=grey(seq(0, 1, length =256)))
            # Here is the size of our matrix  
            num_rows <- dim(df_tmp)[1];  
            num_cols <- dim(df_tmp)[2];
            tmp <- as.vector(t(df_tmp));
            # Vectorize the matrix and get all the values that are != 1 (white)
            location <- which(tmp!=1);
            dist_values <- tmp[location];
            tmp <-t(hist(dist_values,plot=FALSE,breaks=seq(0,1,by=1/10))$counts); 
            bin_count[i,] <- tmp
            info<-sprintf("%d%% done",round((i/num_images)*100))
            setWinProgressBar(pb,i/(num_images)*100,label=info)
      }
      close(pb)
      processing_time<-proc.time() - prt
      processing_time   
      # user   system  elapsed 
      # 9214.99    17.89 10005.95 
      # names(bin_count)<- paste("bin",1:10,sep="")
      bin_count$images <-data.file.names
       save_to <- paste(main_dir,"/","bincount_test",".txt",sep="")
      
       write.csv(bin_count,file=save_to,row.names=FALSE)
      
print(paste(train.class.folders[folder_num]," SAVED"))
}
########################################################################################################
# Import bin count for test data set this shit took like 130 minutes to compute 
test <- read.table("bincount_test.txt",header=TRUE,sep=",")

#######################################################################################################
## import training data set and combine
main_dir <- "C:/Users/kevar_000/Dropbox/Oregon State University/ST 599 - Big Data/Project2"


  # Takes the file location of the txt file we created from the extracted features 
  file_location <- paste(loc_bin_dir,file_names,sep="/")
  ## Here finds the txt file containing the correspond image labels 
  
  ##temporarily loading the data set that are delimmited by commas
  tmp <- read.table(file_names[file_num],sep=",")
  row.names(tmp)<- NULL #Here we temporarily take in the data set 

setwd(loc_bin_dir)

file_list <- list.files(,pattern=".txt")
## merging all the data sets together 
dataset <- do.call("rbind",lapply(file_list,FUN=function(files){read.table(files,header=TRUE, sep=",")}))
save_train <- paste(main_dir,"/","bincount_train.txt",sep="")
write.csv(dataset,file=save_train,row.names=FALSE)


  
