# Binarizing an image with k-means clsutering 
# and extracting bin count. 

library(jpeg)
main_dir <- "C:/Users/kevar_000/Dropbox/Oregon State University/ST 599 - Big Data/Project2"
## First import the data 
setwd("C:/Users/kevar_000/Dropbox/Oregon State University/ST 599 - Big Data/Project2/train")
train.class.folders = substring(list.dirs(),3)[-1]

train_dir <- "C:/Users/kevar_000/Dropbox/Oregon State University/ST 599 - Big Data/Project2/train";

num_of_folders <- length(train.class.folders)

for (folder_num in 1:num_of_folders){
      #Set working directory of where the folders are located and get file names. 
      setwd(paste(train_dir, train.class.folders[folder_num], sep = "/"))
      data.file.names = list.files(pattern=".jpg")
      ## We choose native = FALSE because we want the grayscale to be between 0 and 1 that is on a normalized scale 
      num_images <- length(data.file.names) ## number of images 
      
      print(paste("Starting to process: ",gsub("_"," ",train.class.folders[folder_num]),sep=" "));
      bin_count <-data.frame(bin1=0,bin2=0,bin3=0,bin4=0,bin5=0,bin6=0,bin7=0,bin8=0,bin9=0,bin10=0);
      for (i in 1:num_images)
      {
            # i=10
            par(mfrow=c(1,2))
            # df <- readJPEG(image,native=FALSE)
            # print(paste("Image name:",data.file.names[i]))
             df_tmp <- readJPEG(data.file.names[i],native=FALSE)
            ## Show the Original image 
            image(df_tmp,col=grey(seq(0, 1, length =255)))
            ## Here is the size of our matrix  
            num_rows <- dim(df_tmp)[1];  
            num_cols <- dim(df_tmp)[2];
            tmp <- as.vector(t(df_tmp));
            # Vectorize the matrix and get all the values that are != 1 (white)
            location <- which(tmp!=1);
            dist_values <- tmp[location];
            hist(dist_values,breaks=seq(0,1,by=1/10),main=train.class.folders[folder_num])
            tmp <-t(hist(dist_values,plot=FALSE,breaks=seq(0,1,by=1/10))$counts); 
            bin_count[i,] <- tmp
            rm(df_tmp,tmp)           
      }
      bin_count$name <-rep(train.class.folders[folder_num],num_images);      
      names(bin_count)<- c(paste("bin",1:10,sep=""),"class")
      
      save_to <- paste(main_dir,"/train_bin/",train.class.folders[folder_num],".txt",sep="")
      write.csv(bin_count,file=save_to,row.names=FALSE)
      # This num_folder = 3 
      # setwd(paste(main_dir,"/train_bin",sep=""))
      # test <- read.table("acantharia_protist.txt",header=TRUE,sep=",")

      rm(bin_count)
      
print(paste(train.class.folders[folder_num]," SAVED"))
}

## import training data set and combine
main_dir <- "C:/Users/kevar_000/Dropbox/Oregon State University/ST 599 - Big Data/Project2"
loc_bin_dir <- paste(main_dir,"/","train_bin",sep="")

setwd(loc_bin_dir)

file_list <- list.files(,pattern=".txt")
## merging all the data sets together 
dataset <- do.call("rbind",lapply(file_list,FUN=function(files){read.table(files,header=TRUE, sep=",")}))


