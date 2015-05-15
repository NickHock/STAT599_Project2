
## Set the general path of the project
# data_dir <- "C:/Users/kevar_000/Dropbox/Oregon State University/ST 599 - Big Data/Project2"
data_dir <- "Z:/ST 599 - Big Data/Project2"
setwd(data_dir)
## Directories that contain the train labels with their respective processed data
## from the MatLab
train_data_dir <- paste(data_dir,"/Train_labels", sep="")
train_processed_dir <- paste(data_dir,"/Train_Processed", sep="")
## setting the directory where all the processed files are
## and getting all the file names 
setwd(train_processed_dir)
file_names <- list.files(pattern =".txt") 

## Here we find the gabor filter extracted data 
file_location <- paste(train_processed_dir,file_names[1],sep="/")
## Here finds the txt file containing the correspond image labels 
image_location <-paste(gsub(".txt","",file_names[1]),"_image_labels.txt",sep="")

##temporarily loading the data set that are delimmited by commas
tmp <- read.table(file_names[1],sep=",") #Here we temporarily take in the data set 
image_labels <- read.table(image_location,sep="\n",colClasses="character")
## Clean it up a little more 
tmp$name <- rep(gsub(".txt","",file_names[1]),dim(tmp)[1])

multmerge(train_processed_dir)

## Set working directory
## select a folder, 
# k=3
 class_folder <- paste(train_data_dir,"/",train_class_dir[k],sep="")
 setwd(class_folder)
 ## taking the image names of the folder 
# need pattern = ".jpg" otherwise we got a db file which is not a jpg. 
 image_names <- list.files(pattern =".jpg") 
# i=1
# image_sd3 <- c()
# for (names in image_names){
# img <- readJPEG(names,native=TRUE)
# image_sd3[i] <- sd(img)
# i = i + 1
# }
# 
# k=2
# class_folder <- paste(train_data_dir,"/",train_class_dir[k],sep="")
# setwd(class_folder)
# ## taking the image names of the folder 
# # need pattern = ".jpg" otherwise we got a db file which is not a jpg. 
# image_names <- list.files(pattern =".jpg") 
# i=1
# image_sd2 <- c()
# for (names in image_names){
#   img <- readJPEG(names,native=TRUE)
#   image_sd2[i] <- sd(img)
#   i = i + 1
# }

## ==============================
## Define Functions
## ==============================

## Handy function to display a greyscale image of the plankton
im <- function(image) image(image, col = grey.colors(32))

## Function to extract some simple statistics about the image
extract_stats <- function(working_image = working_image){
    #Invert the image to calculate density
    image <- working_image < mean(working_image)
    im_length <- nrow(image) #
    im_width <- ncol(image) # The length and width of the image is a bias way. 
    im_density <- mean(image)
    im_ratio <- im_length / im_width 
    return(c(length=im_length,width=im_width,density=im_density,ratio=im_ratio))    
}

## Function to calculate multi-class loss on train data
## this is the measure they use on the actual kaggle data  
mcloss <- function (y_actual, y_pred) {
    dat <- rep(0, length(y_actual))
    for(i in 1:length(y_actual)){
        dat_x <- y_pred[i,y_actual[i]]
        dat[i] <- min(1-1e-15,max(1e-15,dat_x))
    }
    return(-sum(log(dat))/length(y_actual))
}

## ==============================
## Read training data
## ==============================
## Create empty data structure to store summary statistics from each image file
train_data <- data.frame(class = character(), filename = character(),length=numeric(),width=numeric(),density=numeric(),ratio=numeric(), stringsAsFactors = FALSE)


## Get list of classes
classes <- list.dirs(train_data_dir, full.names = FALSE)
classes <- setdiff(classes,"")

## Read all the image files and calculate training statistics

for(classID in classes){
    # Get list of all the examples of this classID
    train_file_list <- list.files(paste(train_data_dir,"/",classID,sep=""))
    train_cnt <- length(train_file_list)
    working_data <- data.frame(class = rep("a",train_cnt), filename = "a",lenght=0,width=0,density=0,ratio=0, stringsAsFactors = FALSE)
    idx <- 1
    #Read and process each image
    for(fileID in train_file_list){
        working_file <- paste(train_data_dir,"/",classID,"/",fileID,sep="")
        working_image <- readJPEG(working_file)
        
        # Calculate model statistics
        
        ## YOUR CODE HERE ##
        working_stats <- extract_stats(working_image)
        working_summary <- array(c(classID,fileID,working_stats))
        working_data[idx,] <- working_summary
        idx <- idx + 1
    }
    train_data <- rbind(train_data,working_data)
    cat("Finished processing",classID, '\n')
}




## ==============================
## Create Model
## ==============================

## We need to convert class to a factor for randomForest
## so we might as well get subsets of x and y data for easy model building
y_dat <- as.factor(train_data$class)
x_dat <- train_data[,3:6]
num_trees <- 500

plankton_model <- randomForest(y = y_dat, x = x_dat, ntree = num_trees)



# Compare importance of the variables
importance(plankton_model)


## Check overall accuracy... 24%, not very good but not bad for a simplistic model
table(plankton_model$predicted==y_dat)
#  FALSE  TRUE 
#  22959  7377

## Make predictions and calculate log loss
y_predictions <- predict(plankton_model, type="prob")

ymin <- 1/1000
y_predictions[y_predictions<ymin] <- ymin

mcloss(y_actual = y_dat, y_pred = y_predictions)
# 3.362268


## ==============================
## Read test data and make predictions
## ==============================



## Read all the image files and calculate training statistics
## This should take about 10 minutes, with speed limited by IO of the thousands of files

    # Get list of all the examples of this classID
    test_file_list <- list.files(paste(test_data_dir,sep=""))
    test_cnt <- length(test_file_list)
    test_data <- data.frame(image = rep("a",test_cnt), lenght=0,width=0,density=0,ratio=0, stringsAsFactors = FALSE)
    idx <- 1
    #Read and process each image
    for(fileID in test_file_list){
        working_file <- paste(test_data_dir,"/",fileID,sep="")
        working_image <- readJPEG(working_file)
        
        # Calculate model statistics
        
        ## YOUR CODE HERE ##
        working_stats <- extract_stats(working_image)
        working_summary <- array(c(fileID,working_stats))
        test_data[idx,] <- working_summary
        idx <- idx + 1
        if(idx %% 10000 == 0) cat('Finished processing', idx, 'of', test_cnt, 'test images', '\n')
    }


## Make predictions with class probabilities
test_pred <- predict(plankton_model, test_data, type="prob")
test_pred[test_pred<ymin] <- ymin

## ==============================
## Save Submission File
## ==============================

## Combine image filename and class predictions, then save as csv
submission <- cbind(image = test_data$image, test_pred)
submission_filename <- "results/submission.csv"
write.csv(submission, submission_filename, row.names = FALSE)
cat("saved submission file", '\n')
