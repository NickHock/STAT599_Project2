# Names of all folders in training data
folder.names <- list.files("/home/dane/Documents/ST599_Data/train",
                           full.names = T)
layer.names <- list.files("/home/dane/Documents/ST599_Data/train",
                          full.names = F)

# Number of folders 
j <- length(folder.names)

Ocean.train <- array(data = NA, dim = c(6, 2048, j), dimnames = 
                       c(NULL, NULL, layer.names[1:j]))

for (i in 1:j){
  # Obtain list of file names in each folder
  file.names <- list.files(folder.names[i], full.names = T)
  
  # Store number of files in each folder
  # k <- length(file.names[i])
  
  # For loop that will grab image data and place it in Ocean.train array
  for (l in 1:6){
    
    # Grab name of first file in folder
    file <- file.names[l]
    
    # Store that file as 'img'
    img <- readJPEG(file)
    
    # Extract information 
    feat <- image_features(img)
    feat.vec <- c(rep(NA, 2048))
    for (p in 1:2048){
      feat.vec[p] <- feat[[p]]
      # Store information in Ocean.train array
      Ocean.train[l, , i] <- feat.vec
    } 
  }
}

# Test Data

Ocean.test <- array(data = NA, dim = c(6, 2048, j), dimnames = 
                      c(NULL, NULL, layer.names[1:j]))
prt = proc.time()
for (i in 1:j){
  # Obtain list of file names in each folder
  file.names <- list.files(folder.names[i], full.names = T)
  
  # Store number of files in each folder
  # k <- length(file.names[i])
  
  # For loop that will grab image data and place it in Ocean.test array
  for (l in 1:6){
    
    # Grab name of first file in folder
    file <- file.names[l+6]
    
    # Store that file as 'img'
    img <- readJPEG(file)
    
    # Extract information 
    feat <- image_features(img)
    feat.vec <- c(rep(NA, 2048))
    for (p in 1:2048){
      feat.vec[p] <- feat[[p]]
      # Store information in Ocean.test array
      Ocean.test[l, , i] <- feat.vec
    } 
  }
}

data.time <- proc.time() - prt

# The data needs to be stored as a matrix; not an array.
train = matrix(NA, 60, 2048)
train = rbind(Ocean.train[1:6, ,1], Ocean.train[1:6, ,2])
for (i in 3:j){
  train = rbind(train, Ocean.train[1:6, , i])
}

# The test data needs to be stored as a matrix; not an array.
test = rbind(Ocean.test[1:6, ,1], Ocean.test[1:6, ,2])
for (i in 3:j){
  test = rbind(test, Ocean.train[1:6, , i])
}

# Classification names

cl <- factor(c(rep(layer.names[1:j], each = 6)))

# KNN

KN <- knn3Train(train, test, cl, k = 5, prob = T)
summary(KN)
attributes(KN)
str(KN)
table(KN , cl)
cm.knn = confusionMatrix(data = KN, reference = cl)
cm.knn$overall

# File to submit

# Get image data from test folder
test.file.names <- list.files("/home/dane/Documents/ST599_Data/test",
                           full.names = T)
test.image.names <- list.files("/home/dane/Documents/ST599_Data/test",
                          full.names = F)
x <- length(test.file.names)
Plank.test <- matrix(NA, x, 2048)

for (i in 1:1){
  img <- readJPEG(test.file.names[i])
  feat <- image_features(img)
  feat.vec <- c(rep(NA, 2048))
  for (p in 1:2048){
    feat.vec[p] <- feat[[p]]
    # Store information in Ocean.test array
    Plank.test[i,] <- feat.vec
  } 
}

Plank.test[1,] <- image_features(readJPEG(test.file.names[1]))
