? iris3


# Make training data out of first 25 rows
train1 <- rbind(iris3[1:25,,1], iris3[1:25,,2], iris3[1:25,,3])

# Make test data out of second 25 rows
test1 <- rbind(iris3[26:50,,1], iris3[26:50,,2], iris3[26:50,,3])

# True classifications of training set
# s - Setosa
# c - Versicolor
# v - Virginica

cl1 <- factor(c(rep("s",25), rep("c",25), rep("v",25)))


# K-nearest neighbors
KN2 <- knn3Train(train1, test1, cl1, k = 5, prob = T)

predict(KN2, type = "prob")

attributes(KN2)
cm.cart1 = confusionMatrix(data = KN2, reference = cl1)
cm.cart1$overall
table(KN2, cl1)
str(KN2)

# dim gives (row, column, layer)
array (data = NA, dim = c(4, 5, 3))

# test building data frame
# initialize empty data frame
names <- c("A", "B", "C")
Ocean.train <- array(data = NA, dim = c(6, 2048, 10), dimnames = 
                       c(NULL, NULL, layer.names[1:10]))


# Names of all folders in training data
folder.names <- list.files("/home/dane/Documents/ST599_Data/train",
                             full.names = T)
layer.names <- list.files("/home/dane/Documents/ST599_Data/train",
                          full.names = F)

# Number of folders 
j <- length(folder.names)
j <- 10
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

# KNN

# Classification names

cl <- factor(c(rep(layer.names[1:10], each = 6)))

# Test Data

Ocean.test <- array(data = NA, dim = c(6, 2048, 10), dimnames = 
                      c(NULL, NULL, layer.names[1:10]))

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


train = matrix(NA, 60, 2048)
train = rbind(Ocean.train[1:6, ,1], Ocean.train[1:6, ,2])
for (i in 3:10){
train = rbind(train, Ocean.train[1:6, , i])
}


test = rbind(Ocean.test[1:6, ,1], Ocean.test[1:6, ,2])
for (i in 3:10){
  test = rbind(test, Ocean.train[1:6, , i])
}
KN <- knn3Train(train, test, cl, k = 4, prob = T)

table(KN , cl)
cm.knn = confusionMatrix(data = KN, reference = cl)
cm.knn$byClass
