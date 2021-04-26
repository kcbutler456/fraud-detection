library("readr")
library("kaggler")
library("ggplot2")
library("e1071")
library("caTools")
library("dplyr")
library("ISLR")
library("tree")
library("caret")
library("randomForest")


#Download Data
setwd()
kgl_auth(creds_file = 'kaggle.json')
response <- kgl_datasets_download_all(owner_dataset = "ntnu-testimon/paysim1")

download.file(response[["url"]], "application.zip", mode="wb")
unzip_result <- unzip("application.zip", overwrite = TRUE)
fraud <- read_csv("PS_20174392719_1491204439457_log.csv")


#Data cleaning and EDA
data <- data.frame(fraud) #Make a copy of the data
data$isFlaggedFraud <- as.factor(data$isFlaggedFraud)
data$isFraud <- as.factor(data$isFraud)
data$step <- as.factor(data$step)
data$type <- as.factor(data$type)

nrow(data) #6,362,620 rows
str(data) #Check data types
head(data) #check top rows
tail(data) #check bottom rows
data[is.na(data), ] #check for any null values
summary(data) #evaluate data spread
 #isFraud is unbalanced with only 8,213 rows fraud < .13% of the data
 #isFraud is unbalanced with only 16 rows flagged as fraud


plot(data[data$isFraud=='1',"type"], main="Total Fraud Records by Type" ) #all fraud is happening in cash_out & transfer types
data <- data[data$type=="CASH_OUT"|data$type=="TRANSFER",] #cash out and transfer types only
nrow(data)#2,770,409

data[data$isFraud=='1',]%>%
  group_by(type, isFraud)%>%
  count()%>%
  data.frame()

data[,c("amount", "isFraud")] %>%
  group_by(isFraud)%>%
  summarise(avg = mean((amount)))

data%>%
  group_by(type, isFraud)%>%
  count()%>%
  data.frame()%>%
  ggplot( aes(fill=isFraud, y=n, x=type)) + 
  geom_bar(position="stack", stat="identity")



#Feature selection
set.seed(124)
control <- rfeControl(functions = rfFuncs, method = "cv", number = 10)
results <- rfe(data[,c(2,3,5,6,8,9)], data[,"isFraud"], sizes = c(1:6), rfeControl = control, ntree = 10)
results
predictors(results)
plot(results, type=c("g", "o"))


#Data Preparation of Model
set.seed(124)
dt <- data
dt <- dt[,c(2,3,5,6,8,9,10)]

split = sample.split(dt$isFraud, SplitRatio = 0.75)
train <- subset(dt, split == TRUE)
test <- subset(dt, split == FALSE)




#Decision Tree Classifier
tree.fraud <- tree(isFraud~., data=train)
plot(tree.fraud)
text(tree.fraud, pretty = 0)
pred <- predict(tree.fraud, test, type = "class")
confusionMatrix(pred, test$isFraud, positive = "1")


 
#Random Forest Classifier
set.seed(124)
rf <- randomForest(isFraud~., data = train, ntree = 10)
pred = predict(rf, newdata=test[-7])
confusionMatrix(pred, test$isFraud, positive = "1")



set.seed(124)
rf <- randomForest(isFraud~., data = train, ntree = 20)
pred = predict(rf, newdata=test[-7])
confusionMatrix(pred, test$isFraud, positive = "1")



set.seed(124)
rf <- randomForest(isFraud~., data = train, ntree = 30)
pred = predict(rf, newdata=test[-7])
confusionMatrix(pred, test$isFraud, positive = "1")



#SVM

#linear
set.seed(124)
classifier <- svm(formula = isFraud~.,
                  data = train,
                  type = 'C-classification', 
                  kernel = 'linear')
pred <- predict(classifier, test[-7])
confusionMatrix(pred, test$isFraud, positive = "1")




#polynomial
set.seed(124)
classifier <- svm(formula = isFraud~.,
                  data = train,
                  type = 'C-classification', 
                  kernel = 'polynomial')
pred <- predict(classifier, test[-7])
confusionMatrix(pred, test$isFraud, positive = "1")



#radial
set.seed(124)
classifier <- svm(formula = isFraud~.,
                  data = train,
                  type = 'C-classification', 
                  kernel = 'radial')
pred <- predict(classifier, test[-7])
confusionMatrix(pred, test$isFraud, positive = "1")