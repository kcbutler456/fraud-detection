# Financial Transaction Fraud Detection

One of the most common problems facing financial institutions today is how to detect fraud and money laundering efforts in financial transactions. In 2019, 1.7 million fraud reports were filed with the Consumer Sentinel Network, with a staggering $1.9 billion in consumer losses (Federal Trade Commission, 2020). Financial institutions have detection methods in place currently that aid in identifying fraud in financial transactions including customer risk profiling and alert systems triggered by predefined rules. Machine learning has also been used to classify individual transactions based on transaction information such as amount, transaction type, origination information ect. 

This project seeks to use a binary machine learning classifier to detect fraud in financial transactions. The data used for this project is located on Kaggle. Due to the private nature of financial data, this public data source is synthetic, generated for the purpose of preforming research (Lopez-Rojas et al., 2016). This data is an aggregation of private datasets to provide transaction data the resembles normal operations and injects malicious behaviour to later evaluate the performance of fraud detection methods (Lopez-Rojas et al., 2016).

This is a supervised machine learning task which will utilize decision trees, random forest, and support vector machine and compare the results to select the best performing classifier. The data used contains over 6 million observations and 11 variables (step, amount, nameOrig, oldblanceOrg, newbalanceOrig, nameDest, oldblanceDest, newbalanceDest, isFraud, and isFlaggedFraud)


## Tools and Resources

- R
- Synthetic Financial Datasets For Fraud Detection (Lopez-Rojas et al., 2016)
- Decision Trees in R (Lee, 2018)
- Random Forest in R (Maklin, 2019)
- Implementing an XGBoost Model in R (Grogan, 2020)
- Support Vector Machines in R (Lee, 2018)


## Data Collection

Data collection was done using the kaggler library which downloads data direction from Kaggle. 

```html
install.packages(c("devtools"))
devtools::install_github("ldurazo/kaggler")

library(kaggler)

setwd()
kgl_auth(creds_file = 'kaggle.json')
response <- kgl_datasets_download_all(owner_dataset = "ntnu-testimon/paysim1")

download.file(response[["url"]], "application.zip", mode="wb")
unzip_result <- unzip("application.zip", overwrite = TRUE)
fraud <- read_csv("PS_20174392719_1491204439457_log.csv")

head(fraud)
```
![image](https://user-images.githubusercontent.com/55027593/116000177-b720ab80-a5b4-11eb-9823-e9a41908361f.png)


## Data Cleaning and Exploratory Data Analysis

This data comes fairly well prepared, so minimal data cleaning was needed. I updated data types and did basic exploratory data anlysis to start. From the EDA phase, we find that this data is heavily unbalanced with fraud instances making up less than .13% of the data (8,213 observations). Upon further investigation, these fraud instances can only be found in cash out and transafer transaction types. Therefore, we can exclude cash in, debit, and payment transaction types to reduce the overall observations and non-fraud transactions. 

```html
data <- data.frame(fraud) #Make a copy of the data

data$isFlaggedFraud <- as.factor(data$isFlaggedFraud)
data$isFraud <- as.factor(data$isFraud)
data$step <- as.factor(data$step)
data$type <- as.factor(data$type)
```

```html
nrow(data)
```
[1] 6362620

```html
str(data)
```
![image](https://user-images.githubusercontent.com/55027593/116000463-f13e7d00-a5b5-11eb-8d05-9f0685140914.png)

```html
data[is.na(data), ] #Check the data for null values
summary(data) #Evaluate the spread

```
![image](https://user-images.githubusercontent.com/55027593/116000505-28149300-a5b6-11eb-958f-8f6a7e62fefe.png)             
 
 
 ```html
plot(data[data$isFraud=='1',"type"], main="Total Fraud Records by Type" ) 
data <- data[data$type=="CASH_OUT"|data$type=="TRANSFER",] 
nrow(data)
```
[1] 2770409

![image](https://user-images.githubusercontent.com/55027593/116000845-78d8bb80-a5b7-11eb-9ad8-ceb69e80d4df.png)

```html
data%>%
  group_by(type, isFraud)%>%
  count()%>%
  data.frame()%>%
  ggplot( aes(fill=isFraud, y=n, x=type)) + 
  geom_bar(position="stack", stat="identity")
```
![image](https://user-images.githubusercontent.com/55027593/116000901-ab82b400-a5b7-11eb-8c4b-59a8c9ddb88e.png)


## Feature Selection
A recursive feature elimination method was used to evaluate subsets of attributes to select the subset that produces the highest accuracy. Decision trees were produced at various subsets. The model determined all 6 variables (newbalanceDest, oldbalanceOrg, newbalanceOrig, oldbalanceDest, type, and amount) produces the best results. We proceed with these variables

```html
control <- rfeControl(functions = rfFuncs, method = "cv", number = 10)
results <- rfe(data[,c(2,3,5,6,8,9)], data[,"isFraud"], sizes = c(1:6), rfeControl = control, ntree = 10)
results

predictors(results)
plot(results, type=c("g", "o"))
```

![image](https://user-images.githubusercontent.com/55027593/116008479-6459ea80-a5da-11eb-9712-2b261a2a8efe.png)
![image](https://user-images.githubusercontent.com/55027593/116005512-e4795380-a5cc-11eb-885d-916093786904.png)


## Data Preparation for Models
We now split the data into train and testing datasets for the machine learning phase. We use a 75/25 split. 

```html
set.seed(124)
dt <- data #make a copy of clean data
dt <- dt[,c(2,3,5,6,8,9,10)] #select 6 variables identified in feature selection

split = sample.split(dt$isFraud, SplitRatio = 0.75)
train <- subset(dt, split == TRUE)
test <- subset(dt, split == FALSE)
```

## Decision Tree

```html
tree.fraud <- tree(isFraud~., data=train)
plot(tree.fraud)
text(tree.fraud, pretty = 0)
```
![image](https://user-images.githubusercontent.com/55027593/116008720-8d2eaf80-a5db-11eb-8c90-3c50144e6b74.png)


```html
pred <- predict(tree.fraud, test, type = "class")
confusionMatrix(pred, test$isFraud, positive = "1")
```

![image](https://user-images.githubusercontent.com/55027593/116008747-b3ece600-a5db-11eb-89f8-ad804d31d378.png)

## Random Forest


10 Trees
```html
set.seed(124)
rf <- randomForest(isFraud~., data = train, ntree = 10)
confusionMatrix(pred, test$isFraud, positive = "1")
```
![image](https://user-images.githubusercontent.com/55027593/116008920-6329bd00-a5dc-11eb-9ef8-3b0c5fb912bf.png)

20 Trees
```html
set.seed(124)
rf <- randomForest(isFraud~., data = train, ntree = 20)
confusionMatrix(pred, test$isFraud, positive = "1")
```
![image](https://user-images.githubusercontent.com/55027593/116009059-07abff00-a5dd-11eb-9db1-3b24e98aed03.png)

30 Trees
```html
set.seed(124)
rf <- randomForest(isFraud~., data = train, ntree = 30)
confusionMatrix(pred, test$isFraud, positive = "1")
```

![image](https://user-images.githubusercontent.com/55027593/116009259-1cd55d80-a5de-11eb-96a7-74e565f8993f.png)


## Support Vector Machine 

Linear SVM
```html
set.seed(124)
classifier <- svm(formula = isFraud~.,
                  data = train,
                  type = 'C-classification', 
                  kernel = 'linear')
pred <- predict(classifier, test[-7])
confusionMatrix(pred, test$isFraud, positive = "1")
```
![image](https://user-images.githubusercontent.com/55027593/116011032-5ca14280-a5e8-11eb-9671-50301e7f2adf.png)



polynomial SVM

```html
set.seed(124)
classifier <- svm(formula = isFraud~.,
                  data = train,
                  type = 'C-classification', 
                  kernel = 'polynomial')
pred <- predict(classifier, test[-7])
confusionMatrix(pred, test$isFraud, positive = "1")
```
![image](https://user-images.githubusercontent.com/55027593/116012421-f91b1300-a5ef-11eb-9356-a43b74011834.png)

radial SVM
```html
set.seed(124)
classifier <- svm(formula = isFraud~.,
                  data = train,
                  type = 'C-classification', 
                  kernel = 'radial')
pred <- predict(classifier, test[-7])
confusionMatrix(pred, test$isFraud, positive = "1")
```
![image](https://user-images.githubusercontent.com/55027593/116015355-849ba080-a5fe-11eb-9ce5-33e537452054.png)

## Results and Conclusion

Due to the severely imbalanced nature of the classes, with the rare class being fraud, we will keep an eye on the balanced accuracy results from the confusion matrix output instead of the standard accuracy metric. This balanced accuracy takes into account prevalence of the respective classes. Additionally, we will use kappa to measure the agreement between the predicted and test data. Using the same six variables across each model and comparing them side by side, we can quickly rule out support vector machine for this data and machine learning task. It performed the worst with balanced accuracy ranging from .67 to .76. Additionally, the rate of false negatives was too high to deem acceptable for this business task. This leaves the decision tree classifier and the random forest classifier (ensemble learning method). Due to machine memory limitations, I was limited in the number of trees I could run. Therefore, I decided to compare the results of the random forest at 10, 20, and 30 tree ensembles. Of these, the decision tree classifier with 20 trees performed the best with a balanced accuracy of .86 and a kappa value of .83. Comparing this to our decision tree confusion matrix output, we can see the random forest outperformed, with the decision tree balance accuracy of .85 and a kappa value of .81. Additionally, the false positive rate improved slightly with the random forest classifier.

With all of the confusion matrix output considered, we can conclude the random forest classifier is the best model for this method. Further tuning and hyper parameterization can also be considered to improve this model further. Additionally, in a real business scenario, we would have access to much more predictive variables. For example, we may include information about the customer or their occupation. This type of data may lend itself to higher accuracy in predicting fraud data in financial transactions.


## References
- Federal Trade Commission. (2020, January). Consumer Sentinel Network Data Book 2019. https://www.ftc.gov/system/files/documents/reports/consumer-sentinel-network-data-book-2019/consumer_sentinel_network_data_book_2019.pdf. 
- E. A. Lopez-Rojas , A. Elmir, and S. Axelsson. (2016) "PaySim: A financial mobile money simulator for fraud detection". In: The 28th European Modeling and Simulation Symposium-EMSS, Larnaca, Cyprus. https://www.kaggle.com/ntnu-testimon/paysim1
- Lee, J. (2018). Decision Trees in R. DataCamp Community. https://www.datacamp.com/community/tutorials/decision-trees-R. 
- Maklin, C. (2019, July 30). Random Forest In R. Medium. https://towardsdatascience.com/random-forest-in-r-f66adf80ec9. 
- Grogan, M. (2020, October 5). Implementing an XGBoost Model in R. Medium. https://towardsdatascience.com/implementing-an-xgboost-model-in-r-59ee1892be2f. 
- Lee, J. (2018). Support Vector Machines in R. DataCamp Community. https://www.datacamp.com/community/tutorials/support-vector-machines-r. 
- Zablotski, Yury. (2021). R demo | Deep Exploratory Data Analysis (EDA) | explore your data and start to test hypotheses. https://www.youtube.com/watch?v=Swcp0_l65lw.
- Simplilearn. (2020). Feature Selection In Machine Learning | Feature Selection Techniques With Examples. https://www.youtube.com/watch?v=5bHpPQ6_OU4
- Simplilearn. (2018). Random Forest Algorithm - Random Forest Explained | Random Forest in Machine Learning. https://www.youtube.com/watch?v=eM4uJ6XGnSM&t=639s 
- Simplilearn. (2018). Decision Tree In Machine Learning | Decision Tree Algorithm In Python |Machine Learning. https://www.youtube.com/watch?v=RmajweUFKvM
- Brownlee, J. (2014). Feautre Selection with the Caret R Package. https://machinelearningmastery.com/feature-selection-with-the-caret-r-package/
- Simplilearn. (2020). Ensemble Learning | Ensemble Learning In Machine Learning | Machine Learning Tutorial. https://www.youtube.com/watch?v=WtWxOhhZWX0
- StatQuest. (2018). ROC and AUC in R. https://www.youtube.com/watch?v=qcvAqAH60Yw
