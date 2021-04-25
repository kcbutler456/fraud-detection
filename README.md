# Financial Transaction Fraud Detection

One of the most common problems facing financial institutions today is how to detect fraud and money laundering efforts in financial transactions. In 2019, 1.7 million fraud reports were filed with the Consumer Sentinel Network, with a staggering $1.9 billion in consumer losses (Federal Trade Commission, 2020). Financial institutions have detection methods in place currently that aid in identifying fraud in financial transactions including customer risk profiling and alert systems triggered by predefined rules. Machine learning has also been used to classify individual transactions based on transaction information such as amount, transaction type, origination information ect. 

This project seeks to use a binary machine learning classifier to detect fraud in financial transactions. The data used for this project is located on Kaggle. Due to the private nature of financial data, this public data source is synthetic, generated for the purpose of preforming research (Lopez-Rojas et al., 2016). This data is an aggregation of private datasets to provide transaction data the resembles normal operations and injects malicious behaviour to later evaluate the performance of fraud detection methods (Lopez-Rojas et al., 2016).

This is a supervised machine learning task which will utilize decision trees, random forest, xgboost, and support vector machine and compare the results to select the best performing classifier. The data used contains over 6 million observations and 11 variables (step, amount, nameOrig, oldblanceOrg, newbalanceOrig, nameDest, oldblanceDest, newbalanceDest, isFraud, and isFlaggedFraud)



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



## Data Cleaning and Exploratory Data Ananlysis

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



## Feature Selection - Fri



## Decision Tree - Sat



## Random Forest - Sat



## Support Vector Machine - Sat



## XGBoost - Sun




## Results and Conclusion - Sun



## References
- Federal Trade Commission. (2020, January). Consumer Sentinel Network Data Book 2019. https://www.ftc.gov/system/files/documents/reports/consumer-sentinel-network-data-book-2019/consumer_sentinel_network_data_book_2019.pdf. 
- E. A. Lopez-Rojas , A. Elmir, and S. Axelsson. (2016) "PaySim: A financial mobile money simulator for fraud detection". In: The 28th European Modeling and Simulation Symposium-EMSS, Larnaca, Cyprus. https://www.kaggle.com/ntnu-testimon/paysim1
- Lee, J. (2018). Decision Trees in R. DataCamp Community. https://www.datacamp.com/community/tutorials/decision-trees-R. 
- Maklin, C. (2019, July 30). Random Forest In R. Medium. https://towardsdatascience.com/random-forest-in-r-f66adf80ec9. 
- Grogan, M. (2020, October 5). Implementing an XGBoost Model in R. Medium. https://towardsdatascience.com/implementing-an-xgboost-model-in-r-59ee1892be2f. 
- Lee, J. (2018). Support Vector Machines in R. DataCamp Community. https://www.datacamp.com/community/tutorials/support-vector-machines-r. 
