credit <- read.csv('../../dataset/credit.csv')
table(credit$checking_balance)
table(credit$savings_balance)
summary(credit$months_loan_duration)
summary(credit$amount)
table(credit$default)
set.seed(12345)
credit_rand <- credit[order(runif(1000)),]
summary(credit_rand$amount)
head(credit$amount)
head(credit_rand$amount)
credit_rand$default <- as.factor(credit_rand$default)
credit_train <- credit_rand[1:900,]
credit_test <- credit_rand[901:1000,]
prop.table(table(credit_train$default))
prop.table(table(credit_test$default))
#install.packages("C50")
library(C50)
credit_model <- C5.0(credit_train[-17], credit_train$default)
typeof(credit_train$default)
credit_train$default
credit_model
summary(credit_model)
credit_pred <- predict(credit_model, credit_test)
library(gmodels)
CrossTable(credit_test$,default, credit_pred, prop.chisq=FALSE, prop.c=FALSE, prop.r=FALSE, dnn=c('autual default', 'predicted default'))