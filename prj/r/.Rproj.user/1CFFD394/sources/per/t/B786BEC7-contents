mushrooms <- read.csv('../../dataset/mushrooms.csv', stringsAsFactors = TRUE)
typeof(mushrooms)
mushrooms$veil_type
mushrooms$veil_type <- NULL
table(mushrooms$type)
summary(mushrooms)
length(mushrooms)
library(RWeka)
install.packages('RWeka')
mushrooms_1R <- OneR(type ~ ., data=mushrooms)
mushrooms_1R
summary(mushrooms_1R)
length(mushrooms$type)
length(mushrooms$type) * 0.9
mushrooms_train <- mushrooms[1:7312,]
mushrooms_test <- mushrooms[7313:8142,]
mushrooms_1R1 <- OneR(type ~ ., data=mushrooms_train)
summary(mushrooms_1R1)
mushrooms_pred <- predit(mushrooms_1R1, mushrooms_test)
library(gmodels)
CrossTable(mushrooms_test$type, mushrooms_pred, 
           prop.chisq=FALSE, prop.c=FALSE, prop.r=FALSE, 
           dnn=c('autual type', 'predicted type'))

mushrooms_JRip <- JRip(type ~ ., data=mushrooms)
summary(mushrooms_JRip)