letter <- read.csv('../../dataset/letterdata.csv')
str(letter)
summary(letter)
letter_train <- letter[1:16000, ]
letter_test <- letter[16001:20000, ]
#install.packages("kernlab")
library(kernlab)
letter_classifier <- ksvm(letter ~ ., 
                          data=letter_train, 
                          kernel="vanilladot")
letter_classifier <- ksvm(letter ~ ., 
                          data=letter_train, 
                          kernel="rbfdot")

#str(letter_classifier)
letter_classifier
letter_pred <- predict(letter_classifier, letter_test)
head(letter_pred)
table(letter_pred, letter_test$letter)
agreement <- letter_pred == letter_test$letter
table(agreement)
prop.table(table(agreement))
