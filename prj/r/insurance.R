insurance <- read.csv('../../dataset/insurance.csv', stringsAsFactors = TRUE)
str(insurance)
print(insurance)
summary(insurance$charges)
hist(insurance$charges)
table(insurance$region)
table(insurance$sex)
table(insurance$smoker)
cor(insurance[c('age', 'bmi', 'children', 'charges')])
pairs(insurance[c('age', 'bmi', 'children', 'charges')])
#install.packages('psych')
library(psych)
pairs.panels(insurance[c('age', 'bmi', 'children', 'charges')])
ins_models <- lm(charges ~ age+children+bmi+sex+smoker+region, data=insurance)
summary(ins_models)
