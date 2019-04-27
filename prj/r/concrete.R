concrete <- read.csv('../../dataset/concrete.csv')
str(concrete)
summary(concrete)
length(concrete$slag)
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
concrete_norm <- as.data.frame(lapply(concrete, normalize))
str(concrete_norm)
summary(concrete_norm$slag)
#install.packages("neuralnet")
library(neuralnet)
concrete_train <- concrete_norm[1:773, ]
concrete_test <- concrete_norm[774:1030, ]
concrete_mopdel <- neuralnet(strength ~ cement + slag +
                               ash + water + superplastic +
                               coarseagg + fineagg + age,
                             data = concrete_train)
str(concrete_mopdel)
plot(concrete_mopdel)
results <- compute(concrete_mopdel, concrete_test[1:8])
pred <- results$net.result
cor(pred, concrete_test$strength)
