normlize <- function(x) { 
	return ((x - min(x)) / (max(x) - min(x)) ) 
}
wdbc = read.csv("../../dataset/WDBC/WDBC.dat")
wdbc <- wdbc[2:32]
wdbc_n <- as.data.frame(lapply(wdbc[2:31], normlize))
wdbc_train <- wdbc_n[1:469, ]
wdbc_test <- wdbc_n[470:568, ]
wdbc_train_labels <- wdbc_n[1:469, 1]
wdbc_test_labels <- wdbc_n[470:568, 1]
library("class")
wdbc_pred <- knn(train=wdbc_train, test=wdbc_test, cl=wdbc_train_labels, k=3)
wdbc_test_pred <- knn(train=wdbc_train, test=wdbc_test, cl=wdbc_train_labels, k=21)
library("gmodels")
CrossTable(x=wdbc_test_labels, y=wdbc_test_pred, prop.chisq=FALSE)
