#install.packages("readtext")
#install.packages("tm")
library(readtext)
library(stringr)
library(tm)
sms <- readtext("../../dataset/smspam/SMSSpamCollection.txt")
text <- unlist(sms[2])
text1 <- str_remove(text, "Named chr ")
text_list <- unlist(strsplit(text1, "\n"))
l <- length(text_list)
n <- 1
type = c()
text = c()
repeat {
  line <- unlist(strsplit(text_list[n], "\t"))
  type <- c(type, line[1])
  text <- c(text, line[2])
  if (n > l) {
    break
  }
  n <- n + 1
}
sms_raw <- data.frame(type=type, text=text)
sms_raw$type <- factor(sms_raw$type)
sms_corpus <- Corpus(VectorSource(sms_raw$text))
inspect(sms_corpus[1:3])
corpus_clean <- tm_map(sms_corpus, tolower)
corpus_clean <- tm_map(corpus_clean, removeWords, stopwords())
corpus_clean <- tm_map(corpus_clean, removePunctuation)
cospus_clean <- tm_map(corpus_clean, stripWhitespace)
inspect(corpus_clean[1:3])
sms_dtm <- DocumentTermMatrix(corpus_clean)
sms_raw_train <- sms_raw[1:4182, ]
sms_raw_test <- sms_raw[4183:5575, ]
sms_dtm_train <- sms_dtm[1:4182, ]
sms_dtm_test <- sms_dtm[4183:5575, ]
sms_corpus_train <- corpus_clean[1:4182]
sms_corpus_test <- corpus_clean[4183:5575]
prop.table(table(sms_raw_train$type))
prop.table(table(sms_raw_test$type))
#install.packages("wordcloud")
library(wordcloud)
wordcloud(sms_corpus_train, min.freq = 40, random.order = FALSE)
spam <- subset(sms_raw_train, type=="spam")
ham <- subset(sms_raw_train, type="ham")
wordcloud(spam$text, max.words = 40, scale = c(3, 0.5))
wordcloud(ham$text, max.words = 40, scale = c(3, 0.5))
findFreqTerms(sms_dtm_train, 5)
sms_dict <- findFreqTerms(sms_dtm_train, 5)
sms_train <- DocumentTermMatrix(sms_corpus_train, list(dictionary=sms_dict))
sms_test <- DocumentTermMatrix(sms_corpus_test, list(dictionary=sms_dict))
convert_counts <- function(x) {
  x <- ifelse(x > 0, 1, 0)
  x <- factor(x , levels=c(0,1), labels=c("No", "Yes"))
  return (x)
}                                                              
sms_train <- apply(sms_train, MARGIN=2, convert_counts)                                
sms_test <- apply(sms_test, MARGIN=2, convert_counts)                                
#install.packages("e1071")
library(e1071)
sms_classifier <- naiveBayes(sms_train, sms_raw_train$type)
sms_test_pred <- predict(sms_classifier, sms_test)
library(gmodels)
CrossTable(sms_test_pred, sms_raw_test$type, prop.chisq = FALSE, prop.t = FALSE, dnn=c('predicted', 'actual'))
