library(caret)
library(randomForest)
#library(ROCR)
library(pROC)
library(WVPlots)
library(rpart)
library(rpart.plot)

shark.data <- read.csv(file="shark_attack_final_dataset.csv", header=T, na.strings = c("NA",""," "), stringsAsFactors = F)
shark.data$Wd.Direction <- factor(shark.data$Wd.Direction, ordered = FALSE)
shark.data$Attack <- factor(shark.data$Attack, ordered = FALSE)
shark.data$moonphase <- factor(shark.data$moonphase, ordered = FALSE)
#shark.data$DailyAverageWdSpeed <- as.factor(shark.data$DailyAverageWdSpeed)
shark.data$binned.DailyAverageWdSpeed <- factor(shark.data$binned.DailyAverageWdSpeed, ordered = TRUE)

shark.data.small <- shark.data[-sample(which(shark.data$Attack == "No"), 200),]
table(shark.data.small$Attack)
train.rows <- createDataPartition(y= shark.data.small$Attack, p=0.7, list = FALSE)
train.data <- shark.data.small[train.rows,]
table(train.data$Attack)
test.data <- shark.data.small[-train.rows,]
table(test.data$Attack)
str(shark.data.small)

rf_classifier = randomForest(Attack ~ Wd.Direction + moonphase + DailyAverageWdSpeed, data=train.data, ntree=100, mtry=2, importance=TRUE)
rf_classifier
varImpPlot(rf_classifier)
test.data$forest.pred <- predict(rf_classifier, test.data)
confusionMatrix(test.data$Attack, test.data$forest.pred, positive = "Yes")
GainCurvePlot(test.data, "forest.pred", "Attack", "title")
test.data$forest.pred.prob <- predict(rf_classifier, test.data, type= "prob")
auc(test.data$Attack, test.data$forest.pred.prob[,1])
auc(test.data$Attack, test.data$forest.pred.prob[,2])

plot(roc(test.data$Attack, test.data$forest.pred.prob[,1]))

#Note: If you get an error with the glm rerun the code as the random sampling missed a class.
glm.fit <- glm(Attack ~ Wd.Direction + moonphase + DailyAverageWdSpeed_normalized, data = train.data, family = binomial)
glm.fit
test.data$logistic.pred <- predict(glm.fit, newdata = test.data, type = "response")
GainCurvePlot(test.data, "logistic.pred", "Attack", "title")
RMSE(as.numeric(test.data$logistic.pred), as.numeric(test.data$Attack))
test.data$logistic.pred <- ifelse(test.data$logistic.pred >= 0.5, "Yes", "No")
test.data$logistic.pred <- factor(test.data$logistic.pred)
confusionMatrix(test.data$Attack, test.data$logistic.pred, positive = "Yes")
auc(test.data$Attack, as.numeric(test.data$logistic.pred))

plot(roc(test.data$Attack, as.numeric(test.data$logistic.pred)))

folds <- createFolds(shark.data.small$Attack, k = 3, returnTrain = FALSE)
shark.data.small[folds[[2]],]
shark.data.small[-folds[[2]],]
folds[[2]]

rf_classifier1 = randomForest(Attack ~ Wd.Direction + moonphase + DailyAverageWdSpeed, data=shark.data.small[-folds[[1]],], ntree=100, mtry=2, importance=TRUE)
rf_classifier2 = randomForest(Attack ~ Wd.Direction + moonphase + DailyAverageWdSpeed, data=shark.data.small[-folds[[2]],], ntree=100, mtry=2, importance=TRUE)
rf_classifier3 = randomForest(Attack ~ Wd.Direction + moonphase + DailyAverageWdSpeed, data=shark.data.small[-folds[[3]],], ntree=100, mtry=2, importance=TRUE)
forest.pred.prob1 <- predict(rf_classifier1, shark.data.small[folds[[1]],], type= "prob")
forest.pred.prob2 <- predict(rf_classifier2, shark.data.small[folds[[2]],], type= "prob")
forest.pred.prob3 <- predict(rf_classifier3, shark.data.small[folds[[3]],], type= "prob")
temp <- c()
temp[1] <- auc(shark.data.small[folds[[1]],]$Attack, forest.pred.prob1[,1])
temp[2] <- auc(shark.data.small[folds[[2]],]$Attack, forest.pred.prob2[,1])
temp[3] <- auc(shark.data.small[folds[[3]],]$Attack, forest.pred.prob3[,1])
#choice <- which(temp == max(temp))
# choice
# 
# if (choice == 1) {
#   rf_classifier1
#   varImpPlot(rf_classifier1)
#   forest.pred <- predict(rf_classifier1, shark.data.small[folds[[1]],])
#   confusionMatrix(shark.data.small[folds[[1]],]$Attack, forest.pred, positive = "Yes")
#   GainCurvePlot(test.data, "forest.pred", "Attack", "title")
#   test.data$forest.pred.prob <- predict(rf_classifier, test.data, type= "prob")
#   auc(test.data$Attack, test.data$forest.pred.prob[,1])
# }
# 
# cartfit <- rpart(Attack ~ Wd.Direction + moonphase + DailyAverageWdSpeed,
#                  data = train.data,
#                  method = "class")
# rpart.plot(cartfit)
