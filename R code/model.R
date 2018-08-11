
library(DataExplorer)
library(mice)
library(VIM)
library("rpart"); library("rpart.plot"); library("C50")


shark.data <- read.csv(file="shark_attack_final_dataset.csv", header=T, na.strings = c("NA",""," "), stringsAsFactors = F)
attach(shark.data)

#shark.data.AttackNo <- shark.data[shark.data$Attack == "No"]

#df.No = shark.data.AttackNo.frame(matrix(rnorm(20), nrow=10))



fit <- rpart(shark.data$Attack ~ shark.data$Wd.Direction + shark.data$moonphase+ shark.data$County
             +shark.data$binned.DailyAverageDewPotTemp+shark.data$binned.DailyAverageWdSpeed
             +shark.data$binned.DailyPrecipitation,
             method="class", data=shark.data)

print(fit)
# Plot the decision tree
rpart.plot(fit)


printcp(fit) # display the results 
plotcp(fit) # visualize cross-validation results 
summary(fit) # detailed summary of splits

# plot tree 
plot(fit, uniform=TRUE, 
     main="Classification Tree for Kyphosis")
text(fit, use.n=TRUE, all=TRUE, cex=.8)
