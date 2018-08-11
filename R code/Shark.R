
library(DataExplorer)
library(mice)
library(VIM)



shark.data <- read.csv(file="SharkAttackData31stJuly2018_V4.csv", header=T, na.strings = c("NA",""," "), stringsAsFactors = F)
attach(shark.data)
#library(lunar)
str(shark.data)
shark.data <- shark.data[,-c(5,7,8,11,12,13,14,15,16,18,19,21,22)]

#shark.data$moonphase <- lunar.phase(as.Date(shark.data$Date),name=8)
summary(shark.data)

plot_missing(shark.data)


shark.data$County[grepl(pattern="Charleston",x=shark.data$Location) ] <- "Charleston"
shark.data$County[grepl(pattern="Carteret",x=shark.data$Location) ] <- "Carteret"
shark.data$County[grepl(pattern= "Horry",x=shark.data$Location)] <- "Horry"
shark.data$County[grepl(pattern="Beaufort",x=shark.data$Location)  ] <- "Beaufort"
shark.data$County[grepl(pattern="Georgetown",x=shark.data$Location) ] <- "Georgetown"
shark.data$County[grepl(pattern="Brunswick",x=shark.data$Location) ] <- "Brunswick"
shark.data$County[grepl(pattern="Currituck",x=shark.data$Location) ] <- "Currituck"
shark.data$County[grepl(pattern="Hyde",x=shark.data$Location) ] <- "Hyde"
shark.data$County[grepl(pattern="Dare",x=shark.data$Location) ] <- "Dare"
shark.data$County[grepl(pattern="Pender",x=shark.data$Location) ] <- "Pender"
shark.data$County[grepl(pattern="New Hanover",x=shark.data$Location) ] <- "New Hanover"
shark.data$County[grepl(pattern="Onslow",x=shark.data$Location) ] <- "Onslow"
shark.data$County[grepl(pattern="Colleton",x=shark.data$Location) ] <- "Colleton"



library(ggplot2)

sharkAtackLoc <- shark.data$County[shark.data$Attack== "Yes"]
sharkAtackLoc
# counts
ggplot(data.frame(sharkAtackLoc), aes(x=sharkAtackLoc)) +
  geom_bar()+theme(axis.text.x = element_text(angle = 90, hjust = 1))

windDirection <- shark.data$Wd.Direction[shark.data$Attack== "Yes"]
windDirection
# counts
ggplot(data.frame(windDirection), aes(x=windDirection)) +
  geom_bar()+theme(axis.text.x = element_text(angle = 90, hjust = 1)) + ggtitle("Wind Direction for Attack='Y'")


ggplot() +
  geom_bar(data = shark.data,
           aes(x = factor(shark.data$Wd.Direction),
               fill = factor(shark.data$Attack)),
           position = "stack") +
  scale_x_discrete("Wind Direction") +
  scale_y_continuous("Count") +
  guides(fill=guide_legend(title="Attack")) +
  scale_fill_manual(values=c("blue", "red")) + ggtitle("Overlayed bar graph for Wind Direction")



##### selecting the records in which attack has happened for EDA
shark.data_attack <- subset(shark.data, shark.data$Attack=="Yes")
plot_bar(shark.data_attack$moonphase)

plot_density(shark.data)



plot(shark.data$DailyPrecipitation, shark.data$DailyAverageTemperature)
plot(shark.data$DailyAverageDewPotTemp, shark.data$DailyPrecipitation)
plot(shark.data$DailyAverageDewPotTemp, shark.data$DailyAverageTemperature)

plot_correlation(shark.data[,c("DailyAverageTemperature", "DailyAverageDewPotTemp")], use = "pairwise.complete.obs")



sapply(shark.data[,c("DailyAverageTemperature", "DailyAverageDewPotTemp", "DailyAverageWdSpeed", "DailyPrecipitation")], function(x) sum(is.na(x)))

shark.data.removedNAs <- shark.data[!is.na(shark.data$DailyAverageTemperature),]




shark.data.removedNAs$zscore.DailyAverageTemperature <- (shark.data.removedNAs$DailyAverageTemperature - mean(shark.data.removedNAs$DailyAverageTemperature))/sd(shark.data.removedNAs$DailyAverageTemperature)
cat("Number of lower range outliers: ", length(shark.data.removedNAs$zscore.DailyAverageTemperature[shark.data.removedNAs$zscore.DailyAverageTemperature < -3]))
cat("Number of upper range outliers: ", length(shark.data.removedNAs$zscore.DailyAverageTemperature[shark.data.removedNAs$zscore.DailyAverageTemperature > 3]))


shark.data.removedNAs<- shark.data[!is.na(shark.data$DailyAverageDewPotTemp),]
shark.data.removedNAs$zscore.DailyAverageDewPotTemp <- (shark.data.removedNAs$DailyAverageDewPotTemp - mean(shark.data.removedNAs$DailyAverageDewPotTemp))/sd(shark.data.removedNAs$DailyAverageDewPotTemp)
cat("Number of lower range outliers: ", length(shark.data.removedNAs$zscore.DailyAverageDewPotTemp[shark.data.removedNAs$zscore.DailyAverageDewPotTemp < -3]))
cat("Number of upper range outliers: ", length(shark.data.removedNAs$zscore.DailyAverageDewPotTemp[shark.data.removedNAs$zscore.DailyAverageDewPotTemp > 3]))


shark.data.removedNAs<- shark.data[!is.na(shark.data$DailyAverageWdSpeed),]

shark.data.removedNAs$zscore.DailyAverageWdSpeed <- (shark.data.removedNAs$DailyAverageWdSpeed - mean(shark.data.removedNAs$DailyAverageWdSpeed))/sd(shark.data.removedNAs$DailyAverageWdSpeed)
cat("Number of lower range outliers: ", length(shark.data.removedNAs$zscore.DailyAverageWdSpeed[shark.data.removedNAs$zscore.DailyAverageWdSpeed < -3]))
cat("Number of upper range outliers: ", length(shark.data.removedNAs$zscore.DailyAverageWdSpeed[shark.data.removedNAs$zscore.DailyAverageWdSpeed > 3]))



shark.data.removedNAs<- shark.data[!is.na(shark.data$DailyPrecipitation),]
shark.data.removedNAs$zscore.DailyPrecipitation <- (shark.data.removedNAs$DailyPrecipitation - mean(shark.data.removedNAs$DailyPrecipitation))/sd(shark.data.removedNAs$DailyPrecipitation)
cat("Number of lower range outliers: ", length(shark.data.removedNAs$zscore.DailyPrecipitation[shark.data.removedNAs$zscore.DailyPrecipitation < -3]))
cat("Number of upper range outliers: ", length(shark.data.removedNAs$zscore.DailyPrecipitation[shark.data.removedNAs$zscore.DailyPrecipitation > 3]))

summary(shark.data.removedNAs[,c("DailyAverageTemperature", "DailyAverageDewPotTemp", "DailyAverageWdSpeed", "DailyPrecipitation")])


#Store the Numeric Columns with Missing Values in temporary dataset
str(shark.data)
shark.data.temp <- shark.data[,-c(1,2,3,4,5,6,7,8,9,10,15,16,17,18,19)] # added column 2(date) which will be used to merge imputed data with original data
sapply(shark.data.temp, function(x) sum(is.na(x)))

md.pattern(shark.data.temp)

mice_plot <- aggr(shark.data.temp, col=c('navyblue','yellow'),
numbers=TRUE, sortVars=TRUE,
labels=names(shark.data.temp), cex.axis=.7,
gap=3, ylab=c("Proportion of Missingness","Missingness Pattern"))

imputed_Data <- mice(shark.data.temp, m=5, maxit = 40, method = 'pmm', seed = 500)
imputed_Data

#Store the data with imputed values in complete_data
complete_data=complete(imputed_Data,5)
md.pattern(complete_data)
sapply(complete_data, function(x) sum(is.na(x)))




#shark.data <- shark.data[,-c(10,11,12,13)]
#shark.data$DailyAverageTemperature_imputed <- complete_data$DailyAverageTemperature
shark.data$DailyAverageDewPotTemp_imputed <- complete_data$DailyAverageDewPotTemp
#shark.data$DailyAverageWdSpeed_imputed <- complete_data$DailyAverageWdSpeed
shark.data$DailyPrecipitation_imputed <- complete_data$DailyPrecipitation


# normalization
means = apply(complete_data,2,mean)
sds = apply(complete_data,2,sd)
shark.data.normalized = scale(complete_data,center=means,scale=sds)
shark.data.normalized <- as.data.frame(shark.data.normalized)

shark.data$DailyPrecipitation_normalized <- shark.data.normalized$DailyPrecipitation
shark.data$DailyAverageDewPotTemp_normalized <- shark.data.normalized$DailyAverageDewPotTemp
shark.data$DailyAverageWdSpeed_normalized <- shark.data.normalized$DailyAverageWdSpeed
shark.data$DailyAverageTemperature_normalized <- shark.data.normalized$DailyAverageTemperature


plot_histogram(shark.data$DailyAverageTemperature, title ="Histogram for Daily Avg. Temp. after imputation")
shark.data.temperature <- kmeans(shark.data$DailyAverageTemperature , centers = 3)
shark.data.temperature $centers
whichbin <- shark.data.temperature$cluster
whichbin
shark.data$binned.DailyAverageTemperature <- whichbin
shark.data$binned.DailyAverageTemperature[shark.data$binned.DailyAverageTemperature == 1] <- "Low"
shark.data$binned.DailyAverageTemperature[shark.data$binned.DailyAverageTemperature == 2] <- "High"
shark.data$binned.DailyAverageTemperature[shark.data$binned.DailyAverageTemperature == 3] <- "Medium"
par(mfrow=c(1,2))
barplot(table(shark.data$binned.DailyAverageTemperature), main="All Rows")
barplot(table(shark.data$binned.DailyAverageTemperature[shark.data$Attack == "Yes"]), main="Attacks Only")



summary(shark.data$DailyAverageDewPotTemp_imputed)
plot_histogram(shark.data$DailyAverageDewPotTemp_imputed, title="Histogram for Daily Avg. Dew Point after imputation")
shark.data.dewtemperature <- kmeans(shark.data$DailyAverageDewPotTemp_imputed , centers = 3)
shark.data.dewtemperature $centers
whichbin <- shark.data.dewtemperature$cluster
whichbin
shark.data$binned.DailyAverageDewPotTemp <- whichbin
shark.data$binned.DailyAverageDewPotTemp[shark.data$binned.DailyAverageDewPotTemp == 1] <- "Low"
shark.data$binned.DailyAverageDewPotTemp[shark.data$binned.DailyAverageDewPotTemp == 2] <- "High"
shark.data$binned.DailyAverageDewPotTemp[ shark.data$binned.DailyAverageDewPotTemp == 3] <- "Medium"
par(mfrow=c(1,2))
barplot(table(shark.data$binned.DailyAverageDewPotTemp), main="All Rows")
barplot(table(shark.data$binned.DailyAverageDewPotTemp[shark.data$Attack == "Yes"]), main="Attacks Only")



summary(shark.data$DailyAverageWdSpeed)
plot_histogram(shark.data$DailyAverageWdSpeed, title="Histogram for Daily Avg. Wind Speed after imputation")
shark.data.windSpeed <- kmeans(shark.data$DailyAverageWdSpeed, centers = 3)
shark.data.windSpeed $centers
whichbin <- shark.data.windSpeed$cluster
whichbin
shark.data$binned.DailyAverageWdSpeed <- whichbin
shark.data$binned.DailyAverageWdSpeed[shark.data$binned.DailyAverageWdSpeed == 1] <- "Low"
shark.data$binned.DailyAverageWdSpeed[shark.data$binned.DailyAverageWdSpeed == 2] <- "High"
shark.data$binned.DailyAverageWdSpeed[shark.data$binned.DailyAverageWdSpeed == 3] <- "Medium"
par(mfrow=c(1,2))
barplot(table(shark.data$binned.DailyAverageWdSpeed), main="All Rows")
barplot(table(shark.data$binned.DailyAverageWdSpeed[shark.data$Attack == "Yes"]), main="Attacks Only")




summary(shark.data$DailyPrecipitation_imputed)
plot_histogram(shark.data$DailyPrecipitation_imputed, title="Histogram for Daily Avg. Precipitation after imputation")
shark.data.precipitation <- kmeans(shark.data$DailyPrecipitation_imputed , centers = 3)
shark.data.precipitation $centers
whichbin <- shark.data.precipitation$cluster
whichbin
shark.data$binned.DailyPrecipitation <- whichbin
shark.data$binned.DailyPrecipitation[shark.data$binned.DailyPrecipitation == 1] <- "Low"
shark.data$binned.DailyPrecipitation[shark.data$binned.DailyPrecipitation == 2] <- "High"
shark.data$binned.DailyPrecipitation[ shark.data$binned.DailyPrecipitation == 3] <- "Medium"
par(mfrow=c(1,2))
barplot(table(shark.data$binned.DailyPrecipitation), main="All Rows")
barplot(table(shark.data$binned.DailyPrecipitation[shark.data$Attack == "Yes"]), main="Attacks Only")

####### Principal Component Analysis ################
summary(shark.data)

d <- shark.data[,c(14,12,16,17)]
summary(d)
pc <- princomp(d, cor= TRUE,score= TRUE)
biplot(pc)
###################

#shark_atttack_final_dataset <- merge(shark.data, complete_data,by="Sno")
write.csv(shark.data,file= "shark_attack_final_dataset.csv")


############## Under Sampling with Attack = No ###############33
shark.data.attack.no <- subset(shark.data, shark.data$Attack=="No")
View(shark.data.attack.no)
shark.data.undersample <- shark.data.attack.no[-sample(which(shark.data.attack.no$Attack == "No"), 127),]
nrow(shark.data.undersample)
table(shark.data.undersample$Attack)
View(shark.data.undersample)
write.csv(shark.data.undersample,file= "shark_attack_undersample_dataset.csv")

####################################################
######## Manual Proportion Testing ##########
#Reading the file which has undersampled data
shark.data.undersample <- read.csv(file="shark_attack_undersample_dataset.csv", header=T, na.strings = c("NA",""," "), stringsAsFactors = F)
#Calculating proportion of "New" Moonphase before and after undersampling
new_moon_prop_original    <- (length(shark.data$moonphase[shark.data$moonphase == "New"]) / nrow(shark.data))
new_moon_prop_undersample <- (length(shark.data.undersample$moonphase[shark.data.undersample$moonphase == "New"]) / nrow(shark.data.undersample))
#Calculating proportion of "Full" Moonphase before and after undersampling
full_moon_prop_original    <- (length(shark.data$moonphase[shark.data$moonphase == "Full"]) / nrow(shark.data))
full_moon_prop_undersample <- (length(shark.data.undersample$moonphase[shark.data.undersample$moonphase == "Full"]) / nrow(shark.data.undersample))
#Calculating proportion of "First Quarter" Moonphase before and after undersampling
firstQ_moon_prop_original    <- (length(shark.data$moonphase[shark.data$moonphase == "First quarter"]) / nrow(shark.data))
firstQ_moon_prop_undersample <- (length(shark.data.undersample$moonphase[shark.data.undersample$moonphase == "First quarter"]) / nrow(shark.data.undersample))
#Calculating proportion of "Waning Gibbous" Moonphase before and after undersampling
WaningG_moon_prop_original    <- (length(shark.data$moonphase[shark.data$moonphase == "Waning gibbous"]) / nrow(shark.data))
WaningG_moon_prop_undersample <- (length(shark.data.undersample$moonphase[shark.data.undersample$moonphase == "Waning gibbous"]) / nrow(shark.data.undersample))
#Calculating proportion of wind direction "SW" before and after undersampling
SW_Wdirection_prop_original    <- (length(shark.data$moonphase[shark.data$Wd.Direction == "SW"]) / nrow(shark.data))
SW_Wdirection_prop_undersample <- (length(shark.data.undersample$moonphase[shark.data.undersample$Wd.Direction == "SW"]) / nrow(shark.data.undersample))
#Calculating proportion of wind direction "SSW" before and after undersampling
SSW_Wdirection_prop_original    <- (length(shark.data$moonphase[shark.data$Wd.Direction == "SSW"]) / nrow(shark.data))
SSW_Wdirection_prop_undersample <- (length(shark.data.undersample$moonphase[shark.data.undersample$Wd.Direction == "SSW"]) / nrow(shark.data.undersample))
#Calculating proportion of wind direction "WSW" before and after undersampling
WSW_Wdirection_prop_original    <- (length(shark.data$moonphase[shark.data$Wd.Direction == "WSW"]) / nrow(shark.data))
WSW_Wdirection_prop_undersample <- (length(shark.data.undersample$moonphase[shark.data.undersample$Wd.Direction == "WSW"]) / nrow(shark.data.undersample))
#Calculating proportion of wind direction "EAST" before and after undersampling
E_Wdirection_prop_original    <- (length(shark.data$moonphase[shark.data$Wd.Direction == "E"]) / nrow(shark.data))
E_Wdirection_prop_undersample <- (length(shark.data.undersample$moonphase[shark.data.undersample$Wd.Direction == "E"]) / nrow(shark.data.undersample))
#Calculating proportion of wind speed "High" before and after undersampling
High_Wspeed_prop_original    <- (length(shark.data$moonphase[shark.data$binned.DailyAverageWdSpeed == "High"]) / nrow(shark.data))
High_Wspeed_prop_undersample <- (length(shark.data.undersample$moonphase[shark.data.undersample$binned.DailyAverageWdSpeed == "High"]) / nrow(shark.data.undersample))
#Calculating proportion of wind direction "Medium" before and after undersampling
Medium_Wspeed_prop_original    <- (length(shark.data$moonphase[shark.data$binned.DailyAverageWdSpeed == "Medium"]) / nrow(shark.data))
Medium_Wspeed_prop_undersample <- (length(shark.data.undersample$moonphase[shark.data.undersample$binned.DailyAverageWdSpeed == "Medium"]) / nrow(shark.data.undersample))
#Calculating proportion of wind direction "Low" before and after undersampling
Low_Wspeed_prop_original    <- (length(shark.data$moonphase[shark.data$binned.DailyAverageWdSpeed == "Low"]) / nrow(shark.data))
Low_Wspeed_prop_undersample <- (length(shark.data.undersample$moonphase[shark.data.undersample$binned.DailyAverageWdSpeed == "Low"]) / nrow(shark.data.undersample))
############ Presenting the before and after proportions in a table ##############
#converting all the variables into matrix
proportional_table <- matrix(c( new_moon_prop_original, new_moon_prop_undersample, full_moon_prop_original, full_moon_prop_undersample, 
                                firstQ_moon_prop_original, firstQ_moon_prop_undersample, WaningG_moon_prop_original, WaningG_moon_prop_undersample, 
                                SW_Wdirection_prop_original, SW_Wdirection_prop_undersample,  SSW_Wdirection_prop_original, SSW_Wdirection_prop_undersample, 
                                WSW_Wdirection_prop_original, WSW_Wdirection_prop_undersample, E_Wdirection_prop_original, E_Wdirection_prop_undersample, 
                                High_Wspeed_prop_original, High_Wspeed_prop_undersample, Medium_Wspeed_prop_original, Medium_Wspeed_prop_undersample, 
                                Low_Wspeed_prop_original, Low_Wspeed_prop_undersample ),ncol=2,byrow=TRUE)
#adding column names to the proportional matrix
colnames(proportional_table) <- c("Original Proportion","Undersample Proportion")
#adding row names to the proportional matrix
rownames(proportional_table) <- c("New Moon","Full Moon","First Quarter","Waning Gibbous", "Wind Direction - SW", "Wind Direction - SSW", "Wind Direction - WSW ","Wind Direction - East", "Wind Direction - High", "Wind Direction - Medium", "Wind Direction - Low")
#Converting the matrix into table
proportional_table <- as.table(proportional_table)
#displaying the proportional table
proportional_table
#Merging undersampled records (i.e. attack=No) with records where attack = Yes
shark.data.attack.yes <- subset(shark.data, shark.data$Attack=="Yes")
shark.data.undersample <- shark.data.undersample[,-c(1)] #removed extra column which is getting added after writing the file to .csv
shark.data.modelling <- rbind(shark.data.attack.yes , shark.data.undersample)
#Writing the merged dataset into csv which would be used for Modelling
write.csv(shark.data.modelling, file= "shark_attack_modelling_dataset.csv")




shark.data.modelling$Wd.Direction <- factor(shark.data.modelling$Wd.Direction)
shark.data.modelling$Attack <- factor(shark.data.modelling$Attack)
shark.data.modelling$moonphase <- factor(shark.data.modelling$moonphase)

library(caret)
train.rows <- createDataPartition(y= shark.data.modelling$Attack, p=0.75, list = FALSE)
train.data <- shark.data.modelling[train.rows,]
table(train.data$Attack)
test.data <- shark.data.modelling[-train.rows,]
table(test.data$Attack)
str(shark.data.modelling)



#Import Libraries
if(!require(caret)) install.packages("caret",repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest",repos = "http://cran.us.r-project.org")
if(!require(pROC)) install.packages("pROC",repos = "http://cran.us.r-project.org")
if(!require(WVPlots)) install.packages("WVPlots",repos = "http://cran.us.r-project.org")
library(caret)
library(randomForest)
library(pROC)
library(WVPlots)


rf_classifier = randomForest(Attack ~ Wd.Direction + moonphase + DailyAverageWdSpeed, data=train.data, ntree=100, mtry=2, importance=TRUE)
rf_classifier
varImpPlot(rf_classifier)

test.data$forest.pred <- predict(rf_classifier, test.data)
confusionMatrix(test.data$Attack, test.data$forest.pred, positive = "Yes")
GainCurvePlot(test.data, "forest.pred", "Attack", "Gain Curve for Random Forest")
test.data$forest.pred.prob <- predict(rf_classifier, test.data, type= "prob")
auc(as.numeric(test.data$Attack), test.data$forest.pred.prob[,1])


glm.fit <- glm(Attack ~ Wd.Direction + moonphase + DailyAverageWdSpeed, data = train.data, family = binomial)
summary(glm.fit)


test.data$logistic.pred <- predict(glm.fit, newdata = test.data, type = "response")
test.data$logistic.pred <- ifelse(test.data$logistic.pred >= 0.5, "Yes", "No")
test.data$logistic.pred <- factor(test.data$logistic.pred)
confusionMatrix(test.data$Attack, test.data$logistic.pred, positive = "Yes")
GainCurvePlot(test.data, "logistic.pred", "Attack", "Gain Curve for Logistic Regression")
auc(test.data$Attack, as.numeric(test.data$logistic.pred))