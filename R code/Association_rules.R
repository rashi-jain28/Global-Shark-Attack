## Association Rules ##

shark.data <- read.csv(file="shark_attack_final_dataset.csv", header=T, na.strings = c("NA",""," "), stringsAsFactors = F)

if(!require(arules)) install.packages("arules",repos = "http://cran.us.r-project.org")
library(arules)