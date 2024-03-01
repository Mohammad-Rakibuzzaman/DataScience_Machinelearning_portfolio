data <- read.csv("C:/Users/RTZ/Desktop/10thsemester/Rubymam/R all code here B)/data.csv",header=TRUE)
library(tidyverse)

cor(data)
corMatrix<-round(cor(data),2)

library(GGally)
ggpairs(data)

library(caret)
findCorrelation(corMatrix, cutoff = 0.6, names = TRUE)

data11 <- data %>%
  select(-CRIM, -ZN,-CHAS ,-NOX,-RM,-AGE,-RAD,-TAX,-PTRATIO,-B)
data11

scale01 <- function(x){(x - min(x)) / (max(x) - min(x))}
data111 <- data11 %>%
	mutate_all(scale01)



reg<-lm(MEDV~INDUS+LSTAT+DIS,data111)
summary(reg)
library(car)
vif(reg)



data1_Train <- sample_frac(data111, replace = FALSE, size = 0.80)
data1_Test <- anti_join(data111, data1_Train)

library(neuralnet)
set.seed(12321)
NN1 <- neuralnet(MEDV~INDUS+LSTAT+DIS, data= data1_Train)
plot(NN1, rep = 'best')
Test_NN1_Output <- compute(NN1, data1_Test[, 1:3])$net.result

library(Metrics)

NN1_Test_RMSE <- rmse(Test_NN1_Output,data1_Test[,4])
NN1_Test_RMSE