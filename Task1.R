if(!require("caret")) install.packages("caret"); library("caret") 
if(!require("xgboost")) install.packages("xgboost"); library("xgboost") 
setwd("/Users/sarjodas/Desktop/")
#data = read.csv("data.csv", sep = ";")
#write.csv(data, file = "data_edited_Task2.csv", row.names=FALSE)

##################################################################################################
data = read.csv("data_edited.csv")
#summary(data) 
########################################Noramlizing Data in 0 to 1 scale##########################################################

normalized_rloc = (data$rloc-min(data$rloc)) / (max(data$rloc) - min(data$rloc))
normalized_mccc = (data$mccc-min(data$mccc)) / (max(data$mccc) - min(data$mccc))
normalized_nl4 =  (data$nl4-min(data$nl4)) / (max(data$nl4) - min(data$nl4))

data_norm = data.frame(normalized_nl4, normalized_mccc, normalized_rloc)

#####################################Llinear Regression with Normalized Data#############################################################
set.seed(123)
idx.train <- createDataPartition(y = data_norm$normalized_nl4, p = 0.80, list = FALSE)
train_norm <- data_norm[idx.train, ]
test_norm <-  data_norm[-idx.train, ]

ctrl<-trainControl(method = "cv",number = 5)
lmCVFit_norm <- train(normalized_nl4 ~ ., data = train_norm, method = "lm", trControl = ctrl, metric="Rsquared")
summary(lmCVFit_norm)

prediction_norm = predict(lmCVFit_norm, test_norm)
new.value_norm = data.frame(obs = test_norm$normalized_nl4, pred = prediction_norm)
defaultSummary(new.value_norm)

#Rsquared: 0.78401194

#########################################Linear Regression#######################################################
set.seed(123)
idx.train <- createDataPartition(y = data$nl4, p = .80, list = FALSE)
train <- data[idx.train, ]
test <- data[-idx.train, ]

ctrl<-trainControl(method = "cv",number = 10)
lmCVFit <- train(nl4 ~ ., data = train, method = "lm", trControl = ctrl, metric="Rsquared")
summary(lmCVFit)


prediction = predict(lmCVFit, test)
new.value = data.frame(obs = test$nl4, pred = prediction)
defaultSummary(new.value)

#Rsquared: 0.78401194

xyplot(resid(lmCVFit) ~ predict(lmCVFit), xlim=c(-.5,2e+5),
       type = c("p", "g"),
       main = "Residual Plot: Linear Regression: R^2 = 0.78401194",
       xlab = "Predicted", ylab = "Residuals")
#############################################xgboost#######################################################

model.control <- trainControl(
  method = "cv", number = 5, classProbs = FALSE,
  returnData = TRUE
)

xgb.prams <- expand.grid(nrounds = c(20,40,80), 
                         max_depth = c(5,10),
                         eta = c(0.001,0.01,0.1),
                         gamma = c(0,1),
                         colsample_bytree = c(0.5,0.7, 1),
                         min_child_weight = 1,
                         subsample = c(0.5, 1))

xgb <- train(nl4~.,data = train, method = "xgbTree", tuneGrid = xgb.prams, 
             metric = "Rsquared", trControl = model.control)

prediction_xgb = predict(xgb, test)
new.value_xgb = data.frame(obs = test$nl4, pred = prediction_xgb)
defaultSummary(new.value_xgb)

#Rsquared: 8.315549e-01

#The final values used for the model were nrounds = 20, max_depth = 10, eta = 0.1, gamma = 1, colsample_bytree = 1, min_child_weight = 1 and subsample = 0.5.

#############################################Random Forest##############################################################

rf.parms <- expand.grid(mtry = 1:25)
rf.caret <- train(nl4 ~ ., data = train,  
                  method = "rf", ntree = 500, tuneGrid = rf.parms, 
                  metric = "Rsquared", trControl = model.control)


prediction_xgb = predict(rf.caret, test)
new.value.rf = data.frame(obs = test$nl4, pred = prediction_xgb)
defaultSummary(new.value.rf)

#Rsquared: 9.107395e-01

xyplot(resid(rf.caret) ~ predict(rf.caret), xlim=c(-.5,2e+5),
       type = c("p", "g"),
       main = "Residual Plot: Random Forest: R^2 = 0.9107395",
       xlab = "Predicted", ylab = "Residuals")

#The final value used for the model was mtry = 23


