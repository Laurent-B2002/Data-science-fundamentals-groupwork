library(tidyverse)
library(caret)       #ML training framework
library(pROC)        #ROC/AUC

df <- read_csv("marketing_cleaned Pt1.csv")

df <- df %>%
  select(-CustomerID, -Income_missing) %>%  # drop non-features / leakage
  mutate(
    Conversion      = factor(Conversion, levels = c(0,1), labels = c("No", "Yes")),
    Gender          = as.factor(Gender),
    CampaignChannel = as.factor(CampaignChannel),
    CampaignType    = as.factor(CampaignType)
  ) %>%
  drop_na()  # caret models generally require complete cases

set.seed(42)  
idx<- createDataPartition(df$Conversion, p = 0.8, list = FALSE)  # stratified split
train <- df[idx, ]
test  <- df[-idx, ]

set.seed(42)
ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

gbm_model <- train(
  Conversion ~ .,
  data = train,
  method = "gbm",
  trControl = ctrl,
  metric = "ROC",
  verbose = FALSE,
  tuneLength = 5
)

gbm_prob <- predict(gbm_model, test, type = "prob")[, "Yes"]  # P(Yes)

threshold <- 0.7  

gbm_pred <- ifelse(gbm_prob > threshold, "Yes", "No")
gbm_pred <- factor(gbm_pred, levels = c("No","Yes"))                          # class

confusionMatrix(gbm_pred, test$Conversion, positive = "Yes")   # confusion matrix

roc_obj <- roc(test$Conversion, gbm_prob)  # ROC
auc(roc_obj)                               # AUC
plot(roc_obj)                              # ROC plot

gbm_model$bestTune                          # best hyperparameters found
gbm_model                                  # model summary

#----------------------------------------------------------------------------
cm <- confusionMatrix(gbm_pred, test$Conversion, positive = "Yes")

TN <- unname(cm$table["No", "No"])
FN <- unname(cm$table["No", "Yes"])
FP <- unname(cm$table["Yes", "No"])
TP <- unname(cm$table["Yes", "Yes"])

gbm_results <- data.frame(
  n.trees = gbm_model$bestTune$n.trees,
  interaction.depth = gbm_model$bestTune$interaction.depth,
  shrinkage = gbm_model$bestTune$shrinkage,
  TN = TN,
  FP = FP,
  FN = FN,
  TP = TP,
  Accuracy = unname(cm$overall["Accuracy"]),
  Precision = unname(cm$byClass["Pos Pred Value"]),
  Recall = unname(cm$byClass["Sensitivity"]),
  Specificity = unname(cm$byClass["Specificity"]),
  F1 = unname(cm$byClass["F1"]),
  BalancedAccuracy = unname(cm$byClass["Balanced Accuracy"]),
  AUC = as.numeric(auc(roc_obj))
)

write.csv(gbm_results, "gbm_test_results_07.csv", row.names = FALSE)

write.csv(gbm_model$results, "gbm_cv_results_07.csv", row.names = FALSE)


