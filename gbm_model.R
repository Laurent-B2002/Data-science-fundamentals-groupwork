library(tidyverse)
library(caret)       #ML training framework
library(pROC)        #ROC/AUC

df <- read_csv("marketing_cleaned Pt1.csv")

df <- df %>%
  select(-CustomerID, -Income_missing, -ConversionRate) %>%  # drop non-features / leakage
  mutate(
    Conversion      = factor(Conversion, levels = c(1, 0), labels = c("Yes", "No")),
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
gbm_pred <- predict(gbm_model, test)                          # class

confusionMatrix(gbm_pred, test$Conversion, positive = "Yes")   # confusion matrix

roc_obj <- roc(test$Conversion, gbm_prob)  # ROC
auc(roc_obj)                               # AUC
plot(roc_obj)                              # ROC plot

gbm_model$bestTune                          # best hyperparameters found
gbm_model                                  # model summary
