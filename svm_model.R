library(tidyverse)
library(caret)
library(pROC)
library(e1071)

df <- read_csv("marketing_cleaned Pt1.csv")

df <- df %>%
  select(-CustomerID, -Income_missing, -ConversionRate) %>%
  mutate(
    Conversion      = factor(Conversion, levels = c(0,1), labels = c("No", "Yes")),
    Gender          = as.factor(Gender),
    CampaignChannel = as.factor(CampaignChannel),
    CampaignType    = as.factor(CampaignType)
  ) %>%
  drop_na()

set.seed(42)
idx <- createDataPartition(df$Conversion, p = 0.8, list = FALSE)
train <- df[idx, ]
test <- df[-idx, ]

set.seed(42)

ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

svm_model <- train(
  Conversion ~ .,
  data = train,
  method = "svmRadial",
  trControl = ctrl,
  metric = "ROC",
  preProcess = c("center", "scale"),
  tuneLength = 5
)

svm_prob <- predict(svm_model, test, type = "prob")[,"Yes"]
svm_pred <- predict(svm_model, test)

confusionMatrix(svm_pred, test$Conversion, positive = "Yes")

roc_obj <- roc(test$Conversion, svm_prob)
auc(roc_obj)
plot(roc_obj)