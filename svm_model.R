library(tidyverse)
library(caret)
library(pROC)
library(e1071)
library(kernlab)
library(doParallel)

#make jobs run in parallel with all cores
cl <- makeCluster(parallel::detectCores() - 1)
registerDoParallel(cl)
#read in data
df <- read_csv("marketing_cleaned Pt1.csv")
#remove irrelevant and possibly leakage causing variables
df <- df %>%
  select(-CustomerID, -Income_missing, -ConversionRate) %>%
  mutate(
    Conversion      = factor(Conversion, levels = c(0,1), labels = c("No", "Yes")),
    Gender          = as.factor(Gender),
    CampaignChannel = as.factor(CampaignChannel),
    CampaignType    = as.factor(CampaignType)
  ) %>%
  drop_na()
#train test splits
set.seed(42)
idx <- createDataPartition(df$Conversion, p = 0.8, list = FALSE)
train <- df[idx, ]
test <- df[-idx, ]

#hyperparameter grid for svm
sig <- sigest(Conversion ~ ., data = train)
sigma_grid <- as.numeric(sig)
sigma_grid <- sigma_grid[is.finite(sigma_grid) & sigma_grid > 0]

grid <- expand.grid(
  sigma = sigma_grid,
  C     = 2^seq(-2, 6, by = 1)
)


ctrl <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 3,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)
#svm model
set.seed(42)
svm_model <- train(
  Conversion ~ .,
  data = train,
  method = "svmRadial",
  trControl = ctrl,
  metric = "ROC",
  preProcess = c("center", "scale"),
  tuneGrid = grid
)

svm_prob <- predict(svm_model, test, type = "prob")[,"Yes"]
#roc curve
roc_obj <- roc(test$Conversion, svm_prob)
auc(roc_obj)
plot(roc_obj)
#finding the best threshold
best <- coords(roc_obj, "best", best.method="youden", transpose=FALSE)
best

thr <- best[["threshold"]]
#optimized prediction
svm_pred_opt <- ifelse(svm_prob > thr, "Yes", "No") %>%
  factor(levels = c("No", "Yes"))

confusionMatrix(svm_pred_opt, test$Conversion, positive = "Yes")
#best parameters
svm_model$bestTune
plot(svm_model)
#section below for creating csvs
grid <- svm_model$results[, c("sigma", "C")]
#bigger C causes problems
grid2 <- subset(grid, C <= 8)
results_list <- vector("list", nrow(grid2))

for (i in seq_len(nrow(grid2))) {
  
  sigma_i <- grid2$sigma[i]
  C_i     <- grid2$C[i]
  
  out <- tryCatch({
    
    model_i <- train(
      Conversion ~ .,
      data = train,
      method = "svmRadial",
      trControl = trainControl(method = "none", classProbs = TRUE),
      preProcess = c("center", "scale"),
      tuneGrid = data.frame(sigma = sigma_i, C = C_i)
    )
    
    prob_df <- predict(model_i, test, type = "prob")
    
    if (!("Yes" %in% colnames(prob_df))) stop("No 'Yes' prob column returned.")
    prob_i <- prob_df[,"Yes"]
    if (!is.numeric(prob_i)) stop("Probability column is not numeric.")
    if (anyNA(prob_i)) stop("NA probabilities.")
    
    roc_i <- roc(response = test$Conversion, predictor = prob_i, levels = c("No","Yes"), direction = "<")
    auc_i <- as.numeric(auc(roc_i))
    
    best_i <- coords(roc_i, "best", best.method = "youden", transpose = FALSE)
    thr_i  <- best_i[["threshold"]]
    
    pred_i <- factor(ifelse(prob_i > thr_i, "Yes", "No"), levels = c("No","Yes"))
    
    cm <- confusionMatrix(pred_i, test$Conversion, positive = "Yes")
    
    TN <- unname(cm$table["No",  "No"])
    FN <- unname(cm$table["No",  "Yes"])
    FP <- unname(cm$table["Yes", "No"])
    TP <- unname(cm$table["Yes", "Yes"])
    
    data.frame(
      sigma = sigma_i,
      C = C_i,
      threshold = thr_i,
      AUC = auc_i,
      TN = TN, FP = FP, FN = FN, TP = TP,
      Accuracy = unname(cm$overall["Accuracy"]),
      Precision = unname(cm$byClass["Pos Pred Value"]),
      Recall = unname(cm$byClass["Sensitivity"]),
      Specificity = unname(cm$byClass["Specificity"]),
      F1 = unname(cm$byClass["F1"]),
      BalancedAccuracy = unname(cm$byClass["Balanced Accuracy"]),
      error = NA_character_
    )
    
  }, error = function(e) {
    data.frame(
      sigma = sigma_i,
      C = C_i,
      threshold = NA_real_,
      AUC = NA_real_,
      TN = NA_integer_, FP = NA_integer_, FN = NA_integer_, TP = NA_integer_,
      Accuracy = NA_real_, Precision = NA_real_, Recall = NA_real_,
      Specificity = NA_real_, F1 = NA_real_, BalancedAccuracy = NA_real_,
      error = paste0("FAILED: ", conditionMessage(e))
    )
  })
  
  results_list[[i]] <- out
}

results_df <- do.call(rbind, results_list)

#subset(results_df, is.na(AUC))[, c("sigma","C","error")]
results_ok <- subset(results_df, !is.na(AUC))
#cv results
write.csv(svm_model$results, "svm_cv_results.csv", row.names = FALSE)
#model results with problems removed
write.csv(results_ok, "svm_test_results.csv", row.names = FALSE)

stopCluster(cl)