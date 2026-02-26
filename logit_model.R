library(tidyverse)
library(caret)
library(pROC)
library(broom)

df <- read_csv("marketing_cleaned Pt1.csv")

df <- df %>%
  select(-CustomerID, -Income_missing, -ConversionRate) %>%
  mutate(
    Conversion      = as.integer(Conversion),
    Gender          = as.factor(Gender),
    CampaignChannel = as.factor(CampaignChannel),
    CampaignType    = as.factor(CampaignType)
  )

set.seed(42)
idx <- createDataPartition(df$Conversion, p = 0.8, list = FALSE)
train <- df[idx, ]
test <- df[-idx, ]

fit <- glm(Conversion ~ ., data = train, family = binomial())

summary(fit)

or_table <- tidy(fit, conf.int = TRUE, exponentiate = TRUE) %>%
  arrange(desc(estimate))
or_table

test$prob <- predict(fit, newdata = test, type = "response")
test$pred <- if_else(test$prob >= 0.5, 1L, 0L)

confusionMatrix(
  factor(test$pred, levels = c(0,1)),
  factor(test$Conversion, levels = c(0,1))
)

roc_obj <- roc(test$Conversion, test$prob)
auc(roc_obj)
plot(roc_obj)