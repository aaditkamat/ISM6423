library(dplyr)
library(ggplot2)
library(readr)
library(caret)

# Read the CSV file
marketing_mix_df <- read.csv('marketing_mix.csv')
# View the contents of the first 5 rows
print(head(marketing_mix_df))

# Plot each of the input variables with the target variable
hist(marketing_mix_df$TikTok)
hist(marketing_mix_df$Facebook)
hist(marketing_mix_df$Google.Ads)


# Split the data into training and testing data sets using 80-20 ratio
set.seed(42)
trainIndex <- createDataPartition(marketing_mix_df$Sales, p=0.8, list=FALSE)
MM_train <- marketing_mix_df[trainIndex, -1]
MM_test <- marketing_mix_df[-trainIndex, -1]

# Train linear regression model on marketing mix data using log to adjust for skewedness
LM1 <- lm(Sales ~ log(TikTok + 1) + log(Facebook + 1) + log(Google.Ads + 1), data=MM_train)

# View the summary for the linear regression model
print(summary(LM1))

# Use test data for prediction with the linear regression model
LM1_results <- predict(LM1, MM_test[, -5])

# Compare the predictions with the actual values using error metrics
mae_LM1  <- MAE(LM1_results, MM_test$Sales)
rmse_LM1 <- RMSE(LM1_results, MM_test$Sales)
print(paste('RMSE: ', rmse_LM1, ' MAE: ', mae_LM1))

# Train generalized linear regression model on marketing mix data with tuning
tuneControl <- trainControl(method='cv', number=5)
LM2 <- train(Sales ~. , data=MM_train, 
            method='glmnet', 
            trControl=tuneControl, 
            tuneLength=25)

# Use test data for prediction with the generalized linear regression model
LM2_results <- predict(LM2, MM_test[, -5])

# Compare the predictions with the actual values using error metrics
mae_LM2 <- MAE(LM2_results, MM_test$Sales)
rmse_LM2 <- RMSE(LM2_results, MM_test$Sales)
print(paste('RMSE: ', rmse_LM2, ' MAE: ', mae_LM2))
