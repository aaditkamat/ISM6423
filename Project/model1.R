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
linear <- train(Sales ~ log(TikTok + 1) + log(Facebook + 1) + log(Google.Ads + 1), data=MM_train,
                method='lm',
                metric='Rsquared')

# View the summary for the linear regression model
print(summary(linear))

# Train LASSO model on marketing mix data
tuneControl <- trainControl(method='cv', number=5)
gridParameters <- seq(0.1, 5, 20)

lasso <- train(Sales ~. , data=MM_train, 
            method='glmnet', 
            trControl=tuneControl,
            tuneGrid=expand.grid(alpha=1, lambda=gridParameters),
            metric='Rsquared')

ridge <- train(Sales ~. , data=MM_train, 
               method='glmnet', 
               trControl=tuneControl,
               tuneGrid=expand.grid(alpha=0, lambda=gridParameters),
               metric='Rsquared')

# Compare the coefficients from the different models
data.frame(
  ridge = as.data.frame.matrix(coef(ridge$finalModel, ridge$finalModel$lambdaOpt)),
  lasso = as.data.frame.matrix(coef(lasso$finalModel, lasso$finalModel$lambdaOpt)), 
  linear = linear$finalModel$coefficients
) %>% rename(ridge=s1, lasso=s1.1)

# Compare the results of the different models using different error metrics
predictions_ridge <- predict(ridge, MM_test[, 1:3])
predictions_lasso <- predict(lasso, MM_test[, 1:3])
predictions_linear <- predict(linear, MM_test[, 1:3])

values <- matrix(
  c(R2(predictions_ridge, MM_test$Sales), 
    R2(predictions_lasso, MM_test$Sales), 
    R2(predictions_linear, MM_test$Sales),
    RMSE(predictions_ridge, MM_test$Sales),
    RMSE(predictions_lasso, MM_test$Sales),
    RMSE(predictions_linear, MM_test$Sales),
    MAE(predictions_ridge, MM_test$Sales),
    MAE(predictions_lasso, MM_test$Sales),
    MAE(predictions_linear, MM_test$Sales)),
  nrow=3,
  ncol=3
)
results <- data.frame(values)
rownames(results) <- c('R2', 'RMSE', 'MAE')
colnames(results) <- c('Ridge', 'Lasso', 'Linear')
