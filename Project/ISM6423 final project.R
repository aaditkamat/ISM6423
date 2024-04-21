library(dplyr)
library(ggplot2)
library(readr)
library(maps)

df_marketing <- read.csv("Marketing_Data.csv")
df_user <- read.csv("User_Data.csv")
df_advertising <- read.csv("Advertising_Data.csv")

# Automatically convert all logical columns to integers in User_Data
df_user <- df_user %>% mutate(across(where(is.logical), as.integer)) %>% mutate()

# Generate box plot showing age range per platform by gender
age_per_platform_by_gender <- ggplot(data=df_user, mapping=aes(x=platform, y=age)) + 
  geom_boxplot() + 
  facet_grid(gender ~ .)
# Generate column charts showing interests per country based on demographics
interests_per_country_by_demographics <- ggplot(data=df_user, mapping=aes(x=location, fill=interests)) + 
  geom_bar(position=position_dodge(width = 0.8), binwidth = 25) + 
  facet_grid(demographics ~ .)

# Merging datasets
complete_data <- df_marketing %>%
  filter(tolower(category) == "social") %>%
  left_join(df_advertising, by = "id") %>%
  left_join(df_user, by = "id")

# Checking for missing values
sum(is.na(complete_data$Product_Sold))

# Selecting relevant columns for the model
data_model <- complete_data %>%
  select(mark_spent, time_spent, Product_Sold) %>%
  na.omit()  # Removing rows with missing values

# Splitting data into training and testing sets
set.seed(42)
training_samples <- data_model$Product_Sold %>%
  createDataPartition(p = 0.8, list = FALSE)
train_data <- data_model[training_samples, ]
test_data <- data_model[-training_samples, ]

# Training a linear regression model
model <- lm(Product_Sold ~ mark_spent + time_spent, data = train_data)

# Model summary
summary(model)

# Predicting on test set
predictions <- predict(model, test_data)

# Calculating RMSE
rmse <- sqrt(mean((predictions - test_data$Product_Sold)^2))

# Output RMSE
print(paste("Root Mean Squared Error:", rmse))

# Plotting actual vs predicted values
ggplot(test_data, aes(x = Product_Sold, y = predictions)) +
  geom_point() +
  geom_smooth(method = lm, col = "blue") +
  labs(title = "Actual vs Predicted Sales", x = "Actual Sales", y = "Predicted Sales")


