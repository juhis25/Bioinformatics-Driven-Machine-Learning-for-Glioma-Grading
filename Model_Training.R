# Load necessary libraries
library(caret)
library(randomForest)
library(e1071)

# Split the enhanced data into training and testing sets
set.seed(123)
train_index <- createDataPartition(glioma_df$grade, p = 0.8, list = FALSE)
train_data <- glioma_df[train_index, ]
test_data <- glioma_df[-train_index, ]

# Define the training control
train_control <- trainControl(method = "cv", number = 10)

# Define the models to train
set.seed(123)
models <- list(
    rf = train(grade ~ ., data = train_data, method = "rf", trControl = train_control),
    svm = train(grade ~ ., data = train_data, method = "svmRadial", trControl = train_control),
    glmnet = train(grade ~ ., data = train_data, method = "glmnet", trControl = train_control)
)

# Compare model performances
results <- resamples(models)
summary(results)
