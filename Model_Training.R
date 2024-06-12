# Load necessary libraries
library(caret)
library(randomForest)
library(e1071)


# Train a Random Forest model
set.seed(123)
rf_model <- randomForest(Grade ~ ., data = train_data, importance = TRUE)
rf_predictions <- predict(rf_model, newdata = test_data, type = "response")
rf_prob <- predict(rf_model, newdata = test_data, type = "prob")[,2]

# Confusion Matrix
rf_cm <- confusionMatrix(rf_predictions, test_data$Grade)
print(rf_cm)

# Train an SVM model
set.seed(123)
svm_model <- svm(Grade ~ ., data = train_data, probability = TRUE)
svm_predictions <- predict(svm_model, newdata = test_data, probability = TRUE)
svm_prob <- attr(predict(svm_model, newdata = test_data, probability = TRUE), 
                 "probabilities")[,2]

# Confusion Matrix
svm_cm <- confusionMatrix(svm_predictions, test_data$Grade)
print(svm_cm)

# Train a Neural Network model
set.seed(123)
nn_model <- nnet(Grade ~ ., data = train_data, size = 10, linout = FALSE)

# Predict using the Neural Network model
nn_predictions <- predict(nn_model, newdata = test_data, type = "class")
nn_prob <- predict(nn_model, newdata = test_data, type = "raw")

# Convert predictions to factors with the same levels as test_data$Grade
nn_predictions <- factor(nn_predictions, levels = levels(test_data$Grade))

# Confusion Matrix
nn_cm <- confusionMatrix(nn_predictions, test_data$Grade)
print(nn_cm)
