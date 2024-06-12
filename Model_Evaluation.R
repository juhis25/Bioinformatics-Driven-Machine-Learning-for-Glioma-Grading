# Load necessary libraries
library(pROC)

# Plot ROC curves
roc_rf <- roc(test_data$Grade, as.numeric(rf_prob), levels = rev(levels(test_data$Grade)))
roc_svm <- roc(test_data$Grade, as.numeric(svm_prob), levels = rev(levels(test_data$Grade)))
roc_nn <- roc(test_data$Grade, as.numeric(nn_prob), levels = rev(levels(test_data$Grade)))

plot(roc_rf, col = "blue", main = "ROC Curves for Machine Learning Models")
lines(roc_svm, col = "red")
lines(roc_nn, col = "green")
legend("bottomright", legend = c("Random Forest", "SVM", "Neural Network"), col = c("blue", "red", "green"), lwd = 2)

# Function to plot confusion matrix
plot_confusion_matrix <- function(cm, title) {
  cm_table <- as.data.frame(cm$table)
  ggplot(cm_table, aes(x = Prediction, y = Reference)) +
    geom_tile(aes(fill = Freq), color = "white") +
    geom_text(aes(label = Freq), vjust = 1) +
    scale_fill_gradient(low = "white", high = "steelblue") +
    labs(title = title, x = "Predicted", y = "Actual") +
    theme_minimal()
}

# Plot confusion matrices
plot_confusion_matrix(rf_cm, "Random Forest Confusion Matrix")
plot_confusion_matrix(svm_cm, "SVM Confusion Matrix")
plot_confusion_matrix(nn_cm, "Neural Network Confusion Matrix")



# Replace grade labels 0 and 1 with 'LGG' and 'GBM'
info_with_grade$Grade <- factor(info_with_grade$Grade, levels = c(0, 1), labels = c('LGG', 'GBM'))

# Convert relevant columns to factors
info_with_grade$Gender <- factor(info_with_grade$Gender, levels = c(0, 1), labels = c('Male', 'Female'))

# Handle missing values (if any)
info_with_grade <- na.omit(info_with_grade)

# Normalize numeric columns
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

info_with_grade$Age_at_diagnosis <- normalize(info_with_grade$Age_at_diagnosis)

# Ensure all columns except 'Grade' are numeric
numeric_features <- info_with_grade %>% select(-Grade) %>% mutate_if(is.character, as.numeric)

# Split data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(info_with_grade$Grade, p = .8, 
                                  list = FALSE, 
                                  times = 1)
train_data <- numeric_features[trainIndex,]
test_data <- numeric_features[-trainIndex,]
train_labels <- info_with_grade$Grade[trainIndex]
test_labels <- info_with_grade$Grade[-trainIndex]

# Train a Random Forest model
set.seed(123)
rf_model <- randomForest(Grade ~ ., data = info_with_grade[trainIndex,], importance = TRUE)
rf_predictions <- predict(rf_model, newdata = info_with_grade[-trainIndex,], type = "response")
rf_prob <- predict(rf_model, newdata = info_with_grade[-trainIndex,], type = "prob")[,2]

# Confusion Matrix for Random Forest
rf_cm <- confusionMatrix(rf_predictions, test_labels)
print(rf_cm)

# Train an SVM model
set.seed(123)
svm_model <- svm(Grade ~ ., data = info_with_grade[trainIndex,], probability = TRUE)
svm_predictions <- predict(svm_model, newdata = info_with_grade[-trainIndex,], probability = TRUE)
svm_prob <- attr(predict(svm_model, newdata = info_with_grade[-trainIndex,], probability = TRUE), "probabilities")[,2]

# Confusion Matrix for SVM
svm_cm <- confusionMatrix(svm_predictions, test_labels)
print(svm_cm)

# Train a Neural Network model
set.seed(123)
nn_model <- nnet(Grade ~ ., data = info_with_grade[trainIndex,], size = 10, linout = FALSE)

# Predict using the Neural Network model
nn_predictions <- predict(nn_model, newdata = info_with_grade[-trainIndex,], type = "class")
nn_prob <- predict(nn_model, newdata = info_with_grade[-trainIndex,], type = "raw")

# Convert predictions to factors with the same levels as test_labels
nn_predictions <- factor(nn_predictions, levels = levels(test_labels))

# Confusion Matrix for Neural Network
nn_cm <- confusionMatrix(nn_predictions, test_labels)
print(nn_cm)

# Train a K-Nearest Neighbors (KNN) model
set.seed(123)
knn_model <- knn(train = train_data, test = test_data, cl = train_labels, k = 5)

# Confusion Matrix for KNN
knn_cm <- confusionMatrix(knn_model, test_labels)
print(knn_cm)

# Plot ROC curves
roc_rf <- roc(test_labels, as.numeric(rf_prob), levels = rev(levels(test_labels)))
roc_svm <- roc(test_labels, as.numeric(svm_prob), levels = rev(levels(test_labels)))
roc_nn <- roc(test_labels, as.numeric(nn_prob), levels = rev(levels(test_labels)))
roc_knn <- roc(test_labels, as.numeric(as.character(knn_model)), levels = rev(levels(test_labels)))

plot(roc_rf, col = "blue", main = "ROC Curves for Machine Learning Models")
lines(roc_svm, col = "red")
lines(roc_nn, col = "green")
lines(roc_knn, col = "purple")
legend("bottomright", legend = c("Random Forest", "SVM", "Neural Network", "KNN"), col = c("blue", "red", "green", "purple"), lwd = 2)

# Function to plot confusion matrix
plot_confusion_matrix <- function(cm, title) {
  cm_table <- as.data.frame(cm$table)
  ggplot(cm_table, aes(x = Prediction, y = Reference)) +
    geom_tile(aes(fill = Freq), color = "white") +
    geom_text(aes(label = Freq), vjust = 1) +
    scale_fill_gradient(low = "white", high = "steelblue") +
    labs(title = title, x = "Predicted", y = "Actual") +
    theme_minimal()
}

# Plot confusion matrices
plot_confusion_matrix(rf_cm, "Random Forest Confusion Matrix")
plot_confusion_matrix(svm_cm, "SVM Confusion Matrix")
plot_confusion_matrix(nn_cm, "Neural Network Confusion Matrix")
plot_confusion_matrix(knn_cm, "KNN Confusion Matrix")
