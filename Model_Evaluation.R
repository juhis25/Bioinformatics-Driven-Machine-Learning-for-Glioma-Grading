# Load necessary libraries
library(pROC)

# Evaluate the best model
best_model <- models$rf  # Assume random forest is the best for demonstration
pred <- predict(best_model, newdata = test_data)
confusionMatrix(pred, test_data$grade)

# ROC Curve
roc_curve <- roc(test_data$grade, as.numeric(pred))
plot(roc_curve, print.auc = TRUE)
