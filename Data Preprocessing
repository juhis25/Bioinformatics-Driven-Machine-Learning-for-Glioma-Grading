# Load necessary libraries
library(dplyr)
library(tidyr)

# Load the data
glioma_data <- read.csv("~/glioma_data.csv")

# Inspect the data
str(glioma_data)
summary(glioma_data)

# Handle missing values (if any)
glioma_data <- glioma_data %>%
    drop_na()

# Split the data into features and labels
features <- glioma_data %>%
    select(-grade)  # Assuming 'grade' is the target variable
labels <- glioma_data$grade
