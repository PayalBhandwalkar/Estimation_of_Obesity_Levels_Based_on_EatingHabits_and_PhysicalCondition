# Install necessary libraries
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(tidyverse, caret, nnet, factoextra, rpart, rpart.plot, viridis, scales, ggplot2)

# Load the dataset
file_path <- "/Users/payalbhandwalkar/Downloads/ObesityDataSet_raw_and_data_sinthetic.csv"
obesity <- read.csv(file_path)

# Initial Inspection
print(head(obesity))    # Print the first few rows
str(obesity)            # Display structure of the dataset
summary(obesity)        # Provide summary statistics

# Convert necessary columns to factors
obesity <- obesity %>%
  mutate(
    Gender = as.factor(Gender),
    family_history_with_overweight = as.factor(family_history_with_overweight),
    FAVC = as.factor(FAVC),
    CAEC = as.factor(CAEC),
    SMOKE = as.factor(SMOKE),
    SCC = as.factor(SCC),
    CALC = as.factor(CALC),
    MTRANS = as.factor(MTRANS),
    NObeyesdad = as.factor(NObeyesdad)  # Target variable
  )

# Check for missing values
missing_values <- colSums(is.na(obesity))
print(missing_values)

# Feature Engineering: Calculate BMI
obesity <- obesity %>%
  mutate(BMI = Weight / (Height^2))

# Scaling numeric features
numeric_features <- obesity %>%
  select(Age, Height, Weight, FCVC, NCP, CH2O, FAF, TUE, BMI)
scaled_features <- scale(numeric_features)

# Combine scaled features with categorical columns
obesity_scaled <- cbind(scaled_features, obesity %>%
                          select(Gender, family_history_with_overweight, FAVC, CAEC, SMOKE, SCC, CALC, MTRANS, NObeyesdad))

# Split data into training and testing sets
set.seed(123)
train_index <- createDataPartition(obesity_scaled$NObeyesdad, p = 0.8, list = FALSE)
train_data <- obesity_scaled[train_index, ]
test_data <- obesity_scaled[-train_index, ]

# Visualizations
# 1. Distribution of Obesity Levels
ggplot(obesity, aes(x = NObeyesdad, fill = NObeyesdad)) +
  geom_bar() +
  scale_fill_viridis_d() +
  theme_minimal() +
  labs(title = "Distribution of Obesity Levels",
       x = "Obesity Level",
       y = "Count") +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "none"
  ) +
  coord_flip()

# 2. Age vs BMI by Obesity Level
ggplot(obesity, aes(x = Age, y = BMI, color = NObeyesdad)) +
  geom_point(alpha = 0.7, size = 3) +
  geom_smooth(method = "lm", se = FALSE, color = "darkgray", linetype = "dashed") +
  scale_color_viridis_d() +
  theme_minimal() +
  labs(title = "Age vs. BMI by Obesity Level", x = "Age (years)", y = "BMI")

# 3. Physical Activity Frequency by Obesity Level
ggplot(obesity, aes(x = NObeyesdad, y = FAF, fill = NObeyesdad)) +
  geom_boxplot() +
  scale_fill_viridis_d() +
  theme_minimal() +
  labs(title = "Physical Activity Frequency by Obesity Level",
       x = "Obesity Level",
       y = "Physical Activity Frequency")

# 4. Transportation Method by Obesity Level
ggplot(obesity, aes(x = fct_infreq(NObeyesdad), fill = MTRANS)) +
  geom_bar(position = "fill", width = 0.8) +
  scale_fill_viridis_d() +
  coord_flip() +
  theme_minimal() +
  labs(title = "Transportation Method Distribution Across Obesity Levels",
       x = "Obesity Level",
       y = "Proportion")

# Neural Network Model
set.seed(123)
nn_model <- nnet(
  NObeyesdad ~ ., 
  data = train_data, 
  size = 5, 
  decay = 0.1, 
  maxit = 200
)

# Evaluate Neural Network
nn_predictions <- predict(nn_model, test_data, type = "class")
nn_confusion <- confusionMatrix(as.factor(nn_predictions), test_data$NObeyesdad)
print(nn_confusion)

# Decision Tree Model
set.seed(123)
dt_model <- rpart(NObeyesdad ~ ., data = train_data, method = "class")
rpart.plot(dt_model, extra = 101, under = TRUE)

# Decision Tree Evaluation
dt_predictions <- predict(dt_model, test_data, type = "class")
dt_confusion <- confusionMatrix(dt_predictions, test_data$NObeyesdad)
print(dt_confusion)

# Regression Model
regression_data <- obesity %>%
  select(Age, Height, Weight, FCVC, NCP, CH2O, FAF, TUE, BMI)
set.seed(123)
reg_train_index <- createDataPartition(regression_data$BMI, p = 0.8, list = FALSE)
reg_train_data <- regression_data[reg_train_index, ]
reg_test_data <- regression_data[-reg_train_index, ]

# Train and Evaluate Regression Model
reg_model <- lm(BMI ~ ., data = reg_train_data)
summary(reg_model)

# Predict BMI
reg_predictions <- predict(reg_model, reg_test_data)
mse <- mean((reg_test_data$BMI - reg_predictions)^2)
rmse <- sqrt(mse)
cat("Regression Model RMSE: ", rmse, "\n")

par(mfrow = c(2, 2))  # Set up the plotting area
plot(reg_model)

# Save Models
saveRDS(nn_model, file = "/Users/payalbhandwalkar/Downloads/nn_model.rds")
saveRDS(dt_model, file = "/Users/payalbhandwalkar/Downloads/dt_model.rds")
saveRDS(reg_model, file = "/Users/payalbhandwalkar/Downloads/reg_model.rds")

# Predict for New Sample
new_sample <- data.frame(
  Age = scale(25, center = attr(scaled_features, "scaled:center")["Age"], scale = attr(scaled_features, "scaled:scale")["Age"]),
  BMI = scale(23, center = attr(scaled_features, "scaled:center")["BMI"], scale = attr(scaled_features, "scaled:scale")["BMI"]),
  FAF = scale(1, center = attr(scaled_features, "scaled:center")["FAF"], scale = attr(scaled_features, "scaled:scale")["FAF"]),
  Height = scale(1.70, center = attr(scaled_features, "scaled:center")["Height"], scale = attr(scaled_features, "scaled:scale")["Height"]),
  Weight = scale(65, center = attr(scaled_features, "scaled:center")["Weight"], scale = attr(scaled_features, "scaled:scale")["Weight"]),
  FCVC = 3,
  NCP = 3,
  CH2O = 3,
  TUE = 2,
  Gender = factor("Female", levels = levels(obesity$Gender)),
  family_history_with_overweight = factor("no", levels = levels(obesity$family_history_with_overweight)),
  FAVC = factor("yes", levels = levels(obesity$FAVC)),
  CAEC = factor("Sometimes", levels = levels(obesity$CAEC)),
  SMOKE = factor("no", levels = levels(obesity$SMOKE)),
  SCC = factor("no", levels = levels(obesity$SCC)),
  CALC = factor("Sometimes", levels = levels(obesity$CALC)),
  MTRANS = factor("Automobile", levels = levels(obesity$MTRANS))
)
new_sample_prediction <- predict(nn_model, new_sample, type = "class")
cat("Predicted Obesity Level: ", new_sample_prediction, "\n")





