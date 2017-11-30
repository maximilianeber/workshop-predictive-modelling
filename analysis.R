# TITLE: Introduction to Predictive Modelling
# AUTHORS: fabian.fraenz@quantco.com, maximilian.eber@quantcom
# DATE: 21 Nov 2017
# DATA SOURCE: Kaggle Porto Seguro Competition

# Setup -------------------------------------------------------------------

rm(list = ls())
library(readr)
library(dplyr)
library(xgboost)
library(ggplot2)
library(tidyr)
library(pROC)
library(forcats)

# Load data and recode
if (!file.exists("train.csv")) {
  message("Data not found. Please download and unzip from https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/download/train.7z")
}
data <- read_csv("train.csv") %>% 
  mutate_at(vars(ends_with("_cat")),   funs(factor(., exclude = -1))) %>%
  mutate_at(vars(ends_with("_bin")),   funs(if_else(. == -1, as.integer(NA), as.integer(.)))) %>% 
  mutate_at(vars(matches("_[0-9]+$")), funs(if_else(. == -1, as.numeric(NA), as.numeric(.)))) %>% 
  select(-id)

# Summary Statistics ------------------------------------------------------

# List of variables
data %>% summary()

# Histogram of outcome variable
data %>% ggplot(aes(x = target)) + geom_histogram()
  
# Histograms of selected predictors

# - Example (continuous variable)
data %>% ggplot(aes(x = ps_reg_02)) + geom_histogram(bins = 50)

# - Example (categorical variable):
data %>% ggplot(aes(x = ps_car_01_cat)) + geom_bar()

# TODO: Select a few important variables and plot their distribution

# Correlation of selected predictors with outcome variables

# - Example (continuous variable)
data %>% 
  ggplot(aes(x = ps_reg_02, y = target)) + 
  stat_summary_bin(fun.data = "mean_se", bins = 50)

# - Example (categorical variable):
data %>% 
  ggplot(aes(x = ps_car_01_cat, y = target)) + 
  stat_summary(fun.data = "mean_se")

# TODO: Select a few important variables and plot their correlation with the outcome variable

# Basic Linear Model ------------------------------------------------------------

# Problem: How to deal with missing values?
data %>% count(is.na(ps_car_01_cat))

# Solution: Code explicitly
data_glm <- data %>% 
  select(target, ps_reg_02, ps_car_01_cat) %>% 
  mutate(
    # ps_car_01_cat (categorical)
    ps_car_01_cat = forcats::fct_explicit_na(ps_car_01_cat),
    # ps_reg_02 (continuous)
    ps_reg_02_missing = if_else(is.na(ps_reg_02), 1, 0),
    ps_reg_02 = if_else(is.na(ps_reg_02), 0, ps_reg_02)
  )
  
# Here, we build a simple linear model based on a few features
glm0 <- glm(data = data_glm, formula = target ~ ps_reg_02 + ps_car_01_cat, family = "binomial") # ignore missing indicator
# glm0 <- glm(data = data_glm, formula = target ~ ps_reg_02 + ps_reg_02_missing + ps_car_01_cat, family = "binomial") # including missing indicator
summary(glm0)

# Evaluate in-sample accuracy (metric: AUC)
predicted <- predict(object = glm0, newdata = data_glm, type = "response")
actual <- data_glm %>% pull(target)
auc(actual, predicted)
roc(actual, predicted, plot = TRUE, grid = TRUE)

# TODO: Explain intuitively what this metric does. Why is better than using, say, accuracy?

# Plot predicted vs. actual along a few variables

# Example: 
data_glm %>% 
  mutate(predicted = predict(object = glm0, newdata = ., type = "response")) %>% 
  ggplot(aes(x = ps_reg_02)) + 
  stat_summary_bin(aes(y = target, color = "actual"), fun.data = "mean_se") + 
  stat_summary_bin(aes(y = predicted, color = "predicted"), fun.data = "mean_se")

# TODO: Evaluate the fit of your model along a few more variables. What does this imply for feature engineering?
# - Hint: The numnber of missing values per row is known to be predictive.

# TODO: Avoid overfitting
# - Split the sample in train (50%) and test (50%). Remember to set a seed for reproducability
# - Build the model on the training data only
# - What is the AUC on the training data?  What is the AUC on the test data? Why do they differ?

# TODO: Add a few variables to your model. Does the training AUC improve as you add variables? How about the testing AUC?

# Boosted Trees -----------------------------------------------------------------

# Prepare data for XGBOOST
# - This is necessary to comply with the inner workings of the packages (not native R, optimized for performance)
options(na.action = "na.pass") # keep NAs

# create model matrix, vector with labels, vector with sample (train, test)

dat_x           <- model.matrix(data = data, object = target ~ ps_reg_02 + ps_car_01_cat)
dat_y           <- data %>% pull(target)

# This creates a "one-hot encoded" matrix:
head(dat_x) 

# Create indicators for train-test split
set.seed(281701)
select_train = sample(c(TRUE, FALSE), prob = c(0.5, 0.5), replace = TRUE, size = nrow(dat_x))
select_valid = !select_train

# create data sets for training and validation
train_xgb <- xgb.DMatrix(data = dat_x[select_train,], label = dat_y[select_train])
valid_xgb <- xgb.DMatrix(data = dat_x[select_valid,], label = dat_y[select_valid])

xgb0 <- xgb.train(
  data                  = train_xgb,
  watchlist             = list(train = train_xgb, validation = valid_xgb),
  objective             = "binary:logistic",
  eval_metric           = "auc",
  nrounds               = 100,
  print_every_n         = 25,
  verbose               = 1,
  eta                   = 0.1,
  gamma                 = 0,
  nthread               = parallel::detectCores(),
  seed                  = 182064,
  grow_policy           = "lossguide")

# Print evaluation log
xgb0$evaluation_log %>% 
  ggplot(aes(x = iter)) + 
  geom_line(aes(y = train_auc, color = "train")) +
  geom_line(aes(y = validation_auc, color = "validation"))

# TODO: Estimate the model for various other hyperparameters (e.g. eta, nrounds, gamma, max_depth). 
# - How does the evaluation log change?
# - What are good parameters for eta and nrounds?

# Print feature importance
xgb0 %>% 
  xgb.importance(feature_names = colnames(dat_x)) %>% 
  xgb.ggplot.importance()

# Training:
data_glm[select_train, ] %>%  # select training part only
  mutate(predicted = predict(object = xgb0, newdata = train_xgb, type = "response")) %>% 
  ggplot(aes(x = ps_reg_02)) + 
  stat_summary_bin(aes(y = target, color = "actual"), fun.data = "mean_se") + 
  stat_summary_bin(aes(y = predicted, color = "predicted"), fun.data = "mean_se")

# Testing:
data_glm[select_valid, ] %>%  # select training part only
  mutate(predicted = predict(object = xgb0, newdata = valid_xgb, type = "response")) %>% 
  ggplot(aes(x = ps_reg_02)) + 
  stat_summary_bin(aes(y = target, color = "actual"), fun.data = "mean_se") + 
  stat_summary_bin(aes(y = predicted, color = "predicted"), fun.data = "mean_se")

# TODO: Choose a high value (e.g. 5) for gamma and see how these plots change. Why?

# TODO: Add new feature (e.g. number of NAs per row)
# - Does it improve accuracy?
# - Where does it show up in feature importance?

# TODO: Feature Engineering
# - Build new features
# - Evaluate distribution and correlation with outcome variable
# - Add it to your boosted tree
# - Where does it show up in feature importance?
# - Does accuracy improve?
#
# Hint: try adding the number of NAs per row in addition to the two variables above:
# dat_x <- data %>% 
#   mutate(number_nas = rowSums(is.na(as_data_frame(.) %>% select(-target)))) %>% 
#   model.matrix(object = target ~ ps_reg_02 + ps_car_01_cat + number_nas)

# Additional Tasks ----------------------------------------------------

# TODO: Build a model with all features
# TODO: Cross-validate results
# TODO: Evaluate partial dependency for some variables