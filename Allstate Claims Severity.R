


# libraries 
library(vroom)
library(dbarts)
library(embed)
library(tidymodels)
library(tidyverse)
library(lightgbm)
library(bonsai)

# read in data 
train_data <- vroom("/Users/eliseclark/Documents/Fall 2025/Stat 348/AllstateComp/train.csv")
testData <- vroom("/Users/eliseclark/Documents/Fall 2025/Stat 348/AllstateComp/test.csv")

# set factors in both test and validation sets
train_data <- train_data %>%
  mutate(across(starts_with("cat"), as.factor))
testData <- testData %>%
  mutate(across(starts_with("cat"), as.factor))


# log transformation on loss
train_data <- train_data %>%
  mutate(log_loss = log1p(loss)) %>%
  select(-loss)

########################## BART MODEL ########################## 
# BART model
bart_mod <- parsnip::bart(mode = "regression") %>%
  set_engine("dbarts") %>%
  set_args(trees = tune())  


# recipe
bart_recipe <- recipe(log_loss ~ ., data = train_data) %>%
  step_other(all_nominal_predictors(), threshold = 0.01) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(log_loss))


# workflow  
bart_workflow <- workflow() %>%
  add_recipe(bart_recipe) %>%
  add_model(bart_mod)

# grid of values to tune 
grid_of_tuning_params <- grid_regular(
  trees(range = c(50, 200)),
  levels = 5)


# split data for cv & run it 
folds <- vfold_cv(train_data, v = 3, repeats = 1)

CV_results <- bart_workflow %>%
  tune_grid(resamples = folds,
            grid = grid_of_tuning_params,
            metrics = metric_set(mae), 
            control = control_grid(verbose = TRUE)) 


# find best tuning parameters
bestTune <- CV_results %>%
  select_best(metric = "mae")

## finalize the workflow & fit 
final_wf <- bart_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data = train_data) 

## predictions
bart_preds <- final_wf %>%
  predict(new_data = testData) 

bart_preds <- bart_preds %>%
  mutate(loss = expm1(.pred))

submission <- tibble(
  id = testData$id,
  loss = bart_preds$loss)
  

vroom_write(x = submission, file = "/Users/eliseclark/Documents/Fall 2025/Stat 348/allstate1.csv", delim=",")




  
############### PENALIZED LINEAR REGRESSION ##################



###################### BOOSTED TREES ####################       *Best Model 

# model 
boost_model <- boost_tree(tree_depth = tune(),
                          trees = tune(),
                          learn_rate = tune()) %>%
              set_engine("lightgbm") %>%
              set_mode("regression")


# recipe
boost_recipe <- recipe(log_loss ~ ., data = train_data) %>%
  step_other(all_nominal_predictors(), threshold = 0.01) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(log_loss))


# workflow  
boost_workflow <- workflow() %>%
  add_recipe(boost_recipe) %>%
  add_model(boost_model)

# grid of values to tune 
grid_of_tuning_params <- grid_regular(
  trees(range = c(50, 200)),
  tree_depth(range = c(3, 10)),
  learn_rate(range = c(0.1, 0.5)),
  levels = 5)


# split data for cv & run it 
folds <- vfold_cv(train_data, v = 3, repeats = 1)

CV_results <- boost_workflow %>%
  tune_grid(resamples = folds,
            grid = grid_of_tuning_params,
            metrics = metric_set(mae), 
            control = control_grid(verbose = TRUE)) 


# find best tuning parameters
bestTune <- CV_results %>%
  select_best(metric = "mae")

## finalize the workflow & fit 
final_wf <- boost_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data = train_data) 

## predictions
boost_preds <- final_wf %>%
  predict(new_data = testData) %>%
  mutate(loss = expm1(.pred))


boost_submission <- tibble(
  id = testData$id,
  loss = boost_preds$loss)

vroom_write(x = boost_submission, file = "/Users/eliseclark/Documents/Fall 2025/Stat 348/allstate2.csv", delim=",")


