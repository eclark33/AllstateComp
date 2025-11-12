


# libraries 
library(vroom)
library(dbarts)
library(embed)

# read in data 
train_data <- vroom("/Users/eliseclark/Documents/Fall 2025/Stat 348/AllstateComp/train.csv")
testData <- vroom("/Users/eliseclark/Documents/Fall 2025/Stat 348/AllstateComp/test.csv")


# log transformation on loss
train_data <- train_data %>%
  mutate(log_loss = log1p(loss))

########################## BART MODEL ##########################
# BART model
bart_mod <- parsnip::bart(mode = "regression") %>%
  set_engine("dbarts") %>%
  set_args(trees = tune())  


# recipe
bart_recipe <- recipe(log_loss ~ . , data = train_data) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(log_loss)) %>%
  step_pca(all_predictors(), threshold = 0.87) %>%
  step_normalize(all_numeric_predictors)
  
prepped_bart_recipe <- prep(bart_recipe)
bart_data <- bake(prepped_bart_recipe, new_data = train_data)

# workflow  
bart_workflow <- workflow() %>%
  add_recipe(bart_recipe) %>%
  add_model(bart_mod)

# grid of values to tune 
grid_of_tuning_params <- grid_regular(
  trees(range = c(50, 200)),
  levels = 5)


# split data for cv & run it 
folds <- vfold_cv(log_data, v = 5, repeats=1)

CV_results <- bart_workflow %>%
  tune_grid(resamples = folds,
            grid = grid_of_tuning_params,
            metrics = metric_set(mae))


# find best tuning parameters
bestTune <- CV_results %>%
  select_best(metric = "mae")

## finalize the workflow & fit 
final_wf <- bart_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data = train_data) 

## predictions
bart_preds <- final_wf %>% predict(new_data = testData)
  
  
############### PENALIZED LINEAR REGRESSION ##################





  
  





############## BOOSTED TREES ############








