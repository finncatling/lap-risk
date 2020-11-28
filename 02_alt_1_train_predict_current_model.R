# Loads in preprocessed data for each train and test fold,
# fits random-intercept logistic model on each train fold,
# generates predicted probabilities on the corresponding test fold
# and save predicted probabilities as feather for later analysis in Python

library(lme4)
library(arrow)
library(glue)


setwd('~/Documents/DS130/share/lap-risk/outputs/current_model/feather')


print("Calculating number of train-test splits...")
n_splits <- length(list.files('train'))
print(glue("There are {n_splits} splits."))


print("Constructing formula string for random-intercept logistic model...")
train <- read_feather('train/0_train.feather')
column_names <- names(train)
fixed_effects <- column_names[column_names != 'Target']
fixed_effects <- fixed_effects[fixed_effects != 'HospitalId']
fe_formula <- paste(
  c(
    'Target',
    paste(fixed_effects, collapse=' + ')
  ),
  collapse=' ~ '
)
me_formula <- paste(c(fe_formula, '(1 | HospitalId)'), collapse=' + ')
print(glue("Formula is {me_formula}"))


print("Beginning model fitting and prediction for each train-test split")
for (i in 0 : n_splits - 1){
    train <- read_feather(glue('train/{i}_train.feather'))
    test <- read_feather(glue('test/{i}_test.feather'))

    train$HospitalId <- as.factor(train$HospitalId)
    test$HospitalId <- as.factor(test$HospitalId)

    # Fit model
    me_logreg <- glmer(
      me_formula,
      data=train,
      family='binomial',
      nAGQ=0  # Use less-exact parameter estimation (other options are very slow)
    )

    # Generate predicted probabilities on test fold
    me_y_pred <- predict(
      me_logreg,
      re.form=NA,  # Disregard random intercepts during prediction
      newdata=test,
      type='response'
    )
    me_y_pred_df <- data.frame(me_y_pred)

    write_feather(
      me_y_pred_df,
      glue('y_pred/{i}_y_pred.feather'),
      compression='uncompressed'
    )

    print(glue("Finished split {i}..."))
}
print("Done.")
