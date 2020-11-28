library(lme4)
library(arrow)

setwd('~/Documents/DS130/share/lap-risk/outputs/current_model/feather')

# Generate model formula
train <- read_feather('feather/train.feather')
test <- read_feather('feather/test.feather')

train$HospitalId <- as.factor(train$HospitalId)
test$HospitalId <- as.factor(test$HospitalId)

columns = names(train)
fixed_effects = columns[columns != 'Target']
fixed_effects = fixed_effects[fixed_effects != 'HospitalId']
fe_formula = paste(
  c(
    'Target',
    paste(fixed_effects, collapse=' + ')
    ),
  collapse=' ~ ')


# random intercept logistic model
me_formula = paste(c(fe_formula, '(1 | HospitalId)'), collapse=' + ')
me_logreg <- glmer(me_formula, data=train, family='binomial', nAGQ=0, verbose=2)
print(me_logreg)
me_y_pred = predict(me_logreg, re.form=NA, newdata=test, type='response')
me_y_pred_df = data.frame(me_y_pred)

write_feather(
  me_y_pred_df,
  'feather/glmer_y_pred.feather',
  compression='uncompressed'
)
