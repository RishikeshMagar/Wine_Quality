load("~/WINE_QUALITY/wineQuality.Rdata")
set.seed(1)
df1 = predictors
df2 = response
df = cbind(df1,df2)
s = sample(nrow(df),0.7*nrow(df))
pred.train = df[s,1:12]                                  # don't include redshift or redshift error!
pred.test  = df[-s,1:12]
resp.train = df[s,13]
resp.test  = df[-s,13]

out.log = glm(resp.train~.,data=pred.train,family=binomial)
resp.prob = predict(out.log,newdata=pred.test,type="response")
resp.pred = ifelse(resp.prob>0.5,"GOOD","BAD")
mean(resp.pred!=resp.test)

table(resp.pred)
summary(out.log)

#generate white
set.seed(1)
library(magrittr)
library(dplyr)
df.white = df %>% filter(.,type=="white")
df.red = df%>% filter(.,type == "red")
# defining white variables
s.white = sample(nrow(df.white),0.7*nrow(df.white))
pred.train.white = df.white[s.white,1:11]                                  # don't include redshift or redshift error!
pred.test.white  = df.white[-s.white,1:11]
resp.train.white = df.white[s.white,13]
resp.test.white  = df.white[-s.white,13]

# Defining red varibles
s.red = sample(nrow(df.red),0.7*nrow(df.red))
pred.train.red = df.red[s.red,1:11]                                  # don't include redshift or redshift error!
pred.test.red  = df.red[-s.red,1:11]
resp.train.red = df.red[s.red,13]
resp.test.red  = df.red[-s.red,13]

# output for white
out.log.white = glm(resp.train.white~.,data=pred.train.white,family=binomial)
resp.prob.white = predict(out.log.white,newdata=pred.test.white,type="response")
resp.pred.white = ifelse(resp.prob.white>0.5,"GOOD","BAD")
mean(resp.pred.white!=resp.test.white)
table(resp.pred.white)
summary(out.log.white)

#Red separation
out.log.red = glm(resp.train.red~.,data=pred.train.red,family=binomial)
resp.prob.red = predict(out.log.red,newdata=pred.test.red,type="response")
resp.pred.red = ifelse(resp.prob.red>0.5,"GOOD","BAD")
mean(resp.pred.red!=resp.test.red)
table(resp.pred.red)
summary(out.log.red)

# best subset 
set.seed(101)
if ( require(bestglm) == FALSE ) {
  install.packages("bestglm",repos="https://cloud.r-project.org")
  library(leaps)
}

names(df)[13] = "y"                                           # necessary tweak for bestglm: response is "y"
set.seed(1)
s = sample(nrow(df),0.7*nrow(df))
data.train = df[s,c(1:11,13)]                                  # don't include redshift or redshift error!
data.test  = df[-s,c(1:11,13)]

out.glm = bestglm(data.train,family=binomial,IC="BIC")
out.glm$BestModel

predict.bestglm = function(object,data.train,data.test) {
  form  = formula(object$BestModel$terms)
  out.log = glm(form,data=data.train,family=binomial)
  return(predict(out.log,newdata=data.test,type="response"))
}
resp.prob = predict.bestglm(out.glm,data.train,data.test)
resp.pred = ifelse(resp.prob>0.5,"GOOD","BAD")
mean(resp.pred!=data.test$y)

# glm for red and white
names(df.white)[13] = "y"                                           # necessary tweak for bestglm: response is "y"
set.seed(1)
s = sample(nrow(df.white),0.7*nrow(df.white))
data.train.white = df.white[s,c(1:11,13)]                                  # don't include redshift or redshift error!
data.test.white  = df.white[-s,c(1:11,13)]

out.glm.white = bestglm(data.train.white,family=binomial,IC="BIC")
out.glm.white$BestModel

predict.bestglm.white = function(object,data.train.white,data.test.white) {
  form.white  = formula(object$BestModel$terms)
  out.log.white = glm(form,data=data.train.white,family=binomial)
  return(predict(out.log.white,newdata=data.test.white,type="response"))
}
resp.prob.white = predict.bestglm(out.glm.white,data.train.white,data.test.white)
resp.pred.white = ifelse(resp.prob.white>0.6,"GOOD","BAD")
mean(resp.pred.white!=data.test.white$y)

# Red glm 
names(df.red)[13] = "y"                                           # necessary tweak for bestglm: response is "y"
set.seed(1)
s = sample(nrow(df.red),0.7*nrow(df.red))
data.train.red = df.red[s,c(1:11,13)]                                  # don't include redshift or redshift error!
data.test.red  = df.red[-s,c(1:11,13)]

out.glm.red = bestglm(data.train.red,family=binomial,IC="BIC")
out.glm.red$BestModel

predict.bestglm.red = function(object,data.train.red,data.test.red) {
  form.red  = formula(object$BestModel$terms)
  out.log.red = glm(form,data=data.train.red,family=binomial)
  return(predict(out.log.red,newdata=data.test.red,type="response"))
}
resp.prob.red = predict.bestglm(out.glm.red,data.train.red,data.test.red)
resp.pred.red = ifelse(resp.prob.red>0.6,"GOOD","BAD")
mean(resp.pred.red!=data.test.red$y)

if ( require(glmnet) == FALSE ) {
  install.packages("glmnet",repos="https://cloud.r-project.org")
  library(glmnet)
}
# Red LASSO
set.seed(1)
s = sample(nrow(df.red),0.75*nrow(df.red))
data.train.red.lasso = df.red[s,c(1:11)]                                  # don't include redshift or redshift error!
data.test.red.lasso  = df.red[-s,c(1:11)]
pred.train.lasso.red = data.train.red.lasso
pred.test.lasso.red = data.test.red.lasso
resp.train.lasso.red = df.red[s,c(13)]
resp.test.lasso.red = df.red[-s,c(13)]

x = model.matrix(resp.train.lasso.red~.,pred.train.lasso.red)[,-1]
y = resp.train.lasso.red
out.lasso = glmnet(x,y,alpha=1,family="binomial")  # note the "family" argument
plot(out.lasso,xvar="lambda")

cv = cv.glmnet(x,y,alpha=1,family="binomial")
plot(cv)

x.test    = model.matrix(resp.test.lasso.red~.,pred.test.lasso.red)[,-1]
resp.prob = predict(out.lasso,s=cv$lambda.min,newx=x.test,type="response")
resp.pred = ifelse(resp.prob>0.5,"GOOD","BAD")
mean(resp.pred!=resp.test.lasso.red) # basically the same as for logistic regressionme


# xgboost

if ( require(xgboost) == FALSE ) {
  install.packages("xgboost",repos="http://cloud.r-project.org")
  library(xgboost)
}
set.seed(1)
s = sample(nrow(df),0.7*nrow(df))
data.train.boost = df[s,c(1:11)]
data.test.boost = df[-s,c(1:11)]
resp.train.boost = df[s,c(13)]
resp.test.boost = df[-s,c(13)]

{
  resp.train.change = ifelse(resp.train.boost =="GOOD",1 ,0)#ifelse(resp.train.boost == "BAD",0))
}
{
  resp.test.change = ifelse(resp.test.boost=="GOOD",1,0)#ifelse(resp.test.boost=="GOOD",1))
}
#if ( resp.train.boost == "GOOD" ) 
## resp.train.boost = 1
#}
#if ( resp.train.boost == "BAD" ) 
#{
 # resp.train.boost = 0
#}

#if (resp.test.boost == "GOOD")
#{
 # resp.test.boost  = 1
#}
#if (resp.test.boost == "BAD")
#{
 # resp.test.boost  = 0
#}
# combine predictors and response into a special type of R structure
train = xgb.DMatrix(data=as.matrix(data.train.boost),label=resp.train.change)
test  = xgb.DMatrix(data=as.matrix(data.test.boost),label=resp.test.change)
# perform cross-validation to tune parameters
#   objective="reg:linear" <-> use the mean-squared error to assess models
#   nrounds=30             <-> the maximum number of stumps to grow
#   nfold=5                <-> the number of CV folds (5 or 10, usually)

out.cv = xgb.cv(params=list(objective="binary:logistic"),train,nrounds=100,nfold=5,verbose=0)
# if the optimal number of trees is equal to nrounds, increase nrounds and try again
cat("The optimal number of trees is ",which.min(out.cv$evaluation_log$test_error_mean))
out.xgb = xgboost(train,nrounds=which.min(out.cv$evaluation_log$test_error_mean),params=list(objective="binary:logistic"),verbose=0)
pred    = predict(out.xgb,newdata=test)  # at least this looks similar to before...
pred = ifelse(pred>0.5,1,0)
mean((pred!= resp.test.change))

importance = xgb.importance(feature_names=names(data.train.boost),model=out.xgb)
xgb.plot.importance(importance_matrix=importance)
