library(ISLR2)
library(dplyr)
library(caret)
library(glmnet)
library(leaps)

summary(College) 

hist(College$Accept)

College$Private = ifelse(College$Private == "Yes", 1,0)

College$Accept=log(College$Accept)

data_scaled = College %>% mutate_at(c("Private",
"Apps", "Enroll", "Top10perc", "Top25perc",
"F.Undergrad", "P.Undergrad", "Outstate",
"Room.Board", "Books", "Personal", "PhD",
"Terminal", "S.F.Ratio", "perc.alumni",
"Expend", "Grad.Rate"), ~(scale(.) %>% as.vector))

summary(data_scaled)

stn_train = sample(nrow(data_scaled),0.80 * nrow(data_scaled))

train_stn = data_scaled[stn_train,]

stn_test = sample(nrow(data_scaled),0.20 * nrow(data_scaled))

test_stn = data_scaled[-stn_train, ]

st_mlr = lm(Accept ~ . , data = train_stn)

summary(st_mlr)

predic_stn_mlr = predict(st_mlr, newdata = test_stn)

summary(predic_stn_mlr)

par(mfrow = c(2,2))

plot(st_mlr)

reg.fit.full = regsubsets(Accept ~ ., data = train_stn, nvmax = 17)

reg.summary = summary(reg.fit.full)

which.min(reg.summary$rss)

which.max(reg.summary$adjr2)

which.min(reg.summary$cp)

which.min(reg.summary$bic)

plot(reg.summary$rss, type = "l")

par(mfrow = c(2, 2))

plot(reg.summary$rss , xlab = "Number of Variables",
     ylab = "RSS", type = "l")
plot(reg.summary$adjr2 , xlab = "Number of Variables",
     ylab = "Adjusted RSq", type = "l")

coef(reg.fit.full, 9)

reg.fit.fwd = regsubsets(Accept ~ ., data = train_stn,
                         nvmax = 17, method = "forward")

reg_fwd = summary(reg.fit.fwd)

which.min(reg_fwd$rss)

which.max(reg_fwd$adjr2)

which.min(reg_fwd$cp)

which.min(reg_fwd$bic)

plot(reg_fwd$rss, type = "l")

par(mfrow = c(2, 2))

reg.fit.bwd = regsubsets(Accept ~ ., data = train_stn,
                         nvmax = 17, method = "backward")

reg_bwd = summary(reg.fit.bwd)

which.min(reg_bwd$rss)

which.max(reg_bwd$adjr2)

which.min(reg_bwd$cp)

which.min(reg_bwd$bic)

plot(reg_bwd$rss, type = "l")

par(mfrow = c(2, 2))

plot(reg_bwd$rss , xlab = "Number of Variables",
     ylab = "RSS", type = "l")
plot(reg_bwd$adjr2 , xlab = "Number of Variables",
     ylab = "Adjusted RSq", type = "l")

coef(reg.fit.bwd, 9)

x = model.matrix(Accept ~., data_scaled)[,-1]
y = data_scaled$Accept

x.train = x[stn_train , ]
y.train = y[stn_train]
length(y.train)

x.test = x[stn_test , ]
y.test = y[stn_test]
length(y.test)

grid = 10ˆseq(10, -2, length = 100)

cv.out = cv.glmnet(x[stn_train , ], y[stn_train], alpha = 0)

cv.out$lambda

bestlam = cv.out$lambda.min

bestlam

ridge_m = glmnet(x[stn_train , ], y[stn_train], alpha = 0, lambda = bestlam)

coef(ridge_m)

ridge_p <- predict(ridge_m , s = bestlam, newx = x[stn_test , ])

mean(( ridge_p - y.test)ˆ2)

cv.out_l = cv.glmnet(x[stn_train , ], y[stn_train], alpha = 1)

cv.out_l$lambda

bestlam_l = cv.out_l$lambda.min

bestlam_l

lasso_m = glmnet(x[stn_train , ], y[stn_train], alpha = 1, lambda = bestlam)

coef(lasso_m)

lasso_p <- predict(lasso_m , s = bestlam, newx = x[stn_test , ])

mean(( lasso_p - y.test)ˆ2)
