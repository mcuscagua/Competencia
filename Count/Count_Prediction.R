library(rio)

Train_Data = import('count_train.csv')
Test_Data = import('count_test.csv')

Y = Train_Data[,"yC"]
Yt = Test_Data[,"yC"]
X = Train_Data[,c("x3","x13","x14","x22","x24","x25")]
X = data.frame(apply(X, 2, function(x){(x-mean(x))/sd(x)}))
Xt = Test_Data[,c("x3","x13","x14","x22","x24","x25")]
Xt = data.frame(apply(Xt, 2, function(x){(x-mean(x))/sd(x)}))


poisson_reg = glm(formula = yC ~ x3+x13+x14+x22+x24+x25, data = Train_Data, family = poisson)
predict(poisson_reg, newdata = Xt)
Yt