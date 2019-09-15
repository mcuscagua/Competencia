library(olsrr)
library(BMA)

rawdata <- read.csv(file="E:/OneDrive - CELSIA S.A E.S.P/Maestría/Metodos estadisticos avanzados/Competencia_Camila/databinarystudents.csv", header=TRUE, sep=",")
data <- rawdata[,-c(1,35,36)]

X_train_scaled = scale(data[,-c(1,34,35)])

X_tot_scaled = cbind(data$y,X_train_scaled)

X <- as.data.frame(X_tot_scaled)

# X_test_scaled = scale(X_test, center=attr(X_train_scaled, "scaled:center"), 
#                       scale=attr(X_train_scaled, "scaled:scale"))

model <- bic.glm(f = V1~., data = X,glm.family = binomial(),maxCol = 35,
                 strict = FALSE, OR = 20, OR.fix = 2, nbest = 15, dispersion = NULL,
                 factor.type = FALSE, factor.prior.adjust = FALSE, occam.window = TRUE,
                 call = NULL)
summary(model)