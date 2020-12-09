library(MASS)
library(doMC)
library(glmnet)
library(rms)
library(facetscales)
library(tidyverse)
library(apricom)
library(patchwork)

generate_data <- function(NN, n.true.predictors = n.true.predictors, n.noise.predictors = n.noise.predictors, cor0 = 0.1, cor1 = 0.05, beta.0 = 0){
  
  n.predictors <- n.true.predictors + n.noise.predictors
  mu0 <- rep(0, n.predictors)
  
  # Specify correlation matrix
  Sigma0 <- matrix(0, nrow = n.predictors,  ncol = n.predictors)
  Sigma0[1:n.true.predictors, 1:n.true.predictors] <- cor0
  Sigma0[(n.true.predictors+1):n.predictors, (n.true.predictors+1):n.predictors] <- cor1
  diag(Sigma0) <- 1.0
  
  x <- mvrnorm(NN, mu0, Sigma0)
  
  beta <- c(0.5, 0.3, 0.3, 0.25, 0.25, rep(0, n.noise.predictors))
  
  y <- runif(NN) < 1 / (1 + exp(-beta.0 - x %*% beta))
  
  DATA   <- data.frame(x)
  DATA$y <- y * 1
  DATA
}

B <- 500
N <- c(100, 200, 300, 400, 500, 600, 700, 800, 900, 1000)

n.true.predictors  <- 5
n.noise.predictors <- 15

heur.shrink        <- matrix(ncol = length(N), nrow = B)
boot.shrink        <- matrix(ncol = length(N), nrow = B)
cv.ridge           <- matrix(ncol = length(N), nrow = B)
cv.elasticnet      <- matrix(ncol = length(N), nrow = B)
cv.lasso           <- matrix(ncol = length(N), nrow = B)

c.ML               <- matrix(ncol = length(N), nrow = B)
c.boot             <- matrix(ncol = length(N), nrow = B)
c.heur             <- matrix(ncol = length(N), nrow = B)
c.ridge            <- matrix(ncol = length(N), nrow = B)
c.elasticnet       <- matrix(ncol = length(N), nrow = B)
c.lasso            <- matrix(ncol = length(N), nrow = B)
c.ML.VAL           <- matrix(ncol = length(N), nrow = B)
c.boot.VAL         <- matrix(ncol = length(N), nrow = B)
c.heur.VAL         <- matrix(ncol = length(N), nrow = B)
c.ridge.VAL        <- matrix(ncol = length(N), nrow = B)
c.elasticnet.VAL   <- matrix(ncol = length(N), nrow = B)
c.lasso.VAL        <- matrix(ncol = length(N), nrow = B)

R2.ML              <- matrix(ncol = length(N), nrow = B)
R2.boot            <- matrix(ncol = length(N), nrow = B)
R2.heur            <- matrix(ncol = length(N), nrow = B)
R2.ridge           <- matrix(ncol = length(N), nrow = B)
R2.elasticnet      <- matrix(ncol = length(N), nrow = B)
R2.lasso           <- matrix(ncol = length(N), nrow = B)
R2.ML.VAL          <- matrix(ncol = length(N), nrow = B)
R2.boot.VAL        <- matrix(ncol = length(N), nrow = B)
R2.heur.VAL        <- matrix(ncol = length(N), nrow = B)
R2.ridge.VAL       <- matrix(ncol = length(N), nrow = B)
R2.elasticnet.VAL  <- matrix(ncol = length(N), nrow = B)
R2.lasso.VAL       <- matrix(ncol = length(N), nrow = B)

cal.intercept.fit.val.ML         <- matrix(ncol = length(N), nrow = B)
cal.intercept.fit.val.boot       <- matrix(ncol = length(N), nrow = B)
cal.intercept.fit.val.heur       <- matrix(ncol = length(N), nrow = B)
cal.intercept.fit.val.ridge      <- matrix(ncol = length(N), nrow = B)
cal.intercept.fit.val.elasticnet <- matrix(ncol = length(N), nrow = B)
cal.intercept.fit.val.lasso      <- matrix(ncol = length(N), nrow = B)

cal.slope.fit.val.ML         <- matrix(ncol = length(N), nrow = B)
cal.slope.fit.val.boot       <- matrix(ncol = length(N), nrow = B)
cal.slope.fit.val.heur       <- matrix(ncol = length(N), nrow = B)
cal.slope.fit.val.ridge      <- matrix(ncol = length(N), nrow = B)
cal.slope.fit.val.elasticnet <- matrix(ncol = length(N), nrow = B)
cal.slope.fit.val.lasso      <- matrix(ncol = length(N), nrow = B)

ML.converged      <- matrix(ncol = length(N), nrow = B)
betas.ML <- array(dim = c(B, n.true.predictors + n.noise.predictors + 1, length(N)))

registerDoMC(cores = 4)
set.seed(12412)
# Generate validation sample
DATA.VAL <- generate_data(NN = 5000, n.true.predictors = n.true.predictors, n.noise.predictors = n.noise.predictors)

for(j in 1:length(N)){
  cat("\n loop = ", j)
  for(i in 1:B){
    DATA <- generate_data(NN = N[j], n.true.predictors = n.true.predictors, n.noise.predictors = n.noise.predictors)
    
    ### Cross-validation to get lambda
    fit.glmnet.ridge <- cv.glmnet(x            = as.matrix(DATA[,1:(n.true.predictors+n.noise.predictors)]), 
                                  y            = DATA$y, 
                                  family       = 'binomial', 
                                  alpha        = 0, 
                                  nfolds       = 5,
                                  type.measure = 'deviance', 
                                  parallel     = T)
    
    fit.glmnet.elasticnet <- cv.glmnet(x            = as.matrix(DATA[,1:(n.true.predictors+n.noise.predictors)]), 
                                       y            = DATA$y, 
                                       family       = 'binomial', 
                                       alpha        = 0.5, 
                                       nfolds       = 5,
                                       type.measure = 'deviance', 
                                       parallel     = T)
    
    fit.glmnet.lasso <- cv.glmnet(x            = as.matrix(DATA[,1:(n.true.predictors+n.noise.predictors)]), 
                                  y            = DATA$y, 
                                  family       = 'binomial', 
                                  alpha        = 1, 
                                  nfolds       = 5,
                                  type.measure = 'deviance', 
                                  parallel     = T)
     
    ### Pull out lambda.min
    cv.ridge[i,j]      <- fit.glmnet.ridge$lambda.min
    cv.elasticnet[i,j] <- fit.glmnet.elasticnet$lambda.min
    cv.lasso[i,j]      <- fit.glmnet.lasso$lambda.min
    
    ### Get betas for lambda.min
    optimal.beta.ridge      <- as.numeric(predict(fit.glmnet.ridge,      type = 'coefficients', s = "lambda.min"))
    optimal.beta.elasticnet <- as.numeric(predict(fit.glmnet.elasticnet, type = 'coefficients', s = "lambda.min"))
    optimal.beta.lasso      <- as.numeric(predict(fit.glmnet.lasso,      type = 'coefficients', s = "lambda.min"))
    
    ### create lrm object with coefficients from the 3 penalised approaches
    fit.ridge      <- lrm(y ~ ., data = DATA, init = optimal.beta.ridge,      maxit = 1)
    fit.elasticnet <- lrm(y ~ ., data = DATA, init = optimal.beta.elasticnet, maxit = 1)
    fit.lasso      <- lrm(y ~ ., data = DATA, init = optimal.beta.lasso,      maxit = 1)
    
    ### fit maximum likelihood full model
    fit.glm      <- glm(y ~ ., data = DATA, family = 'binomial')
    fit.ML       <- lrm(y ~ ., data = DATA, init = coef(fit.glm), maxit=1)
    
    ### shrinkage (heuristic and bootstrap)
    DATA2  <- DATA
    DATA2  <- cbind(1, DATA)
    X.boot <- bootval(DATA2,     model = "logistic", int = T, int.adj = T, N = 250)
    X.heur <- shrink.heur(DATA2, model = "logistic", int = T, int.adj = T)
    
    boot.shrink[i, j] <- X.boot$lambda
    heur.shrink[i, j] <- X.heur$lambda
    
    x.boot <- X.boot$shrunk.coeff
    x.heur <- X.heur$shrunk.coeff
    
    fit.boot <- fit.glm
    fit.heur <- fit.glm
    
    fit.boot$coefficients <- x.boot
    fit.heur$coefficients <- x.heur
    
    fit.heur.lrm <- lrm(y ~ ., data = DATA, init = coef(fit.heur), maxit = 1)
    fit.boot.lrm <- lrm(y ~ ., data = DATA, init = coef(fit.boot), maxit = 1)
    
    ML.converged[i, j] <- fit.glm$converged
    if(ML.converged[i, j]) {
      betas.ML[i,,j] <- as.numeric(coef(fit.ML))
      c.ML[i, j] <- as.numeric(fit.ML$stats['C']) 
    } else {
      betas.ML[i, ,j] <- NA
      c.ML[i, j] <- NA
    }
    
    # get validation performance (ridge)
    if(is.na(as.numeric(coef(glm(DATA.VAL$y~predict(fit.ridge, DATA.VAL), family='binomial'))[2])) | as.numeric(coef(glm(DATA.VAL$y~predict(fit.ridge, DATA.VAL), family='binomial'))[2])>100) {
      cal.slope.fit.val.ridge[i, j] <- NA
      cal.intercept.fit.val.ridge[i, j] <- NA
      c.ridge.VAL[i, j]       <- NA
      R2.ridge.VAL[i, j]      <- NA
    } else {
      fit.ridge.VAL           <- lrm(DATA.VAL$y~predict(fit.ridge, DATA.VAL))
      fit.cal.VAL             <- glm(DATA.VAL$y~offset(predict(fit.ridge, DATA.VAL)), family='binomial')
      cal.slope.fit.val.ridge[i, j] <- as.numeric(coef(fit.ridge.VAL)[2])
      cal.intercept.fit.val.ridge[i, j] <- as.numeric(coef(fit.cal.VAL)[1])
      c.ridge.VAL[i, j]       <- as.numeric(fit.ridge.VAL$stats['C'])
      R2.ridge.VAL[i, j]      <- as.numeric(fit.ridge.VAL$stats['R2'])
    }
    
    # get validation performance (elasticnet)
    if(is.na(as.numeric(coef(glm(DATA.VAL$y~predict(fit.elasticnet, DATA.VAL), family='binomial'))[2])) | as.numeric(coef(glm(DATA.VAL$y~predict(fit.elasticnet, DATA.VAL), family='binomial'))[2])>100) {
      cal.slope.fit.val.elasticnet[i, j] <- NA
      cal.intercept.fit.val.elasticnet[i, j] <- NA
      c.elasticnet.VAL[i, j]       <- NA
      R2.elasticnet.VAL[i, j]      <- NA
    } else {
      fit.elasticnet.VAL           <- lrm(DATA.VAL$y~predict(fit.elasticnet, DATA.VAL))
      fit.cal.VAL                  <- glm(DATA.VAL$y~offset(predict(fit.elasticnet, DATA.VAL)), family='binomial')
      cal.slope.fit.val.elasticnet[i, j] <- as.numeric(coef(fit.elasticnet.VAL)[2])
      cal.intercept.fit.val.elasticnet[i, j] <- as.numeric(coef(fit.cal.VAL)[1])
      c.elasticnet.VAL[i, j]       <- as.numeric(fit.elasticnet.VAL$stats['C'])
      R2.elasticnet.VAL[i, j]      <- as.numeric(fit.elasticnet.VAL$stats['R2'])
    }
    
    # get validation performance (lasso)
    if(is.na(as.numeric(coef(glm(DATA.VAL$y~predict(fit.lasso, DATA.VAL), family='binomial'))[2])) | as.numeric(coef(glm(DATA.VAL$y~predict(fit.lasso, DATA.VAL), family='binomial'))[2])>100) {
      cal.slope.fit.val.lasso[i, j] <- NA
      cal.intercept.fit.val.lasso[i, j] <- NA
      c.lasso.VAL[i, j]       <- NA
      R2.lasso.VAL[i, j]      <- NA
    } else {
      fit.lasso.VAL           <- lrm(DATA.VAL$y~predict(fit.lasso, DATA.VAL))
      fit.cal.VAL             <- glm(DATA.VAL$y~offset(predict(fit.lasso, DATA.VAL)), family='binomial')
      cal.slope.fit.val.lasso[i, j] <- as.numeric(coef(fit.lasso.VAL)[2])
      cal.intercept.fit.val.lasso[i, j] <- as.numeric(coef(fit.cal.VAL)[1])
      c.lasso.VAL[i, j]       <- as.numeric(fit.lasso.VAL$stats['C'])
      R2.lasso.VAL[i, j]      <- as.numeric(fit.lasso.VAL$stats['R2'])
    }
    
    fit.ML.VAL   <- lrm(DATA.VAL$y~predict(fit.ML,   DATA.VAL))
    fit.boot.VAL <- lrm(DATA.VAL$y~predict(fit.boot, DATA.VAL))
    fit.heur.VAL <- lrm(DATA.VAL$y~predict(fit.heur, DATA.VAL))

    cal.slope.fit.val.ML[i, j]   <- as.numeric(coef(fit.ML.VAL)[2])
    cal.slope.fit.val.boot[i, j] <- as.numeric(coef(fit.boot.VAL)[2])
    cal.slope.fit.val.heur[i, j] <- as.numeric(coef(fit.heur.VAL)[2])

    fit.cal.ML.VAL   <- glm(DATA.VAL$y~offset(predict(fit.ML,   DATA.VAL)), family='binomial')
    fit.cal.boot.VAL <- glm(DATA.VAL$y~offset(predict(fit.boot, DATA.VAL)), family='binomial')
    fit.cal.heur.VAL <- glm(DATA.VAL$y~offset(predict(fit.heur, DATA.VAL)), family='binomial')
    
    cal.intercept.fit.val.ML[i, j]   <- as.numeric(coef(fit.cal.ML.VAL)[1])
    cal.intercept.fit.val.boot[i, j] <- as.numeric(coef(fit.cal.boot.VAL)[1])
    cal.intercept.fit.val.heur[i, j] <- as.numeric(coef(fit.cal.heur.VAL)[1])
    
    c.ML.VAL[i, j]   <- as.numeric(fit.ML.VAL$stats['C'])
    c.boot.VAL[i, j] <- as.numeric(fit.boot.VAL$stats['C'])
    c.heur.VAL[i, j] <- as.numeric(fit.heur.VAL$stats['C'])
    
    R2.ML.VAL[i, j]   <- as.numeric(fit.ML.VAL$stats['R2'])
    
    ### Get various performance measures (apparent)
    c.ridge[i, j]      <- as.numeric(fit.ridge$stats['C'])
    c.elasticnet[i, j] <- as.numeric(fit.elasticnet$stats['C'])
    c.lasso[i, j]      <- as.numeric(fit.lasso$stats['C'])
    
    R2.ML[i, j]         <- as.numeric(fit.ML$stats['R2'])
    R2.boot[i, j]       <- as.numeric(fit.boot.lrm$stats['R2'])
    R2.heur[i, j]       <- as.numeric(fit.heur.lrm$stats['R2'])  
    R2.ridge[i, j]      <- as.numeric(fit.ridge$stats['R2'])
    R2.elasticnet[i, j] <- as.numeric(fit.elasticnet$stats['R2'])
    R2.lasso[i, j]      <- as.numeric(fit.lasso$stats['R2'])
  }
}


### remove outliers
cv.ridge[cv.ridge > 2.5]           <- NA
cv.elasticnet[cv.elasticnet > 2.5] <- NA
cv.lasso[cv.lasso > 2.5]           <- NA

heur.shrink[heur.shrink < 0] <- NA
boot.shrink[boot.shrink < 0] <- NA

cal.slope.fit.val.ML[abs(cal.slope.fit.val.ML) > 5]                 <- NA
cal.slope.fit.val.heur[abs(cal.slope.fit.val.heur) > 5]             <- NA
cal.slope.fit.val.boot[abs(cal.slope.fit.val.boot) > 5]             <- NA
cal.slope.fit.val.ridge[abs(cal.slope.fit.val.ridge) > 5]           <- NA
cal.slope.fit.val.elasticnet[abs(cal.slope.fit.val.elasticnet) > 5] <- NA
cal.slope.fit.val.lasso[abs(cal.slope.fit.val.lasso) > 5]           <- NA
cal.slope.fit.val.ML[cal.slope.fit.val.ML < -1]                     <- NA
cal.slope.fit.val.heur[cal.slope.fit.val.heur < -1]                 <- NA
cal.slope.fit.val.boot[cal.slope.fit.val.boot < -1]                 <- NA
cal.slope.fit.val.ridge[cal.slope.fit.val.ridge < -1]               <- NA
cal.slope.fit.val.elasticnet[cal.slope.fit.val.elasticnet < -1]     <- NA
cal.slope.fit.val.lasso[cal.slope.fit.val.lasso < -1]               <- NA

cv.ridge.long      <- gather(as_tibble(cv.ridge))
cv.elasticnet.long <- gather(as_tibble(cv.elasticnet))
cv.lasso.long      <- gather(as_tibble(cv.lasso))

heur.shrink.long <- gather(as_tibble(heur.shrink))
boot.shrink.long <- gather(as_tibble(boot.shrink))

c.ML.long             <- gather(as_tibble(c.ML))
c.ML.long.VAL         <- gather(as_tibble(c.ML.VAL))
c.ridge.long          <- gather(as_tibble(c.ridge))
c.ridge.long.VAL      <- gather(as_tibble(c.ridge.VAL))
c.elasticnet.long     <- gather(as_tibble(c.elasticnet))
c.elasticnet.long.VAL <- gather(as_tibble(c.elasticnet.VAL))
c.lasso.long          <- gather(as_tibble(c.lasso))
c.lasso.long.VAL      <- gather(as_tibble(c.lasso.VAL))

R2.ML.long             <- gather(as_tibble(R2.ML))
R2.ML.long.VAL         <- gather(as_tibble(R2.ML.VAL))
R2.ridge.long          <- gather(as_tibble(R2.ridge))
R2.ridge.long.VAL      <- gather(as_tibble(R2.ridge.VAL))
R2.elasticnet.long     <- gather(as_tibble(R2.elasticnet))
R2.elasticnet.long.VAL <- gather(as_tibble(R2.elasticnet.VAL))
R2.lasso.long          <- gather(as_tibble(R2.lasso))
R2.lasso.long.VAL      <- gather(as_tibble(R2.lasso.VAL))

cal.slope.fit.ML.long.VAL         <- gather(as_tibble(cal.slope.fit.val.ML))
cal.slope.fit.heur.long.VAL       <- gather(as_tibble(cal.slope.fit.val.heur))
cal.slope.fit.boot.long.VAL       <- gather(as_tibble(cal.slope.fit.val.boot))
cal.slope.fit.ridge.long.VAL      <- gather(as_tibble(cal.slope.fit.val.ridge))
cal.slope.fit.elasticnet.long.VAL <- gather(as_tibble(cal.slope.fit.val.elasticnet))
cal.slope.fit.lasso.long.VAL      <- gather(as_tibble(cal.slope.fit.val.lasso))

cal.intercept.fit.ML.long.VAL         <- gather(as_tibble(cal.intercept.fit.val.ML))
cal.intercept.fit.heur.long.VAL       <- gather(as_tibble(cal.intercept.fit.val.heur))
cal.intercept.fit.boot.long.VAL       <- gather(as_tibble(cal.intercept.fit.val.boot))
cal.intercept.fit.ridge.long.VAL      <- gather(as_tibble(cal.intercept.fit.val.ridge))
cal.intercept.fit.elasticnet.long.VAL <- gather(as_tibble(cal.intercept.fit.val.elasticnet))
cal.intercept.fit.lasso.long.VAL      <- gather(as_tibble(cal.intercept.fit.val.lasso))

cv.ridge.long      <- add_column(cv.ridge.long,      method = rep("Ridge", nrow(cv.ridge.long)))
cv.elasticnet.long <- add_column(cv.elasticnet.long, method = rep("Elastic net", nrow(cv.elasticnet.long)))
cv.lasso.long      <- add_column(cv.lasso.long,      method = rep("Lasso", nrow(cv.lasso.long)))

heur.shrink.long <- add_column(heur.shrink.long, method = rep("Heuristic shrinkage", nrow(heur.shrink.long)))
boot.shrink.long <- add_column(boot.shrink.long, method = rep("Bootstrap shrinkage", nrow(boot.shrink.long)))

c.ML.long             <- add_column(c.ML.long,             method = rep("Maximum likelihood (Apparent)",  nrow(c.ML.long)))
c.ML.long.VAL         <- add_column(c.ML.long.VAL,         method = rep("Maximum likelihood",             nrow(c.ML.long.VAL)))
c.ridge.long          <- add_column(c.ridge.long,          method = rep("Ridge (Apparent)",               nrow(c.ridge.long)))
c.ridge.long.VAL      <- add_column(c.ridge.long.VAL,      method = rep("Ridge",                          nrow(c.ridge.long.VAL)))
c.elasticnet.long     <- add_column(c.elasticnet.long,     method = rep("Elastic net (Apparent)",         nrow(c.elasticnet.long)))
c.elasticnet.long.VAL <- add_column(c.elasticnet.long.VAL, method = rep("Elastic net",                    nrow(c.elasticnet.long.VAL)))
c.lasso.long          <- add_column(c.lasso.long,          method = rep("Lasso (Apparent)",               nrow(c.lasso.long)))
c.lasso.long.VAL      <- add_column(c.lasso.long.VAL,      method = rep("Lasso",                          nrow(c.lasso.long.VAL)))

c.ML.long             <- add_column(c.ML.long,             validation = rep("Apparent",   nrow(c.ML.long)))
c.ML.long.VAL         <- add_column(c.ML.long.VAL,         validation = rep("Validation", nrow(c.ML.long.VAL)))
c.ridge.long          <- add_column(c.ridge.long,          validation = rep("Apparent",   nrow(c.ridge.long)))
c.ridge.long.VAL      <- add_column(c.ridge.long.VAL,      validation = rep("Validation", nrow(c.ridge.long.VAL)))
c.elasticnet.long     <- add_column(c.elasticnet.long,     validation = rep("Apparent",   nrow(c.elasticnet.long)))
c.elasticnet.long.VAL <- add_column(c.elasticnet.long.VAL, validation = rep("Validation", nrow(c.elasticnet.long.VAL)))
c.lasso.long          <- add_column(c.lasso.long,          validation = rep("Apparent",   nrow(c.lasso.long)))
c.lasso.long.VAL      <- add_column(c.lasso.long.VAL,      validation = rep("Validation", nrow(c.lasso.long.VAL)))

R2.ML.long             <- add_column(R2.ML.long,             method = rep("Maximum likelihood (Apparent)",  nrow(R2.ML.long)))
R2.ML.long.VAL         <- add_column(R2.ML.long.VAL,         method = rep("Maximum likelihood",             nrow(R2.ML.long.VAL)))
R2.ridge.long          <- add_column(R2.ridge.long,          method = rep("Ridge (Apparent)",               nrow(R2.ridge.long)))
R2.ridge.long.VAL      <- add_column(R2.ridge.long.VAL,      method = rep("Ridge",                          nrow(R2.ridge.long.VAL)))
R2.elasticnet.long     <- add_column(R2.elasticnet.long,     method = rep("Elastic net (Apparent)",          nrow(R2.elasticnet.long)))
R2.elasticnet.long.VAL <- add_column(R2.elasticnet.long.VAL, method = rep("Elastic net",                     nrow(R2.elasticnet.long.VAL)))
R2.lasso.long          <- add_column(R2.lasso.long,          method = rep("Lasso (Apparent)",               nrow(R2.lasso.long)))
R2.lasso.long.VAL      <- add_column(R2.lasso.long.VAL,      method = rep("Lasso",                          nrow(R2.lasso.long.VAL)))

R2.ML.long             <- add_column(R2.ML.long,             validation = rep("Apparent",   nrow(R2.ML.long)))
R2.ML.long.VAL         <- add_column(R2.ML.long.VAL,         validation = rep("Validation", nrow(R2.ML.long.VAL)))
R2.ridge.long          <- add_column(R2.ridge.long,          validation = rep("Apparent",   nrow(R2.ridge.long)))
R2.ridge.long.VAL      <- add_column(R2.ridge.long.VAL,      validation = rep("Validation", nrow(R2.ridge.long.VAL)))
R2.elasticnet.long     <- add_column(R2.elasticnet.long,     validation = rep("Apparent",   nrow(R2.elasticnet.long)))
R2.elasticnet.long.VAL <- add_column(R2.elasticnet.long.VAL, validation = rep("Validation", nrow(R2.elasticnet.long.VAL)))
R2.lasso.long          <- add_column(R2.lasso.long,          validation = rep("Apparent",   nrow(R2.lasso.long)))
R2.lasso.long.VAL      <- add_column(R2.lasso.long.VAL,      validation = rep("Validation", nrow(R2.lasso.long.VAL)))

cal.slope.fit.ML.long.VAL         <- add_column(cal.slope.fit.ML.long.VAL,         method = rep("Maximum likelihood",  nrow(cal.slope.fit.ML.long.VAL)))
cal.slope.fit.heur.long.VAL       <- add_column(cal.slope.fit.heur.long.VAL,       method = rep("Heuristic shrinkage", nrow(cal.slope.fit.heur.long.VAL)))
cal.slope.fit.boot.long.VAL       <- add_column(cal.slope.fit.boot.long.VAL,       method = rep("Bootstrap shrinkage", nrow(cal.slope.fit.boot.long.VAL)))
cal.slope.fit.ridge.long.VAL      <- add_column(cal.slope.fit.ridge.long.VAL,      method = rep("Ridge",               nrow(cal.slope.fit.ridge.long.VAL)))
cal.slope.fit.elasticnet.long.VAL <- add_column(cal.slope.fit.elasticnet.long.VAL, method = rep("Elastic net",         nrow(cal.slope.fit.elasticnet.long.VAL)))
cal.slope.fit.lasso.long.VAL      <- add_column(cal.slope.fit.lasso.long.VAL,      method = rep("Lasso",               nrow(cal.slope.fit.lasso.long.VAL)))

cal.slope.fit.ML.long.VAL         <- add_column(cal.slope.fit.ML.long.VAL,         validation = rep("Validation", nrow(cal.slope.fit.ML.long.VAL)))
cal.slope.fit.heur.long.VAL       <- add_column(cal.slope.fit.heur.long.VAL,       validation = rep("Validation", nrow(cal.slope.fit.heur.long.VAL)))
cal.slope.fit.boot.long.VAL       <- add_column(cal.slope.fit.boot.long.VAL,       validation = rep("Validation", nrow(cal.slope.fit.boot.long.VAL)))
cal.slope.fit.ridge.long.VAL      <- add_column(cal.slope.fit.ridge.long.VAL,      validation = rep("Validation", nrow(cal.slope.fit.ridge.long.VAL)))
cal.slope.fit.elasticnet.long.VAL <- add_column(cal.slope.fit.elasticnet.long.VAL, validation = rep("Validation", nrow(cal.slope.fit.elasticnet.long.VAL)))
cal.slope.fit.lasso.long.VAL      <- add_column(cal.slope.fit.lasso.long.VAL,      validation = rep("Validation", nrow(cal.slope.fit.lasso.long.VAL)))

cal.intercept.fit.ML.long.VAL         <- add_column(cal.intercept.fit.ML.long.VAL,         method = rep("Maximum likelihood",  nrow(cal.intercept.fit.ML.long.VAL)))
cal.intercept.fit.heur.long.VAL       <- add_column(cal.intercept.fit.heur.long.VAL,       method = rep("Heuristic shrinkage", nrow(cal.intercept.fit.heur.long.VAL)))
cal.intercept.fit.boot.long.VAL       <- add_column(cal.intercept.fit.boot.long.VAL,       method = rep("Bootstrap shrinkage", nrow(cal.intercept.fit.boot.long.VAL)))
cal.intercept.fit.ridge.long.VAL      <- add_column(cal.intercept.fit.ridge.long.VAL,      method = rep("Ridge",               nrow(cal.intercept.fit.ridge.long.VAL)))
cal.intercept.fit.elasticnet.long.VAL <- add_column(cal.intercept.fit.elasticnet.long.VAL, method = rep("Elastic net",         nrow(cal.intercept.fit.elasticnet.long.VAL)))
cal.intercept.fit.lasso.long.VAL      <- add_column(cal.intercept.fit.lasso.long.VAL,      method = rep("Lasso",               nrow(cal.intercept.fit.lasso.long.VAL)))

cal.intercept.fit.ML.long.VAL         <- add_column(cal.intercept.fit.ML.long.VAL,         validation = rep("Validation", nrow(cal.intercept.fit.ML.long.VAL)))
cal.intercept.fit.heur.long.VAL       <- add_column(cal.intercept.fit.heur.long.VAL,       validation = rep("Validation", nrow(cal.intercept.fit.heur.long.VAL)))
cal.intercept.fit.boot.long.VAL       <- add_column(cal.intercept.fit.boot.long.VAL,       validation = rep("Validation", nrow(cal.intercept.fit.boot.long.VAL)))
cal.intercept.fit.ridge.long.VAL      <- add_column(cal.intercept.fit.ridge.long.VAL,      validation = rep("Validation", nrow(cal.intercept.fit.ridge.long.VAL)))
cal.intercept.fit.elasticnet.long.VAL <- add_column(cal.intercept.fit.elasticnet.long.VAL, validation = rep("Validation", nrow(cal.intercept.fit.elasticnet.long.VAL)))
cal.intercept.fit.lasso.long.VAL      <- add_column(cal.intercept.fit.lasso.long.VAL,      validation = rep("Validation", nrow(cal.intercept.fit.lasso.long.VAL)))

cv.ridge.long      <- mutate(cv.ridge.long,      key = recode(key, "V1"="100", "V2"="200", "V3"="300","V4"="400", "V5"="500", "V6"="600", "V7"="700", "V8"="800", "V9"="900","V10"="1000"))
cv.elasticnet.long <- mutate(cv.elasticnet.long, key = recode(key, "V1"="100", "V2"="200", "V3"="300","V4"="400", "V5"="500", "V6"="600", "V7"="700", "V8"="800", "V9"="900","V10"="1000"))  
cv.lasso.long      <- mutate(cv.lasso.long,      key = recode(key, "V1"="100", "V2"="200", "V3"="300","V4"="400", "V5"="500", "V6"="600", "V7"="700", "V8"="800", "V9"="900","V10"="1000"))

heur.shrink.long <- mutate(heur.shrink.long, key = recode(key, "V1"="100", "V2"="200", "V3"="300","V4"="400", "V5"="500", "V6"="600", "V7"="700", "V8"="800", "V9"="900","V10"="1000"))
boot.shrink.long <- mutate(boot.shrink.long, key = recode(key, "V1"="100", "V2"="200", "V3"="300","V4"="400", "V5"="500", "V6"="600", "V7"="700", "V8"="800", "V9"="900","V10"="1000"))  

c.ML.long             <- mutate(c.ML.long,             key = recode(key, "V1"="100", "V2"="200", "V3"="300","V4"="400", "V5"="500", "V6"="600", "V7"="700", "V8"="800", "V9"="900","V10"="1000"))
c.ML.long.VAL         <- mutate(c.ML.long.VAL,         key = recode(key, "V1"="100", "V2"="200", "V3"="300","V4"="400", "V5"="500", "V6"="600", "V7"="700", "V8"="800", "V9"="900","V10"="1000"))
c.ridge.long          <- mutate(c.ridge.long,          key = recode(key, "V1"="100", "V2"="200", "V3"="300","V4"="400", "V5"="500", "V6"="600", "V7"="700", "V8"="800", "V9"="900","V10"="1000"))
c.ridge.long.VAL      <- mutate(c.ridge.long.VAL,      key = recode(key, "V1"="100", "V2"="200", "V3"="300","V4"="400", "V5"="500", "V6"="600", "V7"="700", "V8"="800", "V9"="900","V10"="1000"))
c.elasticnet.long     <- mutate(c.elasticnet.long,     key = recode(key, "V1"="100", "V2"="200", "V3"="300","V4"="400", "V5"="500", "V6"="600", "V7"="700", "V8"="800", "V9"="900","V10"="1000"))
c.elasticnet.long.VAL <- mutate(c.elasticnet.long.VAL, key = recode(key, "V1"="100", "V2"="200", "V3"="300","V4"="400", "V5"="500", "V6"="600", "V7"="700", "V8"="800", "V9"="900","V10"="1000"))  
c.lasso.long          <- mutate(c.lasso.long,          key = recode(key, "V1"="100", "V2"="200", "V3"="300","V4"="400", "V5"="500", "V6"="600", "V7"="700", "V8"="800", "V9"="900","V10"="1000"))  
c.lasso.long.VAL      <- mutate(c.lasso.long.VAL,      key = recode(key, "V1"="100", "V2"="200", "V3"="300","V4"="400", "V5"="500", "V6"="600", "V7"="700", "V8"="800", "V9"="900","V10"="1000"))  

R2.ML.long             <- mutate(R2.ML.long,             key = recode(key, "V1"="100", "V2"="200", "V3"="300","V4"="400", "V5"="500", "V6"="600", "V7"="700", "V8"="800", "V9"="900","V10"="1000"))
R2.ML.long.VAL         <- mutate(R2.ML.long.VAL,         key = recode(key, "V1"="100", "V2"="200", "V3"="300","V4"="400", "V5"="500", "V6"="600", "V7"="700", "V8"="800", "V9"="900","V10"="1000"))
R2.ridge.long          <- mutate(R2.ridge.long,          key = recode(key, "V1"="100", "V2"="200", "V3"="300","V4"="400", "V5"="500", "V6"="600", "V7"="700", "V8"="800", "V9"="900","V10"="1000"))
R2.ridge.long.VAL      <- mutate(R2.ridge.long.VAL,      key = recode(key, "V1"="100", "V2"="200", "V3"="300","V4"="400", "V5"="500", "V6"="600", "V7"="700", "V8"="800", "V9"="900","V10"="1000"))
R2.elasticnet.long     <- mutate(R2.elasticnet.long,     key = recode(key, "V1"="100", "V2"="200", "V3"="300","V4"="400", "V5"="500", "V6"="600", "V7"="700", "V8"="800", "V9"="900","V10"="1000"))
R2.elasticnet.long.VAL <- mutate(R2.elasticnet.long.VAL, key = recode(key, "V1"="100", "V2"="200", "V3"="300","V4"="400", "V5"="500", "V6"="600", "V7"="700", "V8"="800", "V9"="900","V10"="1000"))  
R2.lasso.long          <- mutate(R2.lasso.long,          key = recode(key, "V1"="100", "V2"="200", "V3"="300","V4"="400", "V5"="500", "V6"="600", "V7"="700", "V8"="800", "V9"="900","V10"="1000"))
R2.lasso.long.VAL      <- mutate(R2.lasso.long.VAL,      key = recode(key, "V1"="100", "V2"="200", "V3"="300","V4"="400", "V5"="500", "V6"="600", "V7"="700", "V8"="800", "V9"="900","V10"="1000"))  

cal.slope.fit.ML.long.VAL         <- mutate(cal.slope.fit.ML.long.VAL,         key = recode(key, "V1"="100", "V2"="200", "V3"="300","V4"="400", "V5"="500", "V6"="600", "V7"="700", "V8"="800", "V9"="900","V10"="1000"))
cal.slope.fit.heur.long.VAL       <- mutate(cal.slope.fit.heur.long.VAL,       key = recode(key, "V1"="100", "V2"="200", "V3"="300","V4"="400", "V5"="500", "V6"="600", "V7"="700", "V8"="800", "V9"="900","V10"="1000"))
cal.slope.fit.boot.long.VAL       <- mutate(cal.slope.fit.boot.long.VAL,       key = recode(key, "V1"="100", "V2"="200", "V3"="300","V4"="400", "V5"="500", "V6"="600", "V7"="700", "V8"="800", "V9"="900","V10"="1000"))
cal.slope.fit.ridge.long.VAL      <- mutate(cal.slope.fit.ridge.long.VAL,      key = recode(key, "V1"="100", "V2"="200", "V3"="300","V4"="400", "V5"="500", "V6"="600", "V7"="700", "V8"="800", "V9"="900","V10"="1000"))
cal.slope.fit.elasticnet.long.VAL <- mutate(cal.slope.fit.elasticnet.long.VAL, key = recode(key, "V1"="100", "V2"="200", "V3"="300","V4"="400", "V5"="500", "V6"="600", "V7"="700", "V8"="800", "V9"="900","V10"="1000"))  
cal.slope.fit.lasso.long.VAL      <- mutate(cal.slope.fit.lasso.long.VAL,      key = recode(key, "V1"="100", "V2"="200", "V3"="300","V4"="400", "V5"="500", "V6"="600", "V7"="700", "V8"="800", "V9"="900","V10"="1000"))

cal.intercept.fit.ML.long.VAL         <- mutate(cal.intercept.fit.ML.long.VAL,         key = recode(key, "V1"="100", "V2"="200", "V3"="300","V4"="400", "V5"="500", "V6"="600", "V7"="700", "V8"="800", "V9"="900","V10"="1000"))
cal.intercept.fit.heur.long.VAL       <- mutate(cal.intercept.fit.heur.long.VAL,       key = recode(key, "V1"="100", "V2"="200", "V3"="300","V4"="400", "V5"="500", "V6"="600", "V7"="700", "V8"="800", "V9"="900","V10"="1000"))
cal.intercept.fit.boot.long.VAL       <- mutate(cal.intercept.fit.boot.long.VAL,       key = recode(key, "V1"="100", "V2"="200", "V3"="300","V4"="400", "V5"="500", "V6"="600", "V7"="700", "V8"="800", "V9"="900","V10"="1000"))
cal.intercept.fit.ridge.long.VAL      <- mutate(cal.intercept.fit.ridge.long.VAL,      key = recode(key, "V1"="100", "V2"="200", "V3"="300","V4"="400", "V5"="500", "V6"="600", "V7"="700", "V8"="800", "V9"="900","V10"="1000"))
cal.intercept.fit.elasticnet.long.VAL <- mutate(cal.intercept.fit.elasticnet.long.VAL, key = recode(key, "V1"="100", "V2"="200", "V3"="300","V4"="400", "V5"="500", "V6"="600", "V7"="700", "V8"="800", "V9"="900","V10"="1000"))  
cal.intercept.fit.lasso.long.VAL      <- mutate(cal.intercept.fit.lasso.long.VAL,      key = recode(key, "V1"="100", "V2"="200", "V3"="300","V4"="400", "V5"="500", "V6"="600", "V7"="700", "V8"="800", "V9"="900","V10"="1000"))  

cv.ridge.long      <- mutate(cv.ridge.long,      key = factor(key, levels = as.character(N)))
cv.elasticnet.long <- mutate(cv.elasticnet.long, key = factor(key, levels = as.character(N)))
cv.lasso.long      <- mutate(cv.lasso.long,      key = factor(key, levels = as.character(N)))

heur.shrink.long <- mutate(heur.shrink.long, key = factor(key, levels = as.character(N)))
boot.shrink.long <- mutate(boot.shrink.long, key = factor(key, levels = as.character(N)))

c.ML.long             <- mutate(c.ML.long,             key = factor(key, levels = as.character(N)))
c.ML.long.VAL         <- mutate(c.ML.long.VAL,         key = factor(key, levels = as.character(N)))
c.ridge.long          <- mutate(c.ridge.long,          key = factor(key, levels = as.character(N)))
c.ridge.long.VAL      <- mutate(c.ridge.long.VAL,      key = factor(key, levels = as.character(N)))
c.elasticnet.long     <- mutate(c.elasticnet.long,     key = factor(key, levels = as.character(N)))
c.elasticnet.long.VAL <- mutate(c.elasticnet.long.VAL, key = factor(key, levels = as.character(N)))
c.lasso.long          <- mutate(c.lasso.long,          key = factor(key, levels = as.character(N)))
c.lasso.long.VAL      <- mutate(c.lasso.long.VAL,      key = factor(key, levels = as.character(N)))

R2.ML.long              <- mutate(R2.ML.long,             key = factor(key, levels = as.character(N)))
R2.ML.long.VAL          <- mutate(R2.ML.long.VAL,         key = factor(key, levels = as.character(N)))
R2.ridge.long           <- mutate(R2.ridge.long,          key = factor(key, levels = as.character(N)))
R2.ridge.long.VAL       <- mutate(R2.ridge.long.VAL,      key = factor(key, levels = as.character(N)))
R2.elasticnet.long      <- mutate(R2.elasticnet.long,     key = factor(key, levels = as.character(N)))
R2.elasticnet.long.VAL  <- mutate(R2.elasticnet.long.VAL, key = factor(key, levels = as.character(N)))
R2.lasso.long           <- mutate(R2.lasso.long,          key = factor(key, levels = as.character(N)))
R2.lasso.long.VAL       <- mutate(R2.lasso.long.VAL,      key = factor(key, levels = as.character(N)))

cal.slope.fit.ML.long.VAL         <- mutate(cal.slope.fit.ML.long.VAL,         key = factor(key, levels = as.character(N)))
cal.slope.fit.heur.long.VAL       <- mutate(cal.slope.fit.heur.long.VAL,       key = factor(key, levels = as.character(N)))
cal.slope.fit.boot.long.VAL       <- mutate(cal.slope.fit.boot.long.VAL,       key = factor(key, levels = as.character(N)))
cal.slope.fit.ridge.long.VAL      <- mutate(cal.slope.fit.ridge.long.VAL,      key = factor(key, levels = as.character(N)))
cal.slope.fit.elasticnet.long.VAL <- mutate(cal.slope.fit.elasticnet.long.VAL, key = factor(key, levels = as.character(N)))
cal.slope.fit.lasso.long.VAL      <- mutate(cal.slope.fit.lasso.long.VAL,      key = factor(key, levels = as.character(N)))

cal.intercept.fit.ML.long.VAL         <- mutate(cal.intercept.fit.ML.long.VAL,         key = factor(key, levels = as.character(N)))
cal.intercept.fit.heur.long.VAL       <- mutate(cal.intercept.fit.heur.long.VAL,       key = factor(key, levels = as.character(N)))
cal.intercept.fit.boot.long.VAL       <- mutate(cal.intercept.fit.boot.long.VAL,       key = factor(key, levels = as.character(N)))
cal.intercept.fit.ridge.long.VAL      <- mutate(cal.intercept.fit.ridge.long.VAL,      key = factor(key, levels = as.character(N)))
cal.intercept.fit.elasticnet.long.VAL <- mutate(cal.intercept.fit.elasticnet.long.VAL, key = factor(key, levels = as.character(N)))
cal.intercept.fit.lasso.long.VAL      <- mutate(cal.intercept.fit.lasso.long.VAL,      key = factor(key, levels = as.character(N)))

OUT.c             <- bind_rows(c.ML.long, c.ML.long.VAL, c.ridge.long, c.ridge.long.VAL, c.elasticnet.long, c.elasticnet.long.VAL, c.lasso.long, c.lasso.long.VAL)  
OUT.R2            <- bind_rows(R2.ML.long, R2.ML.long.VAL, R2.ridge.long, R2.ridge.long.VAL, R2.elasticnet.long, R2.elasticnet.long.VAL, R2.lasso.long, R2.lasso.long.VAL)  
OUT.cal.slope     <- bind_rows(cal.slope.fit.ML.long.VAL, cal.slope.fit.heur.long.VAL, cal.slope.fit.boot.long.VAL, cal.slope.fit.ridge.long.VAL, cal.slope.fit.elasticnet.long.VAL, cal.slope.fit.lasso.long.VAL)
OUT.cal.intercept <- bind_rows(cal.intercept.fit.ML.long.VAL, cal.intercept.fit.heur.long.VAL, cal.intercept.fit.boot.long.VAL, cal.intercept.fit.ridge.long.VAL, cal.intercept.fit.elasticnet.long.VAL, cal.intercept.fit.lasso.long.VAL)
OUT.cv            <- bind_rows(cv.ridge.long, cv.elasticnet.long, cv.lasso.long)
OUT.shrink        <- bind_rows(heur.shrink.long, boot.shrink.long)

OUT.c             <- OUT.c   %>% mutate(method = factor(method, levels = c("Maximum likelihood (Apparent)", "Maximum likelihood", "Ridge (Apparent)", "Ridge", "Elastic net (Apparent)", "Elastic net", "Lasso (Apparent)", "Lasso")))
OUT.R2            <- OUT.R2  %>% mutate(method = factor(method, levels = c("Maximum likelihood (Apparent)", "Maximum likelihood", "Ridge (Apparent)", "Ridge", "Elastic net (Apparent)", "Elastic net", "Lasso (Apparent)", "Lasso")))
OUT.cal.slope     <- OUT.cal.slope %>% mutate(method = factor(method, levels = c("Maximum likelihood","Heuristic shrinkage", "Bootstrap shrinkage", "Ridge", "Elastic net", "Lasso")))
OUT.cal.intercept <- OUT.cal.intercept %>% mutate(method = factor(method, levels = c("Maximum likelihood","Heuristic shrinkage", "Bootstrap shrinkage", "Ridge", "Elastic net", "Lasso")))
OUT.cv            <- OUT.cv %>% mutate(method = factor(method, levels = c("Ridge", "Elastic net", "Lasso")))
OUT.shrink        <- OUT.shrink %>% mutate(method = factor(method, levels = c("Heuristic shrinkage", "Bootstrap shrinkage")))

OUT.c             <- add_column(OUT.c,             measure = rep("c-index", nrow(OUT.c)))
OUT.R2            <- add_column(OUT.R2,            measure = rep("R-squared", nrow(OUT.R2)))
OUT.cal.slope     <- add_column(OUT.cal.slope,     measure = rep("Calibrationn Slope", nrow(OUT.cal.slope)))
OUT.cal.intercept <- add_column(OUT.cal.intercept, measure = rep("CITL", nrow(OUT.cal.intercept)))

OUT <- bind_rows(OUT.c, OUT.R2, OUT.cal.slope, OUT.cal.intercept)
OUT$method = factor(OUT$method, levels = c("Maximum likelihood (Apparent)", "Maximum likelihood", "Heuristic shrinkage", "Bootstrap shrinkage", "Ridge (Apparent)", "Ridge", "Elastic net (Apparent)", "Elastic net", "Lasso (Apparent)", "Lasso"))
OUT2 <- OUT %>% filter(validation=="Validation")

OUT$key <- factor(OUT$key, levels=c("100","200","300","400","500","600","700","800","900","1000"))
OUT_mean <- OUT %>% filter(key=='1000' & validation=='Validation') %>% group_by(measure) %>% summarize(mean_val = mean(value, na.rm = T))
OUT_mean2 <- OUT %>% filter(key=='1000') %>% group_by(measure) %>% summarize(mean_val = mean(value, na.rm = T))
OUT_mean$mean_val[2] <- 0
OUT_mean$mean_val[3] <- 1

scales_y <- list(
  "c-index" = scale_y_continuous(),
  "Calibration Slope" = scale_y_log10(),
  "CITL" = scale_y_continuous(),
  "R-squared" = scale_y_continuous()
)

alpha_val <- 0.15
p1 <- ggplot(OUT.cv, aes(y = value, x = factor(key), fill = method)) + 
  facet_grid(method~., scales = 'free_y') +
  geom_jitter(alpha = alpha_val, aes(color = method)) +
  xlab("Sample size") +
  stat_summary(
    fun = median, 
    geom = "errorbar",
    aes(ymax = ..y.., ymin = ..y..), 
    position = position_dodge(width = 0.8), 
    width = 0.25) + 
  ylab(expression(Tuning~parameter~lambda)) + 
  theme_bw() +
  theme(legend.position = "none") + 
  theme(legend.title = element_blank()) + 
  theme(legend.text = element_text(size = 10)) +
  theme(axis.text = element_text(size = 8)) +
  guides(colour = guide_legend(override.aes = list(alpha = 1)))

p2 <- ggplot(OUT2, aes(y = value, x = factor(key), fill = method)) + 
  facet_grid(measure~., scales = 'free_y', labeller = label_wrap_gen(width = 10,multi_line = TRUE)) + 
  geom_jitter(alpha = alpha_val, aes(color = method), 
              position = position_jitterdodge(jitter.width = 0.45, dodge.width = 0.8),
              size = 1.2) +
  xlab("Sample size") +
  ylab("") +
  stat_summary(
    fun = median, 
    geom = "errorbar",
    aes(ymax = ..y.., ymin = ..y..), 
    position = position_dodge(width = 0.8), 
    width = 0.25) + 
  geom_hline(data = OUT_mean, aes(yintercept = mean_val)) + 
  theme_bw() +
  theme(legend.position = "bottom") + 
  theme(legend.title = element_blank()) + 
  theme(legend.text = element_text(size = 10)) +
  theme(axis.text = element_text(size = 8)) +
  guides(colour = guide_legend(override.aes = list(alpha = 1)))


p1/p2
ggsave("Figure_3.jpg", width = 6, height = 8)


