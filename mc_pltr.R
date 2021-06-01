# Data loading process
library(rstudioapi)
setwd(dirname(getActiveDocumentContext()$path))
getwd()
mc_ap = rep(0, 100)
mc_ap_train = rep(0, 100)

# START OF MONTE CARLO PLTR LOOP
# NOTE: SAMPLES 1, 50, AND 79 WILL NOT CONVERGE DUE TO 
# NUMERICAL INSTABILITY
# YOU WILL NEED TO ADJUST THE FOR LOOP FROM INTERVALS
# 2:100, 51:100, AND 80:100
for (j in 80:100){
  start_time <- Sys.time()
  creditCard <- read.csv(file = glue("mc_pltr_thresh/simul_{j}_thresh.csv"))
  end_time <- Sys.time()
  end_time - start_time
  
  # Data preparation: scaling, remove unnecessary features
  require(glmnet)
  require(Matrix)
  require(PRROC)
  require(glue)
  X <- subset(creditCard, select=-c(X,
                                    TX_FRAUD))
  Y <- subset(creditCard, select = c("TX_FRAUD"))
  X_matrix = as.matrix(X)
  scaled_features = scale(X_matrix[,1:13])
  binary_vars = X[14:ncol(X)]
  X_final = as.matrix(cbind(scaled_features, binary_vars))
  Y_matrix = as.matrix(Y)
  X_sparse = Matrix(X_final, sparse=T)
  
  # 5X2-FOLD DATA SPLIT
  smp_size <- floor(0.5 * nrow(Y_matrix))
  set.seed(12177679)
  ap_scores = rep(0, 10)
  ap_scores_train = rep(0, 10)
  
  # 5 ITERATIONS START HERE
  for (i in 1:5) {
    train_idx <- sample(seq_len(nrow(Y_matrix)),
                        size= smp_size)
    
    X_train = X_sparse[train_idx,]
    Y_train = Y_matrix[train_idx,]
    X_test = X_sparse[-train_idx,]
    Y_test = Y_matrix[-train_idx,]
    
    # Ridge regression
    start_time <- Sys.time()
    ridge1_cv <- cv.glmnet(x = X_train, y = Y_train,
                           nfolds = 5,
                           type.measure="class",
                           alpha=0,
                           family="binomial",
                           standardize=F,
                           trace.it=T)
    end_time <- Sys.time()
    end_time - start_time
    
    coef(ridge1_cv, s = ridge1_cv$lambda.min)
    best_ridge_coef <- as.numeric(coef(ridge1_cv, 
                                       s = ridge1_cv$lambda.min))[-1]
    
    # LASSO regression
    start_time <- Sys.time()
    alasso1_cv <- cv.glmnet(x=X_train,
                            y=Y_train,
                            type.measure="class",
                            alpha = 1,
                            nfolds=3,
                            maxit=1e+06,
                            thresh=1e-01,
                            family="binomial",
                            standardize=F,
                            type.logistic = "modified.Newton",
                            penalty.factor = 1 / abs(best_ridge_coef),
                            keep=T,
                            trace.it=T)
    end_time <- Sys.time()
    end_time - start_time
    
    # Saves LASSO coefficients
    best_alasso_coef1 <- coef(alasso1_cv, s = alasso1_cv$lambda.min)
    writeMM(best_alasso_coef1, 
            file=glue("pltr_coeffs/{j}alasso_coef1_{i}.txt"))
    
    # Prediction and AP score calculation
    Y_pred = predict(alasso1_cv, 
                     newx = X_test, 
                     type = "response", 
                     s = alasso1_cv$lambda.min)
    
    pr <- pr.curve(scores.class0 = Y_pred, 
                   weights.class0 = Y_test,
                   curve=T)
    ap_score = pr$auc.integral
    print(ap_score)
    ap_scores[(i*2)-1] = ap_score
    # Uncomment to see PR curve
    #plot(pr, color="black")
    
    Y_pred_train = predict(alasso1_cv, 
                           newx = X_train, 
                           type = "response", 
                           s = alasso1_cv$lambda.min)
    
    pr_train <- pr.curve(scores.class0 = Y_pred_train, 
                         weights.class0 = Y_train)
    ap_scores_train[(i*2)-1] = pr_train$auc.integral
    
    # The second fold
    start_time <- Sys.time()
    ridge2_cv <- cv.glmnet(x = X_test, y = Y_test,
                           nfolds = 5,
                           type.measure="class",
                           alpha=0,
                           family="binomial",
                           standardize=F,
                           trace.it=T)
    end_time <- Sys.time()
    end_time - start_time
    
    coef(ridge2_cv, s = ridge2_cv$lambda.min)
    best_ridge_coef <- as.numeric(coef(ridge2_cv, 
                                       s = ridge2_cv$lambda.min))[-1]
    
    start_time <- Sys.time()
    alasso2_cv <- cv.glmnet(x=X_test,
                            y=Y_test,
                            type.measure="class",
                            alpha = 1,
                            nfolds=3,
                            maxit=2e+05,
                            thresh=1e-01,
                            family="binomial",
                            standardize=F,
                            type.logistic = "modified.Newton",
                            penalty.factor = 1 / abs(best_ridge_coef),
                            keep=T,
                            trace.it=T)
    end_time <- Sys.time()
    end_time - start_time
  
    best_alasso_coef2 <- coef(alasso2_cv, s = alasso2_cv$lambda.min)
    writeMM(best_alasso_coef2, 
            file=glue("pltr_coeffs/{j}alasso_coef2_{i}.txt"))
    
    Y_pred = predict(alasso2_cv, 
                     newx = X_train, 
                     type = "response", 
                     s = alasso2_cv$lambda.min)
    
    pr <- pr.curve(scores.class0 = Y_pred, 
                   weights.class0 = Y_train,
                   curve=T)
    ap_score = pr$auc.integral
    print(ap_score)
    ap_scores[(i*2)] = ap_score
    #plot(pr, color="black")
    
    Y_pred_train = predict(alasso2_cv, 
                           newx = X_test, 
                           type = "response", 
                           s = alasso2_cv$lambda.min)
    
    pr_train <- pr.curve(scores.class0 = Y_pred_train, 
                         weights.class0 = Y_test)
    ap_scores_train[(i*2)] = pr_train$auc.integral
  }
  mc_ap[j] = mean(ap_scores)
  mc_ap_train[j] = mean(ap_scores_train)
}
mean(mc_ap[mc_ap != 0])
mean(mc_ap_train[mc_ap != 0])
