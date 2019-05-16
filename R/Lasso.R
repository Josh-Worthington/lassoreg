library(base)
require(base)

lasso <- function(data, ...) UseMethod("lasso")

lasso.default <- function(data, target_index, k_folds, num_lambdas, epsilon = 10^-3, ...)
{
  SEED <- 1234
  set.seed(SEED)
  rnd_data <- data[base::sample(nrow(data)), ] # shuffling the dataframe

  y <- rnd_data[, target_index]
  X <- rnd_data[, -target_index]

  n <- dim(X)[1]
  p <- dim(X)[2]

  scaleY <- scale(as.numeric(y))
  matX <- as.matrix(X)
  matX <- matrix(as.numeric(matX), nrow=n, ncol=p)
  scaleX <- scale(matX)
  scaleX[is.nan(scaleX)] <- 0

  max_lambda <- 1/n * max(abs(t(scaleX) %*% as.numeric(y)))

  b <- - 1/(num_lambdas-1) * log(10^3)
  A <- exp(-b) * max_lambda

  lambdas <- A*exp(b * 1:num_lambdas)
  betas <- matrix(NA, nrow=num_lambdas, ncol=(p+1))
  mses <- rep(0, num_lambdas)

  for (i in 1:num_lambdas){
    # for the warm start
    if (i == 1){
      beta <- rep(0, p)
      previous_lambda <- max_lambda
    }
    else {
      beta <- betas[i-1,-1]
      previous_lambda <- lambdas[i-1]
    }

    out <- lasso.cv(y, X, target_index, k_folds, lambdas[i], previous_lambda, beta, epsilon)
    betas[i,] <- out$beta
    mses[i] <- out$mse
  }

  min_idx <- which.min(mses)
  estimate = list("lambda" = lambdas[min_idx], "coefficients" = betas[min_idx,], "mse" = mses[min_idx],
                  "lambdas" = lambdas, "betas" = betas, "mses" = mses)
  if (length(names(X)) == 0) {
    names(estimate$coefficients) <- c("Const", names(X))
  } else {
    nms <- c("Const")
    for (j in 1:p) {
      nms <- c(nms, as.character(j))
    }
  }
  estimate$call <- match.call()

  class(estimate) <- "lasmod"
  estimate
}

naive.lasso <- function(data, target_index, k_folds, num_lambdas, epsilon = 10^-3, ...)
{
  SEED <- 1234
  set.seed(SEED)
  rnd_data <- data[base::sample(nrow(data)), ] # shuffling the dataframe

  y <- rnd_data[, target_index]
  X <- rnd_data[, -target_index]

  n <- dim(X)[1]
  p <- dim(X)[2]

  matX <- as.matrix(X)
  matX <- scale(matrix(as.numeric(matX), nrow=n, ncol=p))
  matX[is.nan(matX)] <- 0

  max_lambda <- 1/n * max(abs(t(matX) %*% as.numeric(y)))

  b <- - 1/(num_lambdas-1) * log(10^3)
  A <- exp(-b) * max_lambda

  lambdas <- A*exp(b * 1:num_lambdas)
  betas <- matrix(NA, nrow=num_lambdas, ncol=(p+1))
  mses <- rep(0, num_lambdas)

  for (i in 1:num_lambdas){
    # for the warm start
    if (i == 1){
      beta <- rep(0, p)
    }
    else {
      beta <- betas[i-1,-1]
    }

    out <- lasso.cv.naive(y, X, target_index, k_folds, lambdas[i], beta, epsilon)
    betas[i,] <- out$beta
    mses[i] <- out$mse
  }

  min_idx <- which.min(mses)
  estimate = list("lambda" = lambdas[min_idx], "coefficients" = betas[min_idx,], "mse" = mses[min_idx],
                  "lambdas" = lambdas, "betas" = betas, "mses" = mses)
  if (length(names(X)) == 0) {
    names(estimate$coefficients) <- c("Const", names(X))
  } else {
    nms <- c("Const")
    for (j in 1:p) {
      nms <- c(nms, as.character(j))
    }
  }
  estimate$call <- match.call()

  class(estimate) <- "lasmod"
  estimate
}

# using k-folds cross validation and the naive method
lasso.cv.naive <- function(y, X, target_index, k_folds, lambda, beta, epsilon)
{
  n <- length(y)
  p <- dim(X)[2]

  betas <- matrix(NA, nrow=k_folds, ncol=(p+1))
  mses <- rep(0, k_folds)

  for (k in 1:k_folds)
  {
    testset <- ((k - 1) * n/k_folds + 1):(k * n/k_folds)
    y_test <- y[testset]
    X_test <- cbind(1, X[testset, ])
    y_train <- y[-testset]
    X_train <- X[-testset, ]

    betas[k,] <- lasso.naive_update(y_train, X_train, lambda, beta, epsilon)
    mses[k] <- MSE(y_test, X_test, betas[k,])
  }

  list("beta" = colMeans(betas), "mse" = mean(mses))
}

lasso.naive_update <- function(y, X, lambda, initial_beta, eps)
{
  matX <- as.matrix(X)
  n <- dim(X)[1]
  p <- dim(X)[2]

  matX <- matrix(as.numeric(matX), nrow=n, ncol=p)
  scaleX <- scale(matX)
  scaleX[is.nan(scaleX)] <- 0
  scaleY <- scale(as.numeric(y))

  beta <- initial_beta
  # need a test for convergence
  converge <- rep(FALSE, p)
  iter <- 0
  while(!identical(converge, rep(TRUE, p)) && iter < 100)
  {
    #residuals <- matrix(rep(as.numeric(y) - matX %*% beta, p), nrow=n, ncol=p)
    #part_res <- residuals - sweep(matX, MARGIN=2, beta, `*`)

    for (j in 1:p)
    {
      residuals <- scaleY - (scaleX %*% beta)

      # naive update of beta
      betastar <- (1/n * t(scaleX[,j]) %*% residuals) + beta[j]

      #sum <- 0
      #for (i in 1:n){
      #  sum <- sum + X[i, j]*part_res[i, j]
      #}
      #test_betastar <- 1/n * sum
      old_beta <- beta[j]

      # soft-thresholding
      beta[j] <- sign(betastar) * max(0, abs(betastar) - lambda)

      converge[j] <- abs(beta[j] - old_beta) < eps
    }

    iter <- iter + 1
  }

  beta0 <- mean(y) - t(colMeans(matX)) %*% beta
  beta <- c(beta0, beta)
  beta
}

# using k-folds cross validation and the covariance method
lasso.cv <- function(y, X, target_index, k_folds, lambda, previous_lambda, beta, epsilon)
{
  n <- length(y)
  p <- dim(X)[2]

  betas <- matrix(NA, nrow=k_folds, ncol=(p+1))
  mses <- rep(0, k_folds)

  for (k in 1:k_folds)
  {
    testset <- ((k - 1) * n/k_folds + 1):(k * n/k_folds)
    y_test <- y[testset]
    X_test <- cbind(1, X[testset, ])
    y_train <- y[-testset]
    X_train <- X[-testset, ]

    betas[k,] <- lasso.cov_update(y_train, X_train, lambda, previous_lambda, beta, epsilon)
    mses[k] <- MSE(y_test, X_test, betas[k,])
  }

  list("beta" = colMeans(betas), "mse" = mean(mses))
}

lasso.cov_update <- function(y, X, lambda, previous_lambda, initial_beta, eps)
{
  n <- dim(X)[1]
  p <- dim(X)[2]

  matX <- as.matrix(X)
  matX <- matrix(as.numeric(matX), nrow=n, ncol=p)
  scaleX <- scale(matX)
  scaleX[is.nan(scaleX)] <- 0
  scaleY <- scale(as.numeric(y))

  beta <- initial_beta

  converge <- rep(TRUE, p)

  # constructing the strong set
  residuals <- scaleY - (scaleX %*% beta)
  S <- list()
  for (j in 1:p)
  {
    if (1/n * abs(t(scaleX[,j]) %*% residuals) > lambda - (previous_lambda - lambda)){
      S <- c(S, j)
      converge[j] <- FALSE
    }
  }

  iter <- 0
  # need a test for convergence
  while(!identical(converge, rep(TRUE, p)) && length(S) > 0 && iter < 100)
  {
    #residuals <- matrix(rep(as.numeric(y) - matX %*% beta, p), nrow=n, ncol=p)
    #part_res <- residuals - sweep(matX, MARGIN=2, beta, `*`)

    for (j in S)
    {
      if (iter == 0) {
        residuals <- scaleY - (scaleX %*% beta)

        # naive update of beta
        betastar <- beta[j] + (1/n * t(scaleX[,j]) %*% residuals)
      }

      else {
        dbeta <- (t(scaleX[,j]) %*% scaleY)

        for (k in S) {
          if (abs(beta[k]) > 0){
            dbeta <- dbeta - (t(scaleX[,j]) %*% scaleX[,k]) * beta[k]
          }
        }

        dbeta <- 1/n * dbeta
        betastar <- beta[j] + dbeta
      }

      #test_betastar <- 1/n * sum
      old_beta <- beta[j]

      # soft-thresholding
      beta[j] <- sign(betastar) * max(0, abs(betastar) - lambda)

      converge[j] <- abs(beta[j] - old_beta) < eps
    }

    iter <- iter + 1
  }

  beta0 <- mean(y) - t(colMeans(matX)) %*% beta
  beta <- c(beta0, beta)
  beta
}

MSE <- function(y, X, beta)
{
  matX <- as.matrix(X)
  n <- dim(X)[1]
  p <- dim(X)[2]
  matX <- matrix(as.numeric(matX), nrow=n, ncol=p)

  predicted <- matX %*% beta
  mse <- 1/n * sum((y - predicted)*(y - predicted))
  mse
}

summary.lasmod <- function(object, ...)
{
  cat("Call:\n")
  print(object$call)
  cat("\nLambdas:\n")
  print(object$lambdas)
  cat("\nBetas:\n")
  print(object$betas)
  cat("\nMean Squared Errors::\n")
  print(object$mses)
  cat("\nLambda:\n")
  print(object$lambda)
  cat("\nCoefficients:\n")
  print(object$coefficients)
  cat("\nMinimum Mean Squared Error:\n")
  print(object$mse)
}

print.lasmod <- function(x, ...)
{
  cat("Call:\n")
  print(x$call)
  cat("\nCoefficients:\n")
  print(x$coefficients)
  cat("\nMean Squared Error:\n")
  print(x$mse)
}

plot.lasmod <- function(x, ...)
{
  X <- x$lambdas
  y <- x$betas[,-1]
  p <- dim(y)[2]

  xrange <- range(X)
  yrange <- range(y)
  plot(xrange, yrange, type = 'n', log = 'x', xlab = "Lambda", ylab = "Coefficients")
  for (j in 1:p) {
    lines(X, y[,j], type = 'l')
  }
}

plotmse.lasmod <- function(object, ...)
{
  x <- object$lambdas
  y <- object$mses

  if (max(y) > 10^2){
    plot(x, y, log = 'xy', xlab = "Lambda", ylab = "Mean Squared Error")
    lines(x, y, type = 'l')
  } else {
    plot(x, y, log = 'x', xlab = "Lambda", ylab = "Mean Squared Error")
    lines(x, y, type = 'l')
  }
}

predict.lasmod <- function(object, x, ...)
{
  beta <- object$coefficients
  x <- c(1, x)
  yhat <- as.numeric(x) %*% beta
  yhat
}
