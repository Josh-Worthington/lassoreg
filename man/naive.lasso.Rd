\name{naive.lasso}
\alias{naive.lasso}
\title{
Lasso Regression with Naive Updates
}
\description{
This will construct a "lasmod" object, a linear model based off the lasso regression with covariance updates, and apply cross-validation. The model is made up of the coefficients (beta) for the model, the lambda value reached to achieve this, and the mean squared error (mse) of the model.
}
\usage{
naive.lasso(data, target_index, k_folds, num_lambdas, epsilon=10^-3, ...)
}
\arguments{
  \item{data}{ a dataframe containing the data from which the model is constructed. }
  \item{target_index}{ the index of the column to use as the response. }
  \item{k_folds}{ the number of folds to use in the K-folds cross-validation. }
  \item{num_lambdas}{ the number of lambda values to run through in the method. }
  \item{epsilon}{ an optional error variable used to check convergence. }
  \item{\dots}{ unused. }
}
\value{
  An object of class \code{lasmod}, a list of the following elements
  \item{lambda}{ the lambda value used to get the coefficients }
  \item{coefficients}{ a named vector of the coefficients }
  \item{mse}{ the mean squared error of the model }
  \item{lambdas}{ a vector of lambda values used in the regression }
  \item{betas}{ a matrix where all rows correspond to the coefficients from the lambda value in the vector }
  \item{mses}{ a vector of mean squared errors for the lambda values }
}
\author{
  Joshua Worthington
}
\seealso{
  \code{\link{plot.lasmod}}
  \code{\link{predict.lasmod}}
}
\examples{
data(mtcars)
model <- naive.lasso(mtcars, 1, 10, 100)
model
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ naive }% use one of  RShowDoc("KEYWORDS")
\keyword{ model }% __ONLY ONE__ keyword per line
