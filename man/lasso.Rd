\name{lasso}
\alias{lasso}
\alias{lasso.default}
\title{
Lasso Regression
}
\description{
This will construct a "lasmod" object, a linear model based off the lasso regression with covariance updates, and apply cross-validation. The model is made up of the coefficients (beta) for the model, the lambda value reached to achieve this, and the mean squared error (mse) of the model.
}
\usage{
lasso(data, ...)

\method{lasso}{default}(data, target_index, k_folds, num_lambdas, epsilon=10^-3, ...)
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
\seealso{
  \code{\link{plot.lasmod}}
  \code{\link{predict.lasmod}}
}
\author{Joshua Worthington}
\examples{
data(mtcars)
model <- lasso(mtcars, 1, 10, 100)
model
}
\keyword{ lasso }
\keyword{ regression }
\keyword{ prediction }
