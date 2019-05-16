\name{predict.lasmod}
\alias{predict.lasmod}
\title{
Predict the response of the input variables
}
\description{
Predicts the response of the input variables, without intercept, based on a previously trained \code{lasmod} model.
}
\usage{
\method{predict}{lasmod}(object, x, ...)
}
\arguments{
  \item{object}{ an object of class \code{lasmod}, the fitted model returned. }
  \item{x}{ a vector data sample to perform prediction on. }
  \item{\dots}{ unused. }
}
\value{
  Estimated response based off the \code{lasmod} model
}
\author{
Joshua Worthington
}
\seealso{
  \code{\link{lasso}}
}
\examples{
data(mtcars)
model <- lasso(mtcars, 1, 10, 100)
model
x <- mtcars[5, -1]
predict.lasmod(model, x)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ estimate }% use one of  RShowDoc("KEYWORDS")
\keyword{ predict }% __ONLY ONE__ keyword per line
