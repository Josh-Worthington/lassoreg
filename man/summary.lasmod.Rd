\name{summary.lasmod}
\alias{summary.lasmod}
\title{
  S3 \code{summary} method
}
\description{
A method that prints a summary of the model and the coordinate descent.
}
\usage{
\method{summary}{lasmod}(object, ...)
}
\arguments{
  \item{object}{ an object of class \code{lasmod}, the fitted model returned. }
  \item{\dots}{ unused. }
}
\value{
  Prints a summary of the model.
}
\author{
Joshua Worthington
}
\examples{
data(mtcars)
model <- lasso(mtcars, 1, 10, 100)
model
summary.lasmod(model)
}
\keyword{ summary }% use one of  RShowDoc("KEYWORDS")
