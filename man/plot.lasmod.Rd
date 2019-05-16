\name{plot.lasmod}
\alias{plot.lasmod}
\title{
Plot Dimension Reduction
}
\description{
This function takes a \code{lasmod} object and returns the plot of lambda against coefficients, with each line representing each coefficient.
}
\usage{
\method{plot}{lasmod}(x, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{x}{ an object of class \code{lasmod}, the fitted model returned. }
  \item{\dots}{ unused. }
}
\value{
  Displays a plot of lambda against coefficient
}
\author{
Joshua Worthington
}
\seealso{
  \code{\link{lasso}}
  \code{\link{plotmse.lasmod}}
}
\examples{
data(mtcars)
model <- lasso(mtcars, 1, 10, 100)
model
plot.lasmod(model)
}
\keyword{ plot }% use one of  RShowDoc("KEYWORDS")
\keyword{ coefficients }
