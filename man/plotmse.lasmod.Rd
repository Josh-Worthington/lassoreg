\name{plotmse.lasmod}
\alias{plotmse.lasmod}
\title{
Plot Mean Squared Error
}
\description{
This function takes a \code{lasmod} object and returns the plot of lambda against the mean squared error associated with that value of lambda during the training.
}
\usage{
plotmse.lasmod(object, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{object}{ an object of class \code{lasmod}, the fitted model returned. }
  \item{\dots}{ unused. }
}
\value{
  Displays a plot of lambda against mean squared error
}
\author{
Joshua Worthington
}
\seealso{
  \code{\link{lasso}}
  \code{\link{plot.lasmod}}
}
\examples{
data(mtcars)
model <- lasso(mtcars, 1, 10, 100)
model
plotmse.lasmod(model)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ plot }% use one of  RShowDoc("KEYWORDS")
\keyword{ mse }% __ONLY ONE__ keyword per line
