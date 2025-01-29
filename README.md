# lassoreg
Implementation of Lasso Regression using k-fold cross validation to choose the bet tuning parameter, written in R for my final year Maths Project.

## What is Lasso Regression?
> Least absolute shrinkage and selection operator (lasso, Lasso, LASSO) regression is a regularization method and a form of supervised statistical learning (i.e., machine learning) that is often applied when there are many potential predictor variables. Typically, when applying lasso regression the analyst’s primary goal is to improve model prediction, and other scientific research goals goals, such as understanding, explaining, or determining the cause of the outcome/phenomenon of interest, are of little or no interest. In other words, lasso regression can be a great exploratory tool for building a model that accurately predicts a particular outcome, especially when the model includes a large number of predictor variables. Of note, given its emphasis on prediction, there is often little if any attention paid to whether a non-zero lasso regression coefficient is statistically significant; with that said, there has been some preliminary work toward developing significance tests for lasso regression coefficients (Lockhart, Taylor, and Tibshirani 2014).
(Chapter 54: Supervised Statistical Learning Using Lasso Regression, R for HR, https://rforhr.com/lassoregression.html)

It is essentially an extension of Ordinary Least Squares (OLS), which seeks to minimise the Residual Sum of Squares (RSS - an error function of training values and predicted values), that also penalises the sum of the absolute value of the weights assigned to predictors. This way, it focuses shrinking the impact of predictors while eliminating the least important ones from the model, attempting to balance accuracy and complexity.

## Usage
After pulling the repository, run the main code base `lasso.R` to initialise the functions.

All examples below are using the WHO Life Expectancy dataset (https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who) to predict life expectancy, with Country and Year removed, "Status" transformed into an integer (-1 for developing, 1 for developed, in doing so assuming they have opposite effect on life expectancy) and any rows with missing data removed, e.g.
```
> Life.Expectancy.Data$Numeric.Status <- ifelse(Life.Expectancy.Data$Status=="Developing", -1, 1)
> Life.Expectancy.Data <- subset(Life.Expectancy.Data, select = -c(Country, Year, Status))
> Life.Expectancy.Data <- Life.Expectancy.Data[complete.cases(Life.Expectancy.Data),]
```

### lasso
The `lasso` function takes a training dataframe, which includes the variable you want to predict, the column index of the variable for prediction, the number of folds for the k-folds cross validation, and the number of tuning parameters to try during cross validation, e.g.
```
> lasso_model <- lasso(Life.Expectancy.Data, 1, 10, 100)
```
This returns a `lasmod` (lasso model) object that can be passed into the other analysis/prediction functions.

Calling the `lasmod` object prints out the results of the regression, namely the Mean Squared Error (MSE) and a vector of the coefficients, e.g.
```
> lasso_model
Call:
lasso.default(data = Life.Expectancy.Data, target_index = 1, 
    k_folds = 10, num_lambdas = 100)

Coefficients:
 [1] 77.73036343 -0.05560336  0.00000000  0.00000000  0.00000000  0.00000000
 [7]  0.00000000  0.00000000  0.00000000  0.00000000  0.00000000  0.00000000
[13]  0.00000000  0.00000000  0.00000000  0.00000000  0.00000000  0.04149574
[19]  0.07448288  0.00000000

Mean Squared Error:
[1] 41.71765
```

### summary.lasmod
The `summary.lasmod` function takes the given `lasmod` object, and prints a summary of the training process and result including the lambdas used during training, coefficients (betas) resulting from that lambda, MSEs with that lambda, the final chosen lambda and the cofficients of the chosen lambda, e.g.
```
> summary.lasmod(lasso_model)
Call:
lasso.default(data = Life.Expectancy.Data, target_index = 1, 
    k_folds = 10, num_lambdas = 100)

Lambdas:
  [1] 6.396959056 5.965825426 5.563748759 5.188770714 4.839064934 4.512928153
  [cut short for brevity]

Betas:
                [,1]         [,2]         [,3]          [,4]         [,5]
  [1,]  6.930051e+01  0.000000000  0.000000000  0.0000000000 0.0000000000
  [cut short for brevity]

Mean Squared Errors:
  [1] 7.740015e+01 7.740015e+01 7.740015e+01 7.740015e+01 7.740015e+01 7.740015e+01
  [cut short for brevity]

Lambda:
[1] 0.5965825

Coefficients:
 [1] 77.73036343 -0.05560336  0.00000000  0.00000000  0.00000000  0.00000000
 [7]  0.00000000  0.00000000  0.00000000  0.00000000  0.00000000  0.00000000
[13]  0.00000000  0.00000000  0.00000000  0.00000000  0.00000000  0.04149574
[19]  0.07448288  0.00000000

Minimum Mean Squared Error:
[1] 41.71765
```

### predict.lasmod
The `predict.lasmod` function takes the given `lasmod` object, and predicts the value for the given sample vector of predictors, e.g. taking the first row of the training data without the life expectancy
```
> x <- Life.Expectany.Data[1,-1]
> predict.lasmod(lasso_model, x)
         [,1]
[1,] 63.87883
```
The actual life expectancy for that row was 65 to those interested.

### plot.lasmod and plotmse.lasmod
The `plot.lasmod` function returns a graph which plots how the coefficients changed during the course of training, coefficient values against lambda, e.g.
```
> plot.lasmod(lasso_model)
```
creates this graph:
![CoefficientsGraph](https://github.com/user-attachments/assets/4f70b936-a3ea-41ae-8e8b-b40bcee6c4cf)

Whereas the `plotmse.lasmod` function returns a graph which plots how the mean squared error changed during the course of training, MSE against lambda, e.g.
```
> plotmse.lasmod(lasso_model)
```
creates this graph:
![MSEGraph](https://github.com/user-attachments/assets/67d53fab-24eb-4536-ac81-c71ee1c7d472)
