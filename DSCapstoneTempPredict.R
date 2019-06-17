#
# This R script was created to submit as partial fulfillment of the Edx Data Science Capstone
# This file contains the chose your own project from the Capstone.
# Submitted by John Wallace
#

# This project will explore a method to develop a predictive time series model.

#  The dataset that is being used represents live captured data from a refrigeration control system. 

#
#  The following is a description of the steps taken to develop the model:
# 
#  Step 1) The dataset is loaded from a csv file
#  Step 2) Preliminay Data exploration is performed.   
#      Note that the data is time series data and represents the temperature of the refrigeration case.
#      sampled at approximately 3 minute intervals.
#      The data is checked for mising values other issues and any required adjustments made.  Visualizaton
#      techniques are used to inspect the data.
#  Step 3) The data is cleansed.  Data is checked for missing values and other issues and any required 
#      adjustments made.  Moving averages are calculated to help smooth the data
#  Step 4) Decompose the data.  Time series analysis relies heavily on detecting seasonality, trend and cycle
#      We will look for these patterns in the data as that will help inform model development
#  Step 5) Creat the model. The forecast package contains a function to create an ARIMA model.  This package
#      will be used to create an ARIMA model
#  Step 6) Use the model to forecast and then check forecast
#

# References
#https://www.datascience.com/blog/introduction-to-forecasting-with-arima-in-r-learn-data-science-tutorials
#https://datascienceplus.com/time-series-analysis-using-arima-model-in-r/

#############################################################
# Load the required libraries
#############################################################

library(tibble)
library(readr)
library(tidyverse)
library(readxl)
library (dplyr)
library (xts)

library('ggplot2')
library('forecast')
library('tseries')
library(readr)
library(caret)



#  Step 1) The dataset is loaded from a csv file
#    Note that the data file must be in the working directory
#    (or add path information to the filename in the read_csv call)
#
CaseTempData <- read_csv("CapstoneTempData.csv")


#  Step 2) Data exploration is performed.   
#      Note that the data is time series data and represents the temperature of the refrigeration case.
#      sampled at approximately 3 minute intervals.
#      The data is checked for mising values other issues and any required adjustments made.  Visualizaton
#      techniques are used to inspect the data.

# Look at the structure of the data
str(CaseTempData)
# Dataframe with 480 samples; Data Column is called "CaseTemp"

head(CaseTempData)
# Data is type dbl and have 1 decimal point (Note this data represents degree's fahrenheit)

# Look at the mean and standard deviation of the data
mean (CaseTempData$CaseTemp)
sd (CaseTempData$CaseTemp)
# Mean is 36.3 with a sd of 5.3

# Plot the raw data to see if there are any patterns 
ggplot(CaseTempData, aes(x = 1:nrow(CaseTempData), y = CaseTempData$CaseTemp)) + geom_line()
# The plot shows that the data is cyclic with a repeating pattern appx every 120 samples
#   we will need to consider this when we create the ARIMA model


#  Step 3) The data is cleansed.  Data is checked for missing values and other issues and any required 
#      adjustments made.  Moving averages are calculated to reduct some of the volativity in the data

# Create a "clean" data set by creating a new data column and add to the dataframe.  This will 
#   remove any outliers which can cause issues when creating the mdoel.  Use the tsclean() function
#   that is part of the forecast package which identifies and replaces outliers via smoothing and
#   decomposition

Temp_ts <- ts(CaseTempData$CaseTemp)

CaseTempData$CleanTemp <- tsclean(Temp_ts)

# And plot the Clean data
ggplot(CaseTempData$CleanTemp, aes(x = 1:length(CaseTempData$CleanTemp), y = CaseTempData$CleanTemp)) + geom_line()

# Let's add some smoothing to the data.  Do this by calculating moving averages.
# Since the data is a time series with samples every 3 minutes, lets do a 15 minute moving average
#  and a 1 hour moving average and see the results.  We use the ma function and add to the 
#  original dataframe.  (note 5 samples = 15 minutes, 20 samples = 1 hour)
CaseTempData$CleanMA15 <- ma(CaseTempData$CleanTemp, order = 5)
CaseTempData$CleanMA60 <- ma(CaseTempData$CleanTemp, order = 20)

# Now plot the data to see how the smoothing looks

ggplot() +
  geom_line(data = CaseTempData$CleanTemp, aes(x = 1:length(CaseTempData$CleanTemp), y = CaseTempData$CleanTemp, colour = "Clean Temp")) +
  geom_line(data = CaseTempData$CleanMA15, aes(x = 1:length(CaseTempData$CleanMA15), y = CaseTempData$CleanMA15,   colour = "Clean 15 min"))  +
  geom_line(data = CaseTempData$CleanMA60, aes(x = 1:length(CaseTempData$CleanMA60), y = CaseTempData$CleanMA60, colour = "Clean 1 Hr"))  +
  ylab('Temp')

# We can see that the smoothing reomves some of the volativity but leaves the underlying pattern in place

#  Step 4) Decompose the data.  Time series analysis relies heavily on detecting seasonality, trend and cycle
#      We will look for these patterns in the data as that will help inform model development

# The building blocks of a time series analysis are seasonality, trend, and cycle.  residual is what can't be 
#  attributed to these 3

# The following code was not used as there is no seasonality in the time series
# The code is left as an example of how to review seasonality 
# First we will calculate the seasonal component
# First get a working version of the Clean moving average turned into hourly data (15 minute MA x 4)
CleanMA15Working <- ts(na.omit(CaseTempData$CleanMA15), frequency = 4)
# and Decompose
#DecompTemp <- stl(CleanMA15Working, s.window="periodic")
#DeSeasonalTemp <- seasadj(DecompTemp)
#plot(DeSeasonalTemp)
# END Seasonal example

# Now check for stationarity   stationarity  means there is no time based trend in the data
# Use the augmented Dickey-Fuller (ADF) statistical test for stationarity.  The null hypothesis
#  assumes the series is non-stationary.  
# The ADF procedure tests whether a change in Y can be explained by a lagged value and linear trend.
# If there is a presence os a trend component, the series is non-stationarity  and the null-hypothesis
#  will not be rejected.  
CleanMA15Working <- ts(na.omit(CaseTempData$CleanMA15), frequency = 4)
adf.test(CleanMA15Working)

# Based on this test, it looks like that the data is stationarity  as the p value is less than 0.05 
#  and the Null hypothesis is non-stationarity 

# We can also use an autocorrelation plot(known as ACF) as a visual tool to confirm whether a series is stationarity 
# The plot can also be useful in choosing the order parameters for the ARIMA model
# Note that if the series is correlated with its lages, then there is some trend or seasonal components.  
# The Acf & Pacf functions are from the forecast package and can detect auto correlerations

# Look at the autocorrelations with the function Acf.  Note that ACF plots can help in determing the order of
#  the MA(q) model.

Acf(CleanMA15Working, main='')
# The Acf shows a very clear indication of autocorrlations with many time lags as shown in the plot. 
# (Note the blue bards in the plot show the 95% signifiance boundaries)

# Pacf (partical autocorrelation plots) display correlations between a variable and it's lags not explained
#  by previous lags.  (PACF plots are useful when determing the order of the AR(p) model
Pacf(CleanMA15Working, main='')
# The PACF shows a significant correlation only for the 1-3 lag


# Since there does appear to be some autocorrelations with prevous lags, let's look at the adf 
#  with a difference of 1
CleanMA15Diff <- diff(CleanMA15Working, differences = 1)
plot(CleanMA15Diff)
# From the plot there still appears to be some trends, let's try a difference of 2
CleanMA15Diff <- diff(CleanMA15Working, differences = 2)
plot(CleanMA15Diff)
# with a difference of 2, there doesn't appear to be any signficant correlations
# Do an adf test to confirm
adf.test(CleanMA15Diff, alternative = "stationary")

# The adf test on the differenced data rejeccts the null hypotheses of non-stationarity which 
# suggests that a differencing of order 2 is sufficient for the model

# Note that spikes at particular lags of the differenced series can help decide the choie of p or q for the model
Acf(CleanMA15Diff, main="ACF for Differenced Series")
# significant auto correlations at lag 5, 6, 7 and beyond
Pacf(CleanMA15Diff, main="ACF for Differenced Series")
# pacf shows a significant pspike at 5

#  Step 5) Create the model. The forecast package contains a function to create an ARIMA model.  This package
#      will be used to create an ARIMA model

# First, let's use the auto arima which will automatically generate a set of optimal (p,d,q) values.  Note that
#  the auto.arima model TBD TBD- Tie in the exploratory work...
# Note that we are using seasonal = FALSE as there is no seasonal pattern in the data so we want to restrict
#  to non-seasonal models.
fit <- auto.arima(CleanMA15Working, seasonal=FALSE)
fit

# Let's see how the model performs.  If the model is good, we should expect no significant 

tsdisplay(residuals(fit), lag.max=45, main="Auto1")

# We do see autocorrelations present at lag 5, 6 and 7; re-fit the model 

fit2<- arima(CleanMA15Working, order=c(6,2,12))
tsdisplay(residuals(fit2), lag.max=45, main="(6,2,12")

fit2

# Step 6) Use the model to forecast and then check forecast

# We can create a forecast using the ARIMA model built by using the forecast function.  To forecast a future value,
#  specify the forecast horizon (h periods ahead for predictions to be made)

# Let's do a forecast for 1 hour ahead (20 steps)
fcast1Hr <- forecast(fit2, h=20)

# and plot the forecast
# Note that the light blue line shows the forecast with the light gray line showing the 95% interval & the 
#  dark gray line showing the 80% interval
plot(fcast1Hr)

# Can also use the predict function to predict
#predict(fit2, n.head = 20)

# In order to see how the model might perform on future samples, we can reserve a portion of the data as a
#  "hold out" set, fit the model and then compare the forecast to actual


# 480 total samples; hold out the last 20 samples of data
hold<- window(ts(CleanMA15Working), start=460)
# And pickup the last 20 actual values for later comparison
Last20Act <- as.vector(CaseTempData$CleanTemp[460:479])
# And refit the model based on all samples except what is being held out.  Note the same order is used
fit2NoHoldout <- arima(CleanMA15Working[1:459], order=c(6,2,12))
# And look at the results
tsdisplay(residuals(fit2NoHoldout), lag.max=45, main="No Hold (6,2,12)")

fit2NoHoldout

# And forecast the last 20 samples
fcastLast20 <- forecast(fit2NoHoldout, h=20)

plot(fcastLast20, main="Last 20")
# And add the original "hold out" data to the plot

lines(CaseTempData$CleanTemp)

# In evaulating the plot we can come to 2 conclusions based on a visual inspection:
#  First, all of the original data falls in the 95% confidence internval and most of the 
#    actual data fall within the 80% CI
#  Second, the model can predict the next few samples of the data accurately but the accuracy falls out
#    the farther ahead the model is asked to predict.

# Let's see what the RMSE is for the next 5, 10 and 20 samples
# Extract the mean value of the forecast
FCast20Mean <- as.vector(fcastLast20$mean)
#Last20Act <- as.vector(CleanMA15Working[460:479])
#Last20Act <- as.vector(CaseTempData$CleanTemp[460:479])
# Error of last 20 look ahead samples
Last20Error <- Last20Act-FCast20Mean
# Error of first 10 look ahead samples
Last10Error <- Last20Act[1:10]- FCast20Mean[1:10]
# Error of first 5 look ahead samples
Last5Error <- Last20Act[1:5] - FCast20Mean[1:5]

RMSE20 <- sqrt(mean(Last20Error^2))
RMSE10 <- sqrt(mean(Last10Error^2))
RMSE5 <- sqrt(mean(Last5Error^2))

# And finally display the results
RMSEResults <- data_frame(Forecast = "All 20", RMSE=RMSE20)
RMSEResults <-bind_rows(RMSEResults, data_frame(Forecast = "First 10", RMSE=RMSE10))
RMSEResults <-bind_rows(RMSEResults, data_frame(Forecast = "First 5", RMSE=RMSE5))
RMSEResults %>% knitr::kable()







