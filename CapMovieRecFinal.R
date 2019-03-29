#
# This R script was created to submit as partial fulfillment of the Edx Data Science Capstone
# Submitted by John Wallace
#
#
# *****************************************************************************************************
# The following is Canned code from edX to create data files

#  Note this is commented out as it was run originally and the files saved to disk to facilitate 
#  working on the models without recreating the original data
#  Uncomment as needed so any new run will produce edx and validation set.
#
# Files Created: edxdata.csv (edx file to be used for training)
#                validationdata.csv (for model validation)
#                movielens.csv (movie lens data in its entirety (before split))
#                movies.csv (movies flie as created below)
#
# Begin canned code:
## File created on 1/20/2019 from Capstone Project "Create Test and Validation Sets"

#NOTE: Updated 1/18/2019.

#Create test and validation sets
#############################################################
# Create edx set, validation set, and submission file
#############################################################

# Note: this process could take a couple of minutes

#if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
#if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

#dl <- tempfile()
#download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

#ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
#                      col.names = c("userId", "movieId", "rating", "timestamp"))

#movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
#colnames(movies) <- c("movieId", "title", "genres")
#movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
#                                           title = as.character(title),
#                                           genres = as.character(genres))

#movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

#set.seed(1)
#test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
#edx <- movielens[-test_index,]
#temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

#validation <- temp %>% 
#  semi_join(edx, by = "movieId") %>%
#  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

#removed <- anti_join(temp, validation)
#edx <- rbind(edx, removed)

#rm(dl, ratings, movies, test_index, temp, movielens, removed)

#
# edx now contains the training set, validation contains the test set.
#
# End Canned Code from Capstone Instructions
#
# *****************************************************************************************

# 
# Now, Save the edx and validationdata to disk to lessen time to recreate when working 
#  on differnet sessions to optimize the model
# 
#write.csv(edx, "edxdata.csv")
#write.csv(validation, "validationdata.csv")
# Note when reading back in you must drop the first column as the write.csv adds an index col
#   As in the following example
# v1 <- read.csv("validationdata.cxv", stringsAsFactors = FALSE)
# v1 <- V1[,-1] # which drops the first column
# Result was checked to insure they were the same using
#which(which(v2 == validation) == FALSE) 
# or all.equal(v2, validation)
# Note that identical does not return identical (I think due to small math variations and tolerance)
#
# End Code to create/save datasets

# Start Model Creation work
# Required Libs
library(tidyverse)
library(caret)

# Begin Code to read data
# Be sure use set_wd() to correct path 
edx <- read.csv("edxdata.csv", stringsAsFactors = FALSE)
validation <- read.csv("validationdata.csv", stringsAsFactors = FALSE)
movielens <- read.csv("movielens.csv", stringsAsFactors = FALSE)
#
# Note that the read.csv function inserts an index as the first field when it reads so that
#   must be removed.
edx <- edx[,-1]
validation <- validation[,-1]
movielens <- movielens[,-1]

#
# Now we need to "cleanse" the data
#

# Do we have any NA's in the data?
edxNA <- apply(edx, 2, function(x) any(is.na(x)))
edxNA

# There are no NA's (Note you would use colSums(is.na(df) and which(is.na(df))) to see which
#  entries were NA's 

# Check for NULL's
edxNULL <- apply(edx, 2, function(x) any(is.null(x)))
edxNULL

# there are no NULL's
#
# End Data Cleaning
#
# 
# Explore the dataset to understand the structure and type of data available
#
typeof(edx)
summary.default(edx)
summary.default(validation)
# Note that edx is about 9M rows and validation is about 999K rows (10%)

# Next determine how many users and how many movies are in the data
edx %>% 
  summarize(n_users = n_distinct(userId),
            n_movies = n_distinct(movieId))
# 69,878 users and 10,677 movies in edx
validation %>%
  summarize(n_users = n_distinct(userId),
            n_movies = n_distinct(movieId))
# 68,534 users and 9,809 movies in validation


# Let's explore the edx data in more detail.  Note we only explore the edx data
#   and leave the validation data as an unknown

# Look at the distribution of Movie ratings. 

edx %>% 
  count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Edx Movies")

# some movies only have a few ratings; others have lots 

# Now look at at the distribution of users and how many movies they provide a ratings for

edx %>% 
  count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Edx Users")

# some users have rated lots  of movies while many only rate a few
 
# End data exploration

#
# Start Model Development
#   Note the rubric for the project stated that the edx data set should be used
#     for the training and the validation set should be used for the final RMSE 
#     calculations to see how good the model is
# 

# Note that the process to create the model follows the method generally outlined in 
#  the EdX Data Science class
#  Note that while some steps could have been combined and just the final result presented, 
#  I chose to keep the steps separate to compare RMSE for different elements of the model

# The model development generally follows the approach as outlined by Dr. Raf and the book 
#   that was created along with the Edx Data Science Course.  Details about the method can
#   be found at the link to the book beolow 
#   https://rafalab.github.io/dsbook/large-datasets.html#recommendation-systems

# First, setup a function to be used to calculate the RMSE and use as a loss function 
#   This function takes 2 variables; the actual ratings (from the data) and the predicted
#   ratings.  It then uses the formula Sqrt(sum(actual-predicted)^2/N) to calculate
#   the RMSE and returns this value
#
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Start with a model assuming y[u,i]= mu + esp[u,i] where mu is the average rating of the
#   movies and and esp[u,i] is an independent errors sampled from the same distribution i
#   centered at 0. This method just uses the average rating across all movies 
mu_hat <- mean(edx$rating)
mu_hat
# The average rating is about 3.5

# Predict all ratings with just the average and see what the RMSE is
Model1_rmse <- RMSE(edx$rating, mu_hat)

# General note on results:  Note that the RMSE returns a number that represents the
#   RMSE.  The range of the rating is from 1 to 5 so an RMSE of 1 represents a pretty 
#   large potential error (1 star)

# Create a results table so we can track performance of differnet models
rmse_results <- data_frame(method = "Model1: Average only", RMSE = Model1_rmse)
# and check the results
rmse_results %>% knitr::kable()
# The first model yields an RMSE of about 1.06

# Next, let's add the movie effects to the model.  
# Some movies have higher ratings than others generally; we can take this into account by
#   Adding a term, b_i for average movie ranking for movie i (because some movies are 
#   rated higher than others ); note that the b_i is for each movie and represents the average
#   for that particular movie
# As noted in the class & book, one way to calculate b_i is to use lm to fit a linear model but as noted
#   below, but due to the size of the data set (1000's of movies) this would take a very long time to do
#   this calculation on a PC.  
#   The following line shows the code to fit the model but it is commented out.
# fit <- lm(rating ~ as.factor(movieId), data = movielens)
#
# One way to estimate the value b_i would be to use the the assumption that Y[u,i] = mu_hat + b_i_hat 
#   and solve for b_i_hat = Y[u,i] - mu_hat (where Y[u,i] is the actual rating)
#   so we can compute as the following; note the _hat notation is droped
# Now create a dataframe movie_avgs that has a value (one value for each movie) which represents 
#   on average how far above or below the overall mean a particular movie is. This will be 
#   b_i for each movie (note the _hat notation is dropped for b_i)
movie_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu_hat))
# Look at the distribution of the "difference" (b_i)  
movie_avgs %>% qplot(b_i, geom ="histogram", bins = 10, data = ., color = I("black"))
# The overall distribution is from -3 to + 1.5 which represents a (rating of -3 + 3.5 (0.5) 
#   to +1.5 + 3.5 (4.5) where 3.5 is the average for all movies (mu_hat)
# As expected most movies are around 0 which represents the overall average for the movies

# Now create the model by adding the b_i for each movie to the appropriate movie
#   predicted_ratings will now contain the ratings for each movie represented by the
#   equation Y[u,i] = mu_hat + b_i 
predicted_ratings <- mu_hat + edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  .$b_i

# Now check the RMSE with the new model which incorporates the movie effect
Model2_rmse <- RMSE(predicted_ratings, edx$rating)

# Add this model to our results
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Model2: Movie Effect Model",  
                                     RMSE = Model2_rmse ))
rmse_results %>% knitr::kable()
# Model2 improves the RMSE from 1.060... to 0.9423...

# Similar to the movie models look at the user behavior.  Some users 
#   generally rate movies higher or lower regardless of the movie so we can add that
#   factor to the model
# 
# Start by looking at the distribution of the average rating of users who have rated over 100 movies
# Note that b_u is a vector where each element represents the average rating the user_id gave movies
edx %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>% 
  filter(n()>=100) %>%
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "black")
# There is some variability across users as the distribution  from about 1.5 to 4.5.
#   Add a user term to adjust for this variability which could help 
#   the RMSE of the model

# The new model which has both a user & movie term can be written as 
#    Y[u,i] = mu + b_i + b_u + exp[u,i] where b_u  is the user specific effect
# Similar to the above note, we could use the lm to fit a model as in the following code
#   but this would take a very long time to run.
#   b_u <- lm(rating ~ as.factor(movieId) + as.factor(userId))
# Similar to the above, note the user factor can be calculated by using the equition
#   Y[u,i] = mu_hat + b_i + b_u and rewritten as b_u = Y[u,i] - mu_hat - b
# 
# user_avgs will contain a term, b_u which represents the user average for each movie
user_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu_hat - b_i))

# Use the new model now to predict the ratings using both a user effect & a movie effect
predicted_ratings <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu_hat + b_i + b_u) %>%
  .$pred

# and  check the RMSE using the new preditions

Model3_rmse <- RMSE(predicted_ratings, edx$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Model3: Movie + User Effects Model",  
                                     RMSE = Model3_rmse ))
rmse_results %>% knitr::kable()
# This model improves the results as well to about 0.8567

# ****************************************************************************


#
# Note that if a movie does not have a lot of ratings, there is more uncertainity and
#   those users that do rate the movie can have an outsized influence on the rating 
#   versus other movies and larger estimate of b_i (+ or -) are more likely.
#   In order to compensate for this ( movies with few ratings) we can use
#   a technique called regularization which effectively takes into account 
#   the number of ratings a movie has versus other movies.  Regularization
#   effectively changes the minimunization of the least square equation by adding a 
#   penalty term.  Reference the book chaper on Regularization for details.  
#   Effectively, the technique adds a penalty term which includes a "tuned" variable
#   (lambda) and the count of the number of ratings a movie has.

# The following code runs iterations to determine the tuning parameter lambda.
# Note: Originally the sequence ran to 10 but after running the code I found that
#   the best value for Lambda was around .5 so I changed the sequence to run to 2 
#   for subsequent runs in order to reduce the time to run
#
lambdas <- seq(0, 2, 0.25)

rmses <- sapply(lambdas, function(l){
  
  mu <- mean(edx$rating)
# Calculate movie term b_i with the penality count (n)lambda
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
# Calculate the user term b_u with the penality count (n) and lambda  
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
# Determine the predicted ratings using lambda  
  predicted_ratings <- 
    edx %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  # and return the RMSE for each lambda
  return(RMSE(predicted_ratings, edx$rating))
})

# and plot the rmse versus the lambda to determine the best tuned value
qplot(lambdas, rmses) 
# from plot it looks like the best will be with lambda somewhere between 0.25 and 2.
#  So modify the seq going forward to reduce the time.  Note this was already done in 
#  the code above as the sequence originally went to 10

# Determine which lambda produced the lowest rmse
lambda <- lambdas[which.min(rmses)]
lambda
# a lambda of 0.5 produced the lowest rmse use this to complete the model
#
# Note that this represents the lambda (regularization) that results in the best 
#   (lowest RMSE) calculated
# Add to the results for comparison
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized Movie + User Effect Model",  
                                     RMSE = min(rmses)))
rmse_results %>% knitr::kable()
# This results in an RMSE of 0.8566952 (marginally better than Model3)
# 
# Now we have the final model based on the edx dataset. We need to  
#   recalculate the b_i, b_u with the best lambda as found above 
#   which will be used to test the  validation model

l <- lambda
mu <- mean(edx$rating)
# b_i will contain movie factor at the tuned lambda value found above
b_i <- edx %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+l))
# b_u is the user factor at the tuned lambda value found above
b_u <- edx %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+l))

predicted_ratings <- 
  edx %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)
# Calculate the complete model using the tuned value of lambda.  Note this
#   is still on the training (edx) data.
Model5_rmse <- RMSE(predicted_ratings, edx$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Model5: Regularization Movie + User",  
                                     RMSE = Model5_rmse ))
rmse_results %>% knitr::kable()

# Now setup the final regularized model to use with the validation data.  Utilize the 
#   valu of mu, b_i and b_u found above.
# Define the model parameters based the best model which is the Regularized 
#   Model
Model_mu <- mean(edx$rating)
Model_b_i <- b_i
Model_b_u <- b_u

# ***************************************************************************************
# Now that we have developed the model using the training (edx) data,
#   Run the model on the validation data now to see the results
#   Note that we keep all the model parameters developed from the training set (edx)
#   and none of the model is based on any data from the validation set.

# First, add the model parameters to the validation data set (create a new data set)
ModelValidation <- validation %>% left_join(Model_b_i, by = "movieId") %>%
  left_join(Model_b_u, by = "userId")
# Model Validation now contains the original validation data + the model's b_i and b_u

# Now do the prediction on the validation dataset using the prediction model developed on the
#   training (edx) dataset
Validation_predicted_ratings <- ModelValidation %>% 
  mutate(pred = Model_mu + b_i + b_u) %>% pull(pred)

# And check the performance by calculation the RMSE of the predicted ratings versus the 
#   actual validation ratings.
Validation_rmse <- RMSE(Validation_predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Validation: ",  
                                     RMSE = Validation_rmse ))
rmse_results %>% knitr::kable()

# The RMSE for the Validation data set with the final model is 0.86522 which is not as 
#   quite as good as for the edx (training) data (which is as expected) but still much 
#   better than the initial average only, user based and movie based models for the
#   original edx training data set. 
#
# print the final model 
print ("The final model is represented by Y[u,i] = mu_hat + b_i + b_u")
print ("where mu_hat is the overall movie average rating, b_i is the movie term (for a particular movie)")
print (" and b_u is the user rating for a particular movie")
print ("The model was developed using the edx data set as directed in the Capstone requirements")
print ("  and run against the validation data set")
print ("The model's RMSE when run against the validation set was 0.86522")
print ("which is better than (below) the thershold noted in the instructions (0.8775) to")
print ("get the full 25 points awarded")
# Document conclusions
# *****************************************************************************************
# In conclusion, the final model Y[u,i] = mu_hat + b_i + b_u where mu_hat is the overall
#   average rating for all movies, b_i is the average for a particular movie and 
#   b_u which is the average for a particular movie represents a reasonable model
#   to use to predict ratings.  The model was run created based on the edx training set
#   and validated using the validation set as defined in the Capstone requirements using
#   the given R script to download and setup the edx and validation data sets.
#   The overall RMSE for the validation set is 0.86522 which is below the threshold 
#   as noted in the Rubric to be awarded 25 points.  
# *******************************************************************************************
 

# *******************************************************************************************

# As noted above, I had tried to utilize the recommenderlab package to develop the models
#  but because of limited memory on my machine, I could not run the recommenderlab 
#  routines on the edx data set as required in the Capstone instructions
#
# In order to familarize my self with the recommenderlab package, I decided to utilze a 
#  smaller dataset (the validation dataset) and use the recommenderlab package to create 
#  a model and check the accuracy of it.  
#
# NOTE THAT THE FOLLOWING CODE IS NOT INCLUDED NOT INTENDED TO BE USED TO JUDGE THE 
#   RMSE OF THE MODEL AS NOTED IN THE CAPSTONE DIRECTION BUT IS INCLUDED AS A REFERENCE
#   TO DETAIL HOW THE RECOMMENDERLAB PACKAGE COULD BE UTILIZED ON A PC WITH ADDITIONAL 
#   MEMORY AND CPU POWER
#
# ******************************************************************************************

# Start recommenderlab reference section
#  Note that if you are running this on a limited memory PC it would be advisable to 
#  clear the session if you have run the code above to free up additional memory
#  If you clear the session, you must reload the validation data set from disk (or rerun)
#  the code above that creates the validation data set initially
# 
# As noted, due to limited memory on my PC the full edx train set could not be utlized with 
#  the recommender lab package.  In order to understand how the package could be used,
#  I decided to use the smaller validation data set with the recommender package to 
#  create an UBCF and IBCF recommendation system and check the RMSE.  Note that the 
#  recommenderlab pacakge was created by Michael Hahsler for use in creating and testing
#  recommendation systems. 
# More information can be found at 
#   https://cran.r-project.org/web/packages/recommenderlab/vignettes/recommenderlab.pdf
# Cited as 
# Michael Hahsler (2019).  recommenderlab: Lab for Developing and Testing Recommender
#   Algorithms.  R pacakge version 0.2-4.
# https://github.com/mhahsler/recommenderlab

#  The following elements of the recommenderlab package were used:
#  1) evaluationScheme- Creates an evaluationScheme object from a dataset.  Manages
#     the splitting of data into training & test data.
#  2) Recommender- Learns a model from data
#  3) predict- given a model (from recommender) and a new dataset, creates predictions 
#       for the new data set.  A number of different prediction 
#       types (topN list, full ratings) are supported
#   4) getdata- used to extract data from teh evaluationScheme object
#
#  Note that the pacakge makes use of a ratingsMatrix or realratingsMatrix type 
#   which are sparse matrices with users as the rows and item ID's (in our case, movies)
#   as the columns.



# First load the required Libaries

library(tidyverse)
library(caret)
library(recommenderlab)
library(reshape2)
library(dplyr)
library(knitr)

# Only use the validationdata set at first since it is smaller and we are just exploring

# As previously noted, the edx and validation datasets were created and saved to disk as a 
#   csv file to expidite exploring the data and creating the model across sessions and avoid
#   having to download and recreate the data each time a new session was started.
#edx <- read.csv("edxdata.csv", stringsAsFactors = FALSE)
# Choose the validationdata set as it is smaller and can be used on a limited memeory PC to
#   run recommenderlab routines
# Read the dataset from disk (note that working directory must be set to the location of the .csv file)
#  by using setwd("...path...")  where ... path is the location of the .csv file
validation <- read.csv("validationdata.csv", stringsAsFactors = FALSE)

# Note that the read.csv function inserts an index as the first field when it reads so that
#   must be removed.
validation <- validation[,-1]
# Explore the data to understand how it is structured and what it contains
summary.default(validation)

# look at the first few elements of the data
kable(head(validation, n=5))

# And determine how many users and movies are in the dataset
validation %>%
  summarize(n_users = n_distinct(userId),
            n_movies = n_distinct(movieId))
# 68,534 users and 9,809 movies in validation

# See how the distribution is for users; Some rate lots of movies and some don't
validation %>% 
  count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Validation Users distribution")

# Look at the distribution for the movies; how many ratings each movie has
validation %>% 
  count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Validation Movie distribution")


# In order to have a good recommender, we need to insure we have users with 
#   a minimum number (20) of ratings and movies with a minimum # (50) of ratings

# First look filter for users with ratings > 20
# Create a new variable to hold the "cooked" validation data (after filtering)
cookedvalid <- validation %>% group_by(userId) %>% filter(n() > 20)

# Look at the distribution again
cookedvalid %>% 
  count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Cooked after user filter")


# Note that the filter changed the type which creates issues
# with other commands like summarize & the as(, "realratingsMatrix")
# Coerce back into data frame 
cookedvalid <- as.data.frame(cookedvalid)

# Now filter for movies that have more than 50 ratings

cookedvalid <- cookedvalid %>% group_by(movieId) %>% filter(n() > 50)

# Look at the distribution again
cookedvalid %>% 
  count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Cooked Movies")

# convert back into dataframe
cookedvalid <- as.data.frame(cookedvalid)

# Determine how many users and movies are now in the cooked data
cookedvalid %>%
  summarize(n_users = n_distinct(userId),
            n_movies = n_distinct(movieId))
# 13348 users and 2582 movies

# Now that we have insured that the data is appropriate for modeling, we can take the 
#   following steps to create and test the model
# Step 1: Wrangle the data to only have the fields of interest (user, movie & rating)
# Step 2: Convert the data into a realratingsMatrix
# Step 3: Setup an evaluationScheme with the data
# Step 4: Create the model using Recommender (note we will create both UBCF and IBCF models)
# Step 5: Create the predictions using predict
# Step 6: Evaluate the results using calcPredictionAccuracy
# Step 7: Display the RMSE

# the Recommenderlab routines use a structure called ratingsMatrix (binary data)
#  or realRatingMatrix (real numbers such as ratings data) as noted above.  The cooked data
#  frame must be converted into a realRatingMatrix before using the models functions
# Note that for our modeling purposes, we are only interested in 
#   the user, movie & the ratings so step 1 is to filter all other columns out

md <- cookedvalid %>% select("userId", "movieId", "rating")
kable(head(md, n=5))

# Step 2 is to convert the data into a realratingsMatrix for use by the recommendlab
#  routines

cooked_rrm <- as(md, "realRatingMatrix")


# Step 3: Setup an evaluation scheme for the models.  Note that we are using 90% of the data
#   for training and 10% for test.
ev <- evaluationScheme(cooked_rrm, method="split", train=0.9, given = 5, goodRating=5)

# Step 4: Create the models.  The Recommender method takes training data and several parameters 
#  and returns a trained model; Note that UBCF will return a User Based model and IBCF will
#  return an Item based model
r1 <- Recommender(getData(ev, "train"), "UBCF")
r1
# Save the model to disk to save time if need to load in a new session
saveRDS(r1, "r1UBCFValid")

# Create a Item based model for comparision (note this takes several minutes to run)
r2 <- Recommender(getData(ev, "train"), "IBCF")
r2
# Save the model
saveRDS(r2, "r2IBCFValid")

# Now perform the prediction  for the user based model
p1 <- predict(r1, getData(ev, "known"), type="ratings")
p1
# and save the prediction for later use if needed
saveRDS(p1, "p1UBCFPredict")
# Perform the prediction for the item based model
p2 <- predict(r2, getData(ev, "known"), type="ratings")
p2
saveRDS(p2, "p2IBCFPredict")

# note that since the models & the predictions have been saved, they can be loaded 
#  for additional analysis via the loadRDS(...) command

# Calculate the error now using unknown data from the evaluation scheme
#  for both the UBCF and IBCF models
error<- rbind(UBCF = calcPredictionAccuracy(p1, getData(ev, "unknown")),
              IBCF = calcPredictionAccuracy(p2, getData(ev,"unknown")))

error

# Note that UBCF model's RMSE was 1.03226 and the IBCF was 1.3199.

# End recommenderlab
# ***************************************************************************************

