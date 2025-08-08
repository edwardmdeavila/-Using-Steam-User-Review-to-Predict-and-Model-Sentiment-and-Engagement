# -Using-Steam-User-Review-to-Predict-and-Model-Sentiment-and-Engagement
 analyze over 5 million Steam game reviews to understand user sentiment, identify market trends, and provide data-driven insights for game developers, marketers, and platform managers.


 #### Accessing DB ####
rm(list=ls(all=T)) 

require(RPostgres)
require(DBI)
library(purrr)
library(dplyr)

## Create connection
con <- dbConnect(RPostgres::Postgres()
                  ,user="cp_user"
                  ,password="read123"
                  ,host="localhost"
                  ,port=5432
                  ,dbname="GP_DP"
)

# Fetch individual tables
players_st <- dbGetQuery(con, "SELECT * FROM players_st")
games_st <- dbGetQuery(con, "SELECT * FROM games_st")
reviews_st <- dbGetQuery(con, "SELECT * FROM reviews_st")


# Close the connection
dbDisconnect(con)

# Now you can proceed with sentiment analysis on `joined_data`

# Fetch the data and perform the joins in a single query
# Fetch the data and perform the joins in a single query
query <- "
SELECT
    r.reviewid,
    r.playerid,
    r.gameid,
    r.review,
    r.helpful,
    r.funny,
    r.awards,
    r.posted,
    p.country,
    p.created,
    g.title,
    g.developers,
    g.publishers,
    g.genres,
    g.supported_languages,
    g.release_date
FROM
    reviews_st r
JOIN
    players_st p ON r.playerid = p.playerid
JOIN
    games_st g ON r.gameid = g.gameid;
"

# Fetch data
data <- dbGetQuery(con, query)
head(data)

#### Overall Sentiment ####
# Perform sentiment analysis
install.packages("tidytext")
library(tidytext)

reviews <- data %>% select(review)
reviews_tidy <- reviews %>% unnest_tokens(word, review)
sentiment <- reviews_tidy %>% inner_join(get_sentiments("bing"))

# Visualize sentiment
sentiment_summary <- sentiment %>% count(sentiment)
ggplot(sentiment_summary, aes(x = sentiment, y = n, fill = sentiment)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(title = "Sentiment Analysis of Steam Reviews", x = "Sentiment", y = "Count")


#### By Genre ####
# Perform sentiment analysis and join with genre
reviews_sentiment <- data %>%
  select(gameid, genres, review) %>%
  unnest_tokens(word, review) %>%
  inner_join(get_sentiments("bing"))

# Calculate sentiment score by game genre
sentiment_by_genre <- reviews_sentiment %>%
  group_by(gameid, genres) %>%
  summarise(sentiment_score = sum(ifelse(sentiment == "positive", 1, -1))) %>%
  ungroup() %>%
  group_by(genres) %>%
  summarise(average_sentiment_score = mean(sentiment_score),
            number_of_games = n_distinct(gameid)) %>%
  arrange(desc(average_sentiment_score))

# Print the sentiment score by game genre table
print(sentiment_by_genre)

#### Top 10 Best/Worst ####
# Get the top 10 best genres
top_10_best_genres <- sentiment_by_genre %>%
  slice_head(n = 10) %>%
  select(genres, average_sentiment_score)

cat("Top 10 Best Genres by Sentiment Score:\n")
print(top_10_best_genres)
cat("\n")

# Get the top 10 worst genres
top_10_worst_genres <- sentiment_by_genre %>%
  arrange(average_sentiment_score) %>%
  slice_head(n = 10) %>%
  select(genres, average_sentiment_score)

cat("Top 10 Worst Genres by Sentiment Score:\n")
print(top_10_worst_genres)

top_genres <- bind_rows(top_10_best_genres, top_10_worst_genres)

# Create the ggplot plot
ggplot(top_genres, aes(x = reorder(genres, average_sentiment_score),
                       y = average_sentiment_score,
                       fill = type)) + # Use type for color fill
  geom_bar(stat = "identity") +
  coord_flip() + # Flip the axes for better readability
  labs(title = "Top 10 Best and Worst Genres by Sentiment Score",
       x = "Genre",
       y = "Average Sentiment Score",
       fill = "Sentiment") + # Change legend title
  theme_minimal() +
  facet_wrap(~type, scales = "free_y") # Separate the best and worst genres into separate panels


#### Sentiment Regression Model ####
# Aggregate sentiment score per game
game_sentiment <- data %>%
  select(gameid, review) %>%
  unnest_tokens(word, review) %>%
  inner_join(get_sentiments("bing")) %>%
  group_by(gameid) %>%
  summarise(average_sentiment_score = sum(ifelse(sentiment == "positive", 1, -1)) / n())  # Calculate average

# Prepare data for modeling
model_data <- data %>%
  distinct(gameid, .keep_all = TRUE) %>%
  select(gameid, genres, developers, publishers,release_date) %>%
  mutate(
    release_year = year(ymd(release_date))
  ) %>%
  select(-release_date) %>%
  left_join(game_sentiment, by = "gameid")  # Join aggregated sentiment

# Convert categorical variables to factors and handle missing values
model_data <- model_data %>%
  mutate(
    genres = as.factor(genres),
    developers = as.factor(developers),
    publishers = as.factor(publishers),
    average_sentiment_score = ifelse(is.na(average_sentiment_score), 0, average_sentiment_score) #impute missing sentiment
  )

# Split data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(model_data$average_sentiment_score, p = 0.8, list = FALSE)
train_data <- model_data[trainIndex, ]
test_data <- model_data[-trainIndex, ]

# Build a linear regression model
model <- lm(average_sentiment_score ~ genres + developers + publishers + release_year, data = train_data)

# Summarize the model
summary(model)

# Make predictions on the test set
predictions <- predict(model, newdata = test_data)

# Evaluate the model performance
rmse <- sqrt(mean((predictions - test_data$average_sentiment_score)^2))
cat("Root Mean Squared Error (RMSE):", rmse, "\n")

# Visualize predictions vs. actual
plot(test_data$average_sentiment_score, predictions,
     xlab = "Actual Sentiment Score", ylab = "Predicted Sentiment Score",
     main = "Actual vs. Predicted Sentiment Score")
abline(a = 0, b = 1, col = "red")  # Add a diagonal line
