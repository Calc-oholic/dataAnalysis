from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import yfinance as yf
from GoogleNews import GoogleNews
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import nltk

nltk.download('vader_lexicon')

analyzer = SentimentIntensityAnalyzer()
news = GoogleNews(lang='en')

def fetch_news(ticker, date):
    news_date = date.strftime('%m/%d/%Y')
    next_day = date + timedelta(days=1)
    next_day_str = next_day.strftime('%m/%d/%Y')
    news.clear()
    news.set_time_range(news_date, next_day_str)
    news.get_news(f'{ticker}')
    return news.results()

def calculate_sentiment(xnews):
    scores = [analyzer.polarity_scores(article['title'])['compound'] for article in xnews]
    return np.mean(scores) if scores else 0

def fetch_stock_data(ticker, start_date, end_date):
    return yf.download(ticker, start=start_date, end=end_date)


def build_sentiment_dataset(ticker, days):
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)
    stock_data = fetch_stock_data(ticker, start_date, end_date)

    sentiment_scores = []
    dates = []
    price_changes = []

    for i in range(1, len(stock_data)):
        date = stock_data.index[i - 1].date()
        next_date = stock_data.index[i].date()

        xnews = fetch_news(ticker, date)
        sentiment = calculate_sentiment(xnews)

        if not np.isnan(stock_data['Adj Close'].pct_change().values[i]):
            sentiment_scores.append(sentiment)
            dates.append(next_date.strftime('%m/%d'))
            price_changes.append(stock_data['Adj Close'].pct_change().values[i])

    # match lengths
    return pd.DataFrame(
        {'Date': dates, 'Sentiment': sentiment_scores, 'Price Change': price_changes}
    )

# linear regression
def linear_regression_analysis(X, y):
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    return model, mse, r2

# random forest
def random_forest_analysis(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, mse, r2

# random forest validation
def cross_validation(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    return scores.mean()

# data analysis
def perform_analysis(ticker, days):
    df = build_sentiment_dataset(ticker, days)

    # prep
    X = df[['Sentiment']].values  # Using sentiment as the predictor
    y = df['Price Change'].values  # Target: price change

    # linear regression
    lin_reg_model, lin_reg_mse, lin_reg_r2 = linear_regression_analysis(X, y)
    print(f"Linear Regression MSE: {lin_reg_mse}, R2: {lin_reg_r2}")

    # random forest
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model, rf_mse, rf_r2 = random_forest_analysis(X_train, X_test, y_train, y_test)
    print(f"Random Forest MSE: {rf_mse}, R2: {rf_r2}")

    # random forest validation
    cv_score = cross_validation(X, y)
    print(f"Cross-Validation R2: {cv_score}")

ticker = 'HOG'
days = 25
perform_analysis(ticker, days)
