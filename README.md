# Twitter Recommendation System

## Project Overview

This project is a Twitter recommendation system that provides suggestions based on the analysis of tweets and user metrics. The system evaluates the influence and reputation of users to generate recommendations, leveraging advanced NLP techniques and BERT models for tweet analysis.

## Files & Modules

### 1. main.py

The entry point for the application. Initializes the `Recommender` class and prints generated recommendations.

### 2. Recommender.py

Contains the main recommendation logic, including:
- Fetching tweets and user details using the `pytwitter` library.
- Calculating influence and reputation scores for users.
- Generating recommendations based on user metrics.

### 3. TweetAnalyzer.py

Handles the NLP and tweet analysis aspects, including:
- Preprocessing of tweets using the `nltk` library and BERT tokenizer.
- Text analysis using TF-IDF vectorization and BERT models.
- Additional utilities for text processing, such as stemming and lemmatization.

## Setup & Installation

1. Clone the repository.
2. Install the required Python packages:

```bash
pip install pytwitter nltk transformers
