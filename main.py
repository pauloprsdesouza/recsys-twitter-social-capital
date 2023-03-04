import re
import string
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import datetime
import emoji
from pytwitter import Api
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import math
from pytwitter.models import User;
from pytwitter.models import Tweet;
import datetime
import emoji

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw')
nltk.download('averaged_perceptron_tagger')

stemmer = SnowballStemmer('portuguese')
stop_words = set(stopwords.words('portuguese'))

api = Api(bearer_token="AAAAAAAAAAAAAAAAAAAAAMhqlAEAAAAA4Pqzn354Z5nlkP5lKaW98vzlVlA%3D7GIA03xacVKdFYTFg7qmgvWTZThpa2FFd4SNPUqP7uPK7Xjue5")



def calculate_influence(user: User):
    follower_count = user.public_metrics.followers_count

    # Get user's tweet count and average engagement rate
    tweets = api.get_timelines(user.id, max_results=50, tweet_fields=["attachments","author_id","context_annotations","created_at","entities","geo","in_reply_to_user_id","lang","public_metrics","reply_settings","source"])
    tweet_count = len(tweets.data)
    total_engagement = 0
    
    for tweet in tweets.data:
        total_engagement += tweet.public_metrics.like_count + tweet.public_metrics.retweet_count + tweet.public_metrics.quote_count + tweet.public_metrics.reply_count
        
    if tweet_count > 0:
        avg_engagement_rate = total_engagement / (tweet_count * follower_count) if total_engagement > 0 and tweet_count > 0 and follower_count > 0 else 0
    else:
        avg_engagement_rate = 0



    # Calculate influence score
    influence_score = math.log(follower_count + 1, 10) * (avg_engagement_rate + 1)
    
    return influence_score


def calculate_reputation(user: User):
    # Get user's recent mentions and replies
    mentions = api.search_tweets(query=f"@{user.username}", max_results=50)
    replies = api.search_tweets(query=f"to:{user.username}", max_results=50)

    # Calculate reputation score based on sentiment analysis of mentions and replies
    positive_sentiments = 0
    negative_sentiments = 0
    
    for mention in mentions.data:
        if mention.author_id != user.id:
            sentiment = get_sentiment_score(mention.text)
            if sentiment > 0:
                positive_sentiments += 1
            elif sentiment < 0:
                
                negative_sentiments += 1
                
    for reply in replies.data:
        if reply.author_id != user.id:
            sentiment = get_sentiment_score(reply.text)
            if sentiment > 0:
                positive_sentiments += 1
            elif sentiment < 0:
                negative_sentiments += 1
                
    if (positive_sentiments + negative_sentiments) > 0:
        
        reputation_score = positive_sentiments / (positive_sentiments + negative_sentiments)
    else:
        reputation_score = 0
        
    fine_adjustment = 0.01
    normalized_reputation_score = (reputation_score + user.public_metrics.listed_count * fine_adjustment) / (1 + (user.public_metrics.listed_count * fine_adjustment))

    # Return influence and reputation scores
    return normalized_reputation_score


def remove_urls(text):
    return re.sub(r"http\S+", "", text)

def remove_mentions(text):
    return re.sub(r"@\S+", "", text)

def remove_hashtags(text):
    return re.sub(r"#\S+", "", text)

def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def tokenize(text):
    return word_tokenize(text, language='portuguese')

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('portuguese'))
    return [token for token in tokens if not token in stop_words]

def lemmatize(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for token in tokens:
        lemma = lemmatizer.lemmatize(token, wordnet.VERB)
        if lemma == token:
            lemma = lemmatizer.lemmatize(token, wordnet.NOUN)
        if lemma == token:
            lemma = lemmatizer.lemmatize(token, wordnet.ADJ)
        if lemma == token:
            lemma = lemmatizer.lemmatize(token, wordnet.ADV)
        lemmas.append(lemma)
    return lemmas

def synonymize(tokens):
    synonyms = []
    for token in tokens:
        synsets = wordnet.synsets(token, lang='por')
        if synsets:
            synset = synsets[0]
            for lemma in synset.lemmas(lang='por'):
                synonym = lemma.name().lower()
                if synonym not in synonyms and synonym != token:
                    synonyms.append(synonym)
    return synonyms

def polysemmize(tokens):
    for word in tokens:
       pos = nltk.pos_tag(tokens)[0][1][0].lower()
       if pos not in ['n', 'v']:
           continue
        # Use simple_lesk to disambiguate the sense of the word
       synset = nltk.wsd.lesk(tokens, word, pos=pos)
    
       if synset is not None:
           # Replace the token with the lemma of the most likely sense
           word = synset.lemmas()[0].name().lower()
            
    return tokens

def preprocess_tweet(text):
    text = text.lower()
    text = remove_urls(text)
    text = remove_mentions(text)
    text = remove_hashtags(text)
    text = remove_punctuation(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    tokens = polysemmize(tokens)
    tokens.extend(synonymize(tokens))
    return set(tokens)



def calculate_social_capital(tweet: Tweet, usersById: dict):
    # Extract relevant information from the tweet
    tokens = preprocess_tweet(tweet.text)
    urls = re.findall('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', tweet.text)
    emojis = emoji.emoji_count(tweet.text)
    likes = tweet.public_metrics.like_count
    retweets = tweet.public_metrics.retweet_count
    replies = tweet.public_metrics.reply_count
    quotes = tweet.public_metrics.quote_count
    created_at = tweet.created_at
    hashtags = len(re.findall(r'#(\w+)', tweet.text))
    
    num_medias = 0
    if tweet.attachments is not None and tweet.attachments.media_keys is not None:
        for attachment in tweet.attachments.media_keys:
            num_medias += 1

    # Calculate the length of the tweet in characters
    length = len(tweet.text)

    # Calculate the sentiment score of the tweet
    sentiment_score = get_sentiment_score_v2(tweet.text)

    # Calculate the diversity score of the tweet
    diversity_score = calculate_diversity_score(tokens)

    # Calculate the number of resources in the tweet
    num_resources = len(urls) + emojis + num_medias
    
    # Calculate the recency score of the tweet
    recency_score = calculate_recency_score(created_at)

    # Calculate the social capital score of the tweet
    engagement = (likes + replies + quotes)
    social_capital_score = (retweets if retweets > 0 else 1) * ((engagement + num_resources + diversity_score + len(tokens) + hashtags + length + sum(usersById[tweet.author_id])) * recency_score)

    return {'tweet': tweet, 'score': social_capital_score } 

def get_sentiment_score(text):
    # Use a sentiment analysis library or model to calculate the sentiment score of the tweet text
    # For example, using TextBlob library
    from textblob import TextBlob

    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity

    return sentiment_score

def calculate_diversity_score(words):
    # Use a measure of lexical diversi
    # ty to calculate the diversity score of the tweet text
    # For example, using the type-token ratio (TTR) metric
    num_words = len(words)
    num_unique_words = len(set(words))
    diversity_score = num_unique_words / num_words

    return diversity_score

def calculate_recency_score(created_at):
    # Calculate the recency score of the tweet based on its age
    # For example, using a logarithmic decay function
    
    tweet_date = datetime.datetime.strptime(created_at, '%Y-%m-%dT%H:%M:%S.%fZ')
    now = datetime.datetime.now()
    
    age_in_seconds = (now - tweet_date).total_seconds()
    
    if age_in_seconds < 0: 
       age_in_seconds = (tweet_date - now).total_seconds()
    
    # Set the decay factor
    decay_factor = 0.1
    
    # Calculate the recency score using a logarithmic decay function
    recency_score = 1 / (1 + decay_factor * math.log10(1 + age_in_seconds))
    
    return recency_score

def generateRecommendations():
    public_tweets = api.search_tweets(query="allan dos santos lang:pt has:hashtags -is:retweet has:media", expansions=["referenced_tweets.id.author_id","in_reply_to_user_id","attachments.media_keys","author_id","entities.mentions.username"], 
                                  user_fields=["created_at","entities","id","location","name","pinned_tweet_id","profile_image_url","protected","public_metrics","url","username","verified"],
                                  tweet_fields=["attachments","author_id","context_annotations","created_at","entities","geo","in_reply_to_user_id","lang","public_metrics","reply_settings","source"], max_results=100, query_type='recent')

    usersByUsername = {}
    usersById = {}
    for user in public_tweets.includes.users:
       result = [calculate_influence(user), calculate_reputation(user)]
       usersByUsername[user.username] = result
       usersById[user.id] = result

    ranking = {}
    for tweet in public_tweets.data:          
       ranking[tweet.id] = calculate_social_capital(tweet, usersById)
    
    ranked = dict(sorted(ranking.items(), key=lambda item: item[1]['score'], reverse=True))
    return ranked	


print(generateRecommendations())
