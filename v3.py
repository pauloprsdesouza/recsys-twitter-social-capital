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
import math
from pytwitter.models import User, Tweet, TweetEntities, TweetEntitiesUrl, TweetPublicMetrics, TweetEntitiesMention, TweetEntitiesHashtag, TweetEntitiesAnnotation
from pytwitter import Api
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

stemmer = SnowballStemmer('english')
stop_words = set(stopwords.words('english'))

api = Api(bearer_token="AAAAAAAAAAAAAAAAAAAAAMhqlAEAAAAA4Pqzn354Z5nlkP5lKaW98vzlVlA%3D7GIA03xacVKdFYTFg7qmgvWTZThpa2FFd4SNPUqP7uPK7Xjue5")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def preprocess_tweet(text):
    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove mentions and hashtags
    text = re.sub(r"@\S+", "", text)
    text = re.sub(r"#\S+", "", text)

    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stop words
    tokens = [token for token in tokens if not token in stop_words]

    # Lemmatize
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
        lemmas.append

    # Disambiguate polysemous words
    tokens = disambiguate_polysemous_words(tokens)

    # Expand synonyms
    tokens = expand_synonyms(tokens)

    return tokens


def disambiguate_polysemous_words(tokens):
    # Disambiguate polysemous words using the Lesk algorithm
    disambiguated_tokens = []
    for token in tokens:
        synsets = wordnet.synsets(token)
        if synsets:
            synset = synsets[0]
            context = ' '.join(tokens)
            sense = nltk.wsd.lesk(context, token, synset.pos(), synsets)
            if sense:
                disambiguated_tokens.append(sense.name().split('.')[0])
            else:
                disambiguated_tokens.append(token)
        else:
            disambiguated_tokens.append(token)

    return disambiguated_tokens


def expand_synonyms(tokens):
    # Expand synonyms using WordNet
    synonyms = []
    for token in tokens:
        synsets = wordnet.synsets(token)
        if synsets:
            synset = synsets[0]
            for lemma in synset.lemmas():
                synonym = lemma.name().lower()
                if synonym not in synonyms and synonym != token:
                    synonyms.append(synonym)
    return tokens + synonyms

def get_sentiment_score_v2(text):
    # Use BERT to calculate the sentiment score of the tweet text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    inputs.to(device)
    outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)
    sentiment_score = probs[0][1].item() - probs[0][0].item()

    return sentiment_score

def calculate_diversity_score(words):
    # Use a measure of lexical diversity to calculate the diversity score of the tweet text
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


def calculate_context_score(text, topic_keywords):
    # Calculate the context score of the tweet based on its relevance to the given topic keywords
    # For example, using the cosine similarity between the tweet and the topic keywords
    vectorizer = TfidfVectorizer()
    corpus = [text] + topic_keywords
    X = vectorizer.fit_transform(corpus)
    cosine_similarities = np.dot(X[0], X[1:].T).toarray()[0]
    context_score = max(cosine_similarities)

    return context_score

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
            sentiment = get_sentiment_score_v2(mention.text)
            if sentiment > 0:
                positive_sentiments += 1
            elif sentiment < 0:
                
                negative_sentiments += 1
                
    for reply in replies.data:
        if reply.author_id != user.id:
            sentiment = get_sentiment_score_v2(reply.text)
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

def user_strength_score(user, usersById):
    influence_score = usersById[user]["influence_score"]
    reputation_score = usersById[user]["reputation_score"]
    
    return reputation_score * influence_score

def calculate_social_capital_score(tweet: Tweet, usersById, usersByUserName):
    # Extract relevant information from the tweet
    text = tweet.text
    attachments = tweet.attachments
    public_metrics = tweet.public_metrics
    created_at = tweet.created_at

    # Preprocess tweet text
    tokens = preprocess_tweet(text)

    # Calculate sentiment score
    sentiment_score = get_sentiment_score_v2(text)

    # Calculate diversity score
    diversity_score = calculate_diversity_score(tokens)

    # Calculate recency score
    recency_score = calculate_recency_score(created_at)

    # Calculate context score
    context_score = calculate_context_score(text, tokens)

    # Calculate social capital score
    likes = public_metrics.like_count
    retweets = public_metrics.retweet_count
    replies = public_metrics.reply_count
    impressions_count = public_metrics.impression_count
    num_hashtags = len(tweet.entities.hashtags)
    num_urls = len(tweet.entities.urls)

    mention_users_strenght_score = 0
    if tweet.entities.mentions is not None:
      for mention in tweet.entities.mentions:
        mention_users_strenght_score += user_strength_score(mention.username)

    num_media = 0
    if attachments is not None and attachments.media_keys is not None:
        num_media = len(attachments.media_keys)
    
    social_capital_score = ((user_strength_score(tweet.author_id) + retweets + likes + replies + impressions_count + num_media + num_hashtags + num_urls + diversity_score + mention_users_strenght_score + len(tokens) + context_score) * recency_score)

    return {'tweet': tweet, 'score': social_capital_score }

def get_initialization():
    public_tweets = api.search_tweets(query="recommender systems lang:en has:hashtags -is:retweet has:media", expansions=["referenced_tweets.id.author_id","in_reply_to_user_id","attachments.media_keys","author_id","entities.mentions.username"], 
                                  user_fields=["created_at","entities","id","location","name","pinned_tweet_id","profile_image_url","protected","public_metrics","url","username","verified"],
                                  tweet_fields=["attachments","author_id","context_annotations","created_at","entities","geo","in_reply_to_user_id","lang","public_metrics","reply_settings","source"], max_results=50)

    public_tweets.data[0].text = "just read an interesting article about the importance of social capital in recommender systems. It's amazing how much impact a user's social connections can have on the quality of recommendations they receive! #socialcapital #recommendations #AI"
    public_tweets.data[1].text = "Excited to share our latest research on the role of social capital in improving recommender systems! Our findings show that incorporating user social networks can lead to significant improvements in recommendation quality. #research #socialcapital #recommendations"

    usersById = {}
    usersByUserName = {}
    for user in public_tweets.includes.users:
       influence_score = calculate_influence(user)
       reputation_score = calculate_reputation(user)
       usersById[user.id] = {"influence_score": influence_score, "reputation_score": reputation_score}
       usersByUserName[user.username] = {"influence_score": influence_score, "reputation_score": reputation_score}

    for tweet in public_tweets.data:
        user_id = tweet.author_id
        if user_id not in usersById:
            user = api.get_user(user_id)
            influence_score = calculate_influence(user)
            reputation_score = calculate_reputation(user)
            usersById[user_id] = {"influence_score": influence_score, "reputation_score": reputation_score}
            usersByUserName[user.username] = {"influence_score": influence_score, "reputation_score": reputation_score}
    
    return public_tweets, usersById, usersByUserName

def generate_recommendations(public_tweets, usersById, usersByUserName):
    ranking = {}
    for tweet in public_tweets.data:
        social_capital_score = calculate_social_capital_score(tweet, usersById, usersByUserName)
        ranking[tweet.id] = social_capital_score

    ranked = dict(sorted(ranking.items(), key=lambda item: item[1]['score'], reverse=True))    
    
    for tweetId in ranked: 
        print(ranking[tweetId]['score'], ranking[tweetId]['tweet'].id + "    ----  " + ranking[tweetId]['tweet'].text)
    
    return ranked

public_tweets, usersById, usersByUserName = get_initialization()

generate_recommendations(public_tweets, usersById, usersByUserName)