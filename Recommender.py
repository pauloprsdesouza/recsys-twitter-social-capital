import math
import re
from TweetAnalyzer import TweetAnalyzer
from pytwitter.models import User, Tweet
from pytwitter import Api

class Recommender:
    def __init__(self, twitterApiToken):
        self.tweetAnalyzer = TweetAnalyzer()
        self.api = Api(bearer_token=twitterApiToken)

    def user_strength_score(self, user, users): 
        influence_score = users[user]["influence_score"]
        reputation_score = users[user]["reputation_score"]
    
        return reputation_score * influence_score
    
    def calculate_influence(self, user: User):
        follower_count = user.public_metrics.followers_count

        # Get user's tweet count and average engagement rate
        tweets = self.api.get_timelines(user.id, max_results=50, tweet_fields=["attachments","author_id","context_annotations","created_at","entities","geo","in_reply_to_user_id","lang","public_metrics","reply_settings","source"])
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


    def calculate_reputation(self, user: User):
        # Get user's recent mentions and replies
        mentions = self.api.search_tweets(query=f"@{user.username}", max_results=50)
        replies = self.api.search_tweets(query=f"to:{user.username}", max_results=50)

        # Calculate reputation score based on sentiment analysis of mentions and replies
        positive_sentiments = 0
        negative_sentiments = 0
        
        for mention in mentions.data:
            if mention.author_id != user.id:
                sentiment = self.get_sentiment_score(mention.text)
                if sentiment > 0:
                    positive_sentiments += 1
                elif sentiment < 0:
                    negative_sentiments += 1
                    
        for reply in replies.data:
            if reply.author_id != user.id:
                sentiment = self.get_sentiment_score(reply.text)
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

    def calculate_social_capital_score(self, tweet, usersById):
         # Extract relevant information from the tweet
        text = tweet.text
        attachments = tweet.attachments
        public_metrics = tweet.public_metrics
        created_at = tweet.created_at

        # Preprocess tweet text
        tokens = self.tweetAnalyzer.pre_process(text)

        # Calculate diversity score
        diversity_score = self.calculate_diversity_score(tokens)

        # Calculate recency score
        recency_score = self.calculate_recency_score(created_at)

        # Calculate context score
        context_score = self.calculate_context_score(text, tokens)

        # Calculate social capital score
        likes = public_metrics.like_count
        retweets = public_metrics.retweet_count
        replies = public_metrics.reply_count
        impressions_count = public_metrics.impression_count
        hashtags = len(re.findall(r'#(\w+)', text))
        
        mentions_strenth = 0
        
        if tweet.entities.mentions is not None:
            for mention in tweet.entities.mentions:
                mentions_strenth +=  self.user_strength_score(mention.username, usersById)
        
        num_media = 0
        if attachments is not None and attachments.media_keys is not None:
            num_media = len(attachments.media_keys)
        
        social_capital_score = (self.user_strength_score(tweet.author_id, usersById) + (retweets + likes + replies + impressions_count + num_media + diversity_score + hashtags + len(tokens) + context_score)) * recency_score

        return {'tweet': tweet, 'score': social_capital_score }

    def generate_recommendations(self):
        public_tweets = self.api.search_tweets(query="lula lang:pt has:hashtags -is:retweet has:media", expansions=["referenced_tweets.id.author_id","in_reply_to_user_id","attachments.media_keys","author_id","entities.mentions.username"], 
                                  user_fields=["created_at","entities","id","location","name","pinned_tweet_id","profile_image_url","protected","public_metrics","url","username","verified"],
                                  tweet_fields=["attachments","author_id","context_annotations","created_at","entities","geo","in_reply_to_user_id","lang","public_metrics","reply_settings","source"], max_results=100, query_type='recent')

        usersById = {}
        for user in public_tweets.includes.users:
            influence_score = self.calculate_influence(user)
            reputation_score = self.calculate_reputation(user)
            usersById[user.id] = {"influence_score": influence_score, "reputation_score": reputation_score}

        for tweet in public_tweets.data:
            user_id = tweet.author_id
            if user_id not in usersById:
                user = self.api.get_user(user_id)
                influence_score = self.calculate_influence(user)
                reputation_score = self.calculate_reputation(user)
                usersById[user_id] = {"influence_score": influence_score, "reputation_score": reputation_score}

        ranking = {}
        for tweet in public_tweets.data:
            social_capital_score = self.calculate_social_capital_score(tweet, usersById)
            ranking[tweet.id] = social_capital_score

        ranked = dict(sorted(ranking.items(), key=lambda item: item[1]['score'], reverse=True))    
        
        for tweetId in ranked: 
            print(ranking[tweetId]['score'], ranking[tweetId]['tweet'].id + "    ----  " + ranking[tweetId]['tweet'].text)
        
        return ranked