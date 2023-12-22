class TweetEntity:
    def __init__(self, data):
        self.pk = data["PK"]["S"]
        self.sk = data["SK"]["S"]
        self.diversity_score = float(data["DiversityScore"]["N"])
        self.sentiment_score = float(data["SentimentScore"]["N"])
        self.reply_count = int(data["ReplyCount"]["N"])
        self.type = data["Type"]["S"]
        self.author_id = data["AuthorId"]["S"]
        self.domains = data.get("Domains", {}).get("SS", [])
        self.quote_count = int(data["QuoteCount"]["N"])
        self.retweet_count = int(data["RetweetCount"]["N"])
        self.tokens = data.get("Tokens", {}).get("SS", [])
        self.entities = data.get("Entities", {}).get("SS", [])
        self.engagement_score = float(data["EngagementScore"]["N"])
        self.context_score = float(data["ContextScore"]["N"])
        self.mentions = data.get("Mentions", {}).get("SS", [])
        self.like_count = int(data["LikeCount"]["N"])
        self.impression_count = int(data["ImpressionCount"]["N"])
        self.text = data["Text"]["S"]
        self.id = data["Id"]["S"]
        self.created_at = data["CreatedAt"]["S"]
        self.social_capital_score = float(data["SocialCapitalScore"]["N"])
        self.recency_score = float(data["RecencyScore"]["N"])
        # Additional fields like Hashtags, Urls, etc. can be added here

    def __repr__(self):
        return f"Tweet(pk={self.pk}, author_id={self.author_id}, text={self.text})"