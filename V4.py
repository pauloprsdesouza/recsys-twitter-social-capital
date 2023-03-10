def create_user_profile(rated_tweets):
    # Initialize user profile
    user_profile = {
        'keywords': [],
        'authors': [],
        'topics': []
    }

    # Update user profile based on rated tweets
    for tweet_id, rating in rated_tweets.items():
        # Get tweet object
        tweet = api.get_tweet(tweet_id)

        # Update keywords
        keywords = extract_keywords(tweet.text)
        for keyword in keywords:
            if rating >= 3:
                user_profile['keywords'].append(keyword)
            elif keyword in user_profile['keywords']:
                user_profile['keywords'].remove(keyword)

        # Update authors
        author_id = tweet.author_id
        if rating >= 3 and author_id not in user_profile['authors']:
            user_profile['authors'].append(author_id)
        elif rating <= 2 and author_id in user_profile['authors']:
            user_profile['authors'].remove(author_id)

        # Update topics
        topics = extract_topics(tweet.text)
        for topic in topics:
            if rating >= 3:
                user_profile['topics'].append(topic)
            elif topic in user_profile['topics']:
                user_profile['topics'].remove(topic)

    # Remove duplicates
    user_profile['keywords'] = list(set(user_profile['keywords']))
    user_profile['authors'] = list(set(user_profile['authors']))
    user_profile['topics'] = list(set(user_profile['topics']))

    return user_profile
