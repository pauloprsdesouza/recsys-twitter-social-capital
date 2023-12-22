# from Recommender import Recommender

# recommender = Recommender("AAAAAAAAAAAAAAAAAAAAALsjowEAAAAAX8xdgNjzcZzke0z0baOw8cXDRkc%3DbbhAd6Z6dIwX3iEYWlUukKGOb8tGIOalkw4CwiMOl9ff3fpZd8")
# recommendations = recommender.generate_recommendations()

# print(recommendations)

import json

from domains.Twitter.TwitteEntity import TweetEntity
from dynamodb_json import json_util as json

# Usage example
file_path = 'D:\Projects\twitter-phd-project\recsys-twitter-social-capital\infrastructure\Database\Inputs\4fkhdbpgle3nnfx26gy4jobtf4.json'

def read_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")

# Usage example
data = read_json_file(file_path)

tweets = json.loads(data)
# Print the tweets to check
for tweet in tweets:
    print(tweet)