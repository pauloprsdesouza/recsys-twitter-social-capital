from abc import ABC, abstractmethod

class TwitterServiceInterface(ABC):
    @abstractmethod
    def fetch_tweets(self):
        pass