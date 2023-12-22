from abc import ABC, abstractmethod

class RecommendationServiceInterface(ABC):
    @abstractmethod
    def rank_tweets(self, tweets):
        pass