from domains.Recommendation.RecommendationServiceInterface import RecommendationServiceInterface

class RecommendaionService(RecommendationServiceInterface):
    def rank_tweets(self, tweets):
        return