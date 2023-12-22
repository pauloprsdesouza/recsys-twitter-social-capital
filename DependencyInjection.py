from dependency_injector import containers, providers
from domains.PreProcess.PreProcessServiceInterface import PreProcessServiceInterface
from domains.Recommendation.RecommendationServiceInterface import RecommendationServiceInterface
from services.Twitter.TwitterService import TwitterService

class Container(containers.DeclarativeContainer):
    twitter_service = providers.Factory(TwitterService)
    recommendation_service = providers.Factory(RecommendationServiceInterface)
    pre_process_service = providers.Factory(PreProcessServiceInterface)