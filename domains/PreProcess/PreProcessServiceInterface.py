from abc import ABC, abstractmethod

class PreProcessServiceInterface(ABC):
    @abstractmethod
    def pre_process(self, tweets):
        pass
    
    @abstractmethod
    def disambiguate_polysemous_words(tokens):
        pass
    
    @abstractmethod
    def expand_synonyms(tokens):
        pass
    
    @abstractmethod
    def get_sentiment_score(text):
        pass
            
    @abstractmethod
    def calculate_diversity_score(words):
        pass
    
    @abstractmethod 
    def calculate_context_score(text, topic_keywords): 
        pass
    
    @abstractmethod 
    def calculate_recency_score(date):
      pass