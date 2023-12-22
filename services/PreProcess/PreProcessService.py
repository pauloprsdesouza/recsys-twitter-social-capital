from domains.PreProcess.PreProcessServiceInterface import PreProcessServiceInterface
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertForSequenceClassification
import datetime
import math
import nltk
import numpy as np
import preprocessor as TweetProcessor
import ssl
import string
import torch

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw')
nltk.download('averaged_perceptron_tagger')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.to(device)
stemmer = SnowballStemmer('portuguese')
stop_words = set(stopwords.words('portuguese'))
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class PreProcessService(PreProcessServiceInterface):
	def __init__(self, stopwords=None):
		self.stemmer = SnowballStemmer('portuguese')
		self.stop_words = stopwords if stopwords else set()
		TweetProcessor.set_options(TweetProcessor.OPT.URL, TweetProcessor.OPT.EMOJI, TweetProcessor.OPT.MENTION, TweetProcessor.OPT.HASHTAG)
  	
	def pre_process(self, text):
		cleaned_text = TweetProcessor.clean(text.lower())

		translator = str.maketrans('', '', string.punctuation)
		cleaned_text = cleaned_text.translate(translator)

		tokens = word_tokenize(cleaned_text)

		tokens = [token for token in tokens if not token in stop_words]

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

		tokens = self.disambiguate_polysemous_words(tokens)

		tokens = self.expand_synonyms(tokens)

		return tokens
	def disambiguate_polysemous_words(tokens):
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
	def get_sentiment_score(text):
		inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
		inputs.to(device)
		outputs = model(**inputs)
		logits = outputs.logits
		probs = torch.softmax(logits, dim=1)
		sentiment_score = probs[0][1].item() - probs[0][0].item()

		return sentiment_score
	def calculate_diversity_score(words):
		num_words = len(words)
		num_unique_words = len(set(words))
		diversity_score = num_unique_words / num_words

		return diversity_score
	def calculate_context_score(text, topic_keywords):
		vectorizer = TfidfVectorizer()
		corpus = [text] + topic_keywords
		X = vectorizer.fit_transform(corpus)
		cosine_similarities = np.dot(X[0], X[1:].T).toarray()[0]
		context_score = max(cosine_similarities)

		return context_score
	def calculate_recency_score(date):
		tweet_date = datetime.datetime.strptime(date, '%Y-%m-%dT%H:%M:%S.%fZ')
		now = datetime.datetime.now()

		age_in_seconds = (now - tweet_date).total_seconds()

		if age_in_seconds < 0: 
			age_in_seconds = (tweet_date - now).total_seconds()

		decay_factor = 0.1

		recency_score = 1 / (1 + decay_factor * math.log10(1 + age_in_seconds))

		return recency_score