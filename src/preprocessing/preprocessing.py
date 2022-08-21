import json
import re
from abc import ABC, abstractmethod
from collections import Counter

import emoji
import nltk
import numpy as np
import pandas as pd
import spacy
from enelvo.normaliser import Normaliser
from IPython.display import clear_output
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from sklearn.base import BaseEstimator, TransformerMixin

nltk.download('stopwords')


###################################################
# !python -m spacy download pt_core_news_sm

class BaseTreater(ABC):
    '''
    Base class for Treaters
    '''
    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def transform(self):
        pass


class TreaterMixIn:
    '''
    Implement common methods for Treaters
    '''

    BASE_PATH = './data-utils/'
    
    @staticmethod
    def get_filename(var, default):
        if var is None:
            return TreaterMixIn.BASE_PATH + default
        else:
            return var

    @staticmethod
    def save(object, filename):
        with open(filename, 'w') as file:
            json.dump(object, file)

    @staticmethod
    def load(filename):
        try:
            with open(filename, 'r') as file:
                return json.load(file)
        except IOError as e:
            raise IOError('File not found, most likely the transformer has not been fitted.')



class EmojisTreater(BaseTreater, TreaterMixIn):
    '''
    Emojis Treatment

    Most frequent EMOJIS are translated to words, while the least frequent ones are discarted.

    Params:
        tokenizer = tokenizer object to be used to tokenization.
        mapdict_rank_file = (str | None) - Json file mapping ranked emojis to words
        mapdict_file = (str | None) - Json file containing all the translations available 
    '''

    def __init__(self, tokenizer, mapdict_rank_file = None, mapdict_file=None):
        self.fullemjs_dict = emoji.UNICODE_EMOJI['pt'] 
        self.tokenizer = tokenizer
        self.rank = None
        self.mapdict_rank_file = self.get_filename(mapdict_rank_file, 'mapdict_rank.json')
        self.mapdict_file = self.get_filename(mapdict_file, 'emojis_map_dict.json')


    def transform(self, tweet_tkz):
        '''
        Chain transformations.
        '''
        out = self._transform(tweet_tkz)
        tweet_tkz = self.remove_brs(out)
        return tweet_tkz


    def _transform(self, tweet_tkz):
        '''
        Replaces or removes emojis from tokenized tweet.
        The result depends on the frequency emojis rank and available translations. 
        '''
        mapdict_rank = self.load(self.mapdict_rank_file)

        new_tweet_tkz = []
        for token in tweet_tkz:
            if token in mapdict_rank:
                new_tweet_tkz.append(mapdict_rank[token])
            elif token in self.fullemjs_dict:

                pass
            else:
                new_tweet_tkz.append(token)
        return new_tweet_tkz


    def remove_brs(self, tweet_tkz, replacement = 'brasil'):
        '''
        After tokenazation 🇧🇷 are turned into '🇧', '🇷'. Replacing them for "brasil".
        '''
        for i, _ in enumerate(tweet_tkz):
            if tweet_tkz[i] in ('🇧','🇷'):
                if i+1 <= len(tweet_tkz):
                    if tweet_tkz[i+1] in ('🇧','🇷'):
                        tweet_tkz[i] = replacement
                        tweet_tkz[i+1] = ''
        tweet_tkz = [i for i in tweet_tkz if i != '']
        return tweet_tkz


    def fit(self, corpus, limit=40):
        '''
        Return translation dictionary for the most frequent emojis
        '''
        mapdict = self.get_mapdict()
        top_rank_emjs = [i for i, _ in self.get_rank(corpus, limit)]
        mapdict_rank = { k:v for k,v in  mapdict.items() if k in top_rank_emjs} 

        self.save(mapdict_rank, self.mapdict_rank_file)
        return self


    def get_mapdict(self):
        '''
        Return a emojis dictionary mapping and their respective word to be replaced in the documents.
        '''
        mapdict = self.load(self.mapdict_file)

        return { emoji: mapdict.get(item.replace(':',''))
                for emoji, item in self.fullemjs_dict.items() 
                if mapdict.get(item.replace(':','')) is not None 
            }


    def get_rank(self, corpus, limit=40):
        '''
        Rank the most frequent emotions in the corpus.
        Returns a list of tuples.
        '''
        list_of_lists = [self._extract_emojis(token)
                        for token in corpus 
                        if not isinstance(self._extract_emojis(token), float)]

        flat_list = [item for sublist in list_of_lists for item in sublist]
        c = Counter(flat_list)
        return c.most_common(limit)


    def _extract_emojis(self, tweet):
        '''
        Return a list of emotions in a tweet.
        '''
        emjs_dict = emoji.UNICODE_EMOJI['pt'] 
        emjs = []
        tweet_tkz = self.tokenizer.tokenize(tweet)
        for token in tweet_tkz:
            if token in emjs_dict:
                emjs.append(token)
        if len(emjs)==0:
            return np.nan
        return emjs


class LinkedWordsTreater(BaseTreater):
    '''
    Try to tokenze words without space
    '''

    def fit(self):
        return self

    def transform(self, tweet_tkz):
        ''' 
        Tokenize words without spaces.
        '''
        new_tweet_tkz = []
        for token in tweet_tkz:
            token = self._split_joined_words(token)
            for sub_token in token:
                new_tweet_tkz.append(sub_token)
        
        return new_tweet_tkz


    def _split_joined_words(self, token):
        '''
        Rertuns a list of splited tokens, if splitable.
        '''
        if not token.isupper():
            return re.sub( r"([A-Z])", r" \1", token).split()

        return [token]



class UsersTreater(BaseTreater, TreaterMixIn):
    '''
    User Treatment

    Most frequent users are kept, while the least frequent ones are discarted.
    '''

    def __init__(self, ranking_file = None):
        self.ranking_file = self.get_filename(ranking_file, 'users_ranking.json')
        self.rank = None

    def fit(self, corpus, ranking_size = 40):
        '''
        Fit Treater to data.
        '''
        rank = self._get_rank(corpus, ranking_size)
        self.rank = [i for i, _ in rank]
        self.save(self.rank, self.ranking_file)
        return self
    

    def _get_rank(self, corpus, ranking_size):
        '''
        Returns rank (list of tuples) of most the frequent users in the corpus
        '''
        frame = pd.DataFrame({'corpus': corpus})
        frame = frame[frame.corpus.str.contains('@')]
        frame['at_'] = frame.corpus.apply(lambda doc: [word for word in doc.split() if '@' in word])
        flatten = [num for elem in frame.at_.tolist() for num in elem]
        rank = Counter(flatten).most_common(ranking_size)
        return rank  


    def transform(self, tweet_tkz):
        '''
        Apply functions to treat users
        '''
        if self.rank is None:
            self.rank = self.load(self.ranking_file)

        tweet_tkz = self._remove_ats_frequet_users(tweet_tkz)
        return self._remove_users(tweet_tkz)



    def _remove_ats_frequet_users(self, tweet_tkz):
        '''
        Remove ats from frequent users, so they are kept in the corpus.
        '''
        intersection_set = set(tweet_tkz).intersection(set(self.rank))
        if len(intersection_set) != 0 :
            for user in intersection_set:
                tweet_tkz = [token.replace('@','')  if token == user else token for token in tweet_tkz]
        return tweet_tkz


    def _remove_users(self, tweet_tkz):
        '''
        Remove users with @.
        '''
        tweet = re.sub('@[^\s]+','user',' '.join(tweet_tkz))
        tweet_tkz = [token for token in tweet.split(' ') if token != 'user']
        return tweet_tkz



class StopwordsTreater(BaseTreater, TreaterMixIn):
    '''Remove stopwords'''

    def __init__(self, filename = None):
        self.stopwords = self.get_stopwords(filename)

    def get_stopwords(self, filename):
        filename = self.get_filename(filename, 'stopwords.json')
        custom_stopwords = self.load(filename)
        nltk_stopwords = set(stopwords.words('portuguese'))
        return nltk_stopwords.union(set(custom_stopwords))

    def fit(self):
        return self

    def transform(self, tweet_tkz):
        return [token for token in tweet_tkz if token.lower() not in self.stopwords]



def treat_kkk(tweet_tkz):
    ''' 
    Replace laugh kkkk for "risada".
    '''
    return [ 'risada'  if token in ['kkk'+i*'k' for i in range(25)] else token for token in tweet_tkz ]



class TweetsTextPreprocessor(BaseEstimator, TransformerMixin, TreaterMixIn):
    '''
    Custom sklearn transformer to preprocess tweets text.
    '''
    def __init__(self, verbose = False):
        self.tokenizer = TweetTokenizer()
        self.verbose = verbose
        self.norm = Normaliser(tokenizer='readable')
        self.spc = spacy.load("pt_core_news_sm")
        self.treat_users = UsersTreater()
        self.treat_emojis = EmojisTreater(self.tokenizer)
        self.treat_stopwords = StopwordsTreater()
        self.tokenize_joined_words = LinkedWordsTreater()

    def fit(self,X,y=None):
        return self


    def transform(self,X,y=None):
        
        bag = []
        for i, tweet in enumerate(X):
            if self.verbose:
                clear_output(wait=True)
                print(round(i/len(X)*100,2), " %")

            tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','url',tweet)
            tweet = tweet.replace('R$','reais').replace('-','').replace('url','')
            
            tweet = self.tokenizer.tokenize(tweet)      # Tokenizing
            
            tweet = self.treat_users.transform(tweet)
            tweet = self.treat_emojis.transform(tweet)
            tweet = self.treat_stopwords.transform(tweet)

            tweet = re.sub(r'[^\w\s]','',' '.join(tweet)).split(' ')    # Clean up symbols and punctuation
            tweet = [token for token in tweet if token != '']           # Remove empty strings

            tweet = [token.lemma_.lower() for token in self.spc(' '.join(tweet))]   # Lematizing
            tweet = treat_kkk(tweet)
            tweet = self.tokenize_joined_words.transform(tweet)
            # tweet = self.norm.normalise(' '.join([token.lower() for token in tweet])).split(' ') # Normalizer
            tweet = self.treat_stopwords.transform(tweet)

            # Special corrections
            tweet = [token if token != 'suboficial' else 'lulaoficial' for token in tweet ]
            tweet = [token if token != 'vagar' else 'vagabundo' for token in tweet ]
            tweet = [token if token != 'firsar' else 'risada' for token in tweet ]


            if tweet == []:
                tweet = ['null']
            bag.append(' '.join(tweet))

        return bag # pd.DataFrame({'text':bag})


    
        



    