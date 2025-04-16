import os
import json
# from google.colab import files
import pandas as pd
import numpy as np
from itertools import product


def preprocess() -> list[str]:
    '''
    Return a list of characters, representing the shared vocabulary of all languages
    '''
    vocab = []
    data_files = [file for file in os.listdir('data') if file.endswith('.csv')]
    for file in data_files:
        with open(os.path.join('data', file), 'r') as f:
            data = pd.read_csv(f)
            for tweet in data['tweet_text']:
                vocab.extend(list(tweet))
    
    vocab.append('<start>')
    vocab.append('<end>')        
    vocab = list(set(vocab))
    
    return vocab

def get_all_tweets(lang: str,number_of_starts: int = 1) -> list[str]:
    data_files = [file for file in os.listdir('data') if file.endswith('.csv') and file.startswith(lang)]
    all_tweets = []
    for file in data_files:
        with open(os.path.join('data', file), 'r') as f:
            data = pd.read_csv(f)
            for tweet in data['tweet_text']:
                text = ['<start>']*number_of_starts + list(tweet) + ['<end>']
                all_tweets.append(text)
    return all_tweets
                
def get_ngrams(text: str, n: int) -> list[str]:
    ngrams = []
    for i in range(len(text) - n + 1):
        ngrams.append(text[i:i+n])
    return ngrams

def count_next_tokens(LM: dict[str, dict[str, float]], n: int,all_tweets: list[str]) -> dict[str, dict[str, float]]:
	n_ngrams = (get_ngrams(tweet, n) for tweet in all_tweets)
	for sentence in n_ngrams:
		for word in sentence:
			LM["".join(word[:-1])][word[-1]] += 1
	return LM

def normalize_LM(LM: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
	for key in LM:
		total = sum(LM[key].values())
		for next_token in LM[key]:
			if total != 0:
				LM[key][next_token] /= total
	return LM


 
def build_lm(lang: str, n: int, smoothed: bool = False) -> dict[str, dict[str, float]]:
    '''
    Return a language model for the given lang and n_gram (n)
    :param lang: the language of the model
    :param n: the n_gram value
    :param smoothed: boolean indicating whether to apply smoothing
    :return: a dictionary where the keys are n_grams and the values are dictionaries
    '''

    all_tweets = get_all_tweets(lang,n-1)
    keys_t = (get_ngrams(tweet, n-1) for tweet in all_tweets)
    sentences = (ngram for ngram in keys_t)
    keys = []
    for sentence in sentences:
        keys.extend(sentence)
    
    
    vocab_prob = {char: int(smoothed) for char in vocab}
    if smoothed:
        vocab_prob['<unk>'] = 1 / len(vocab)
    LM = {"".join(key): vocab_prob.copy() for key in keys}

    LM = count_next_tokens(LM, n, all_tweets)
    LM = normalize_LM(LM)
    return LM
	
def test_build_lm():
    return {
        'english_2_gram_length': len(build_lm('en', 2, True)),
        'english_3_gram_length': len(build_lm('en', 3, True)),
        'french_3_gram_length': len(build_lm('fr', 3, True)),
        'spanish_3_gram_length': len(build_lm('es', 3, True)),
    }
    
def g_test_build_lm(results):
    if results["english_3_gram_length"] != 8239:
        return f"English 3-gram length is {results['english_3_gram_length']}, expected 8239"
    if results["french_3_gram_length"] != 8286:
        return f"French 3-gram length is {results['french_3_gram_length']}, expected 8286"
    if results["spanish_3_gram_length"] != 8469:
        return f"Spanish 3-gram length is {results['spanish_3_gram_length']}, expected 8469"
    return 1

vocab = preprocess()
results = test_build_lm()
print(results)
print(g_test_build_lm(results))