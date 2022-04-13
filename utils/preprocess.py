from nltk.tokenize import NLTKWordTokenizer
from nltk.corpus import stopwords
import os
import pandas as pd
import json
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModel, AutoTokenizer


def create_tree_pheme(dir):
	trees =[]
	for f in os.listdir(dir):
		tree=[]
		if f.endswith('.csv'):
			data = pd.read_csv(dir+f, lineterminator='\n')
			roots = data[data['is_source_tweet']==True]
			data = data[data['is_source_tweet']==False]
			for root in roots.iterrows():
				root_txt = root[1][2]
				root_id = root[1][0]
				replies = data[data['thread'] == root_id]
				replies_out=[]
				for reply in replies.iterrows():
					rep_dict ={'resp_id':int(reply[1][0]), 'resp_txt':reply[1][2] }
					replies_out.append(rep_dict)
				tree.append({'root_id':int(root_id ), 'root_txt':root_txt, 'responses': replies_out})
			trees.append(tree)
	output = json.dumps(trees)
	with open('pheme_tree.json', 'w') as outfile:
		outfile.write(output)


#Create word embeddings for the tweets
def word_embeddings(tweets):
	sentences = normalize_text(tweets)
	model = Word2Vec(sentences=sentences, vector_size=200, window=5, min_count=5)
	embeddings = model.wv
	embeddings.save("pretrained/word2vec.wordvectors")


#Bertweet embedding
def bert_tweet(tweets, fine_tune=False):
	tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)
	if fine_tune:



def transformer_sentences(tweets, fine_tune=False):
	if fine_tune:



def normalize_text(sentences):
	word_tokenizer = NLTKWordTokenizer()
	stop_words = set(stopwords.words('english'))
	normalized_sentences =[]
	for txt in sentences:
		tkns = word_tokenizer.tokenize(txt)
		tkns = [''.join(t.split('-')).lower() for t in tkns if t not in stop_words and t not in '@.,!#$%*:;"' and 'http' not in t and 'www' not in t]
		normalized_sentences.append(' '.join(tkns))
	return normalized_sentences