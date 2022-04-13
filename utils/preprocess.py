from nltk.tokenize import NLTKWordTokenizer
from nltk.corpus import stopwords
import os
import pandas as pd
import json

word_tokenizer = NLTKWordTokenizer()

stop_words = set(stopwords.words('english'))


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
	with open('json_data.json', 'w') as outfile:
		outfile.write(output)


def normalize_text(sentences):
	normalized_sentences =[]
	for txt in sentences:
		tkns = word_tokenizer.tokenize(txt)
		tkns = [''.join(t.split('-')).lower() for t in tkns if t not in stop_words and t not in '@.,!#$%*:;"' and 'http' not in t and 'www' not in t]
		normalized_sentences.append(' '.join(tkns))
	return normalized_sentences