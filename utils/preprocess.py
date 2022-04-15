from nltk.tokenize import NLTKWordTokenizer
from nltk.corpus import stopwords
import os
import pandas as pd
import json
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
import torch


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
					rep_dict ={'resp_id':int(reply[1][0]), 'resp_txt':reply[1][2], 'label':1 if data['is_rumor']=='TRUE' else 0}
					replies_out.append(rep_dict)
				tree.append({'root_id':int(root_id ), 'root_txt':root_txt, 'root_label':1 if data['is_rumor']=='TRUE' else 0, 'responses': replies_out})
			trees.append(tree)
	output = json.dumps(trees)
	with open('pheme_tree.json', 'w') as outfile:
		outfile.write(output)


#Create word embeddings for the tweets using shallow models
def word_embeddings(tweets):
	sentences = normalize_text(tweets)
	model = Word2Vec(sentences=sentences, vector_size=200, window=5, min_count=5)
	embeddings = model.wv
	embeddings.save("pretrained/word2vec.wordvectors")
	return embeddings


#Bertweet embedding
def bert_tweet(tweets):
	tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)
	tweet_model = AutoModel.from_pretrained("vinai/bertweet-base")
	tokens = {'input_ids': [], 'attention_mask': []}

	for sen in tweets:
		tkn = tokenizer.encode_plus(sen, max_length=130,
		                            truncation=True, padding='max_length',
		                            return_tensors='pt')
		tokens['input_ids'].append(tkn['input_ids'][0])
		tokens['attention_mask'].append(tkn['attention_mask'][0])

	tokens['input_ids'] = torch.stack(tokens['input_ids'])
	tokens['attention_mask'] = torch.stack(tokens['attention_mask'])

	with torch.no_grad():
		output = tweet_model(**tokens)

	tweet_embeddings = mean_pooling(output, tokens['attention_mask'])

	return tweet_embeddings


def transformer_sentences(tweets, fine_tune=False):
	model = SentenceTransformer('bert-base-nli-mean-tokens')
	embeddings = model.encode(tweets)
	return embeddings



def normalize_text(sentences):
	word_tokenizer = NLTKWordTokenizer()
	stop_words = set(stopwords.words('english'))
	normalized_sentences =[]
	for txt in sentences:
		tkns = word_tokenizer.tokenize(txt)
		tkns = [''.join(t.split('-')).lower() for t in tkns if t not in stop_words and t not in '@.,!#$%*:;"' and 'http' not in t and 'www' not in t]
		normalized_sentences.append(' '.join(tkns))
	return normalized_sentences


#Mean Pooling using the attention mask of the tokens
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


#Fine tuning for Transformer-based models for text classification
def transformer_fine_tuning(model_name, train_data, test_data, tokenizer, nlabels):
	model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=nlabels)
	data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
	training_args = TrainingArguments(output_dir="data/trained/model_"+model_name, learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,)
	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=train_data,
		eval_dataset=test_data,
		tokenizer=tokenizer,
		data_collator=data_collator,
	)
	trainer.train()

	return model
