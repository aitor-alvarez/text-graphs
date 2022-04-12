import nltk
from nltk.tokenize import NLTKWordTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


word_tokenizer = NLTKWordTokenizer()

stop_words = set(stopwords.words('english'))





def normalize_text(sentences):
	normalized_sentences =[]
	for txt in sentences:
		tkns = word_tokenizer.tokenize(txt)
		tkns = [''.join(t.split('-')).lower() for t in tkns if t not in stop_words and t not in '@.,!#$%*:;"' and 'http' not in t and 'www' not in t]
		normalized_sentences.append(' '.join(tkns))
	return normalized_sentences