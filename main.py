import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

def json_to_dic(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    #le dictionnaire avec les titres et resumes
    movie_dict = {movie['Title']: movie['Plot'] for movie in data}
    return movie_dict

#print(json_to_dic('film.json'))

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

def nlp(sentence):
    #tokenisation
    tokens = word_tokenize(sentence.lower())
    
    #lemmatisation
    lemmatizer = WordNetLemmatizer()
    from nltk import pos_tag
    from nltk.corpus import wordnet

    def get_wordnet_pos(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    pos_tags = pos_tag(tokens)
    lemmatized_tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(tag)) for token, tag in pos_tags]
    
    #on supp les mots vides
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in lemmatized_tokens if token.isalpha() and token not in stop_words]
    
    return filtered_tokens


sentence = "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency"
print(nlp(sentence))