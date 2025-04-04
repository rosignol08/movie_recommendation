import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from math import log10

def json_to_dic(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    #le dictionnaire avec les titres et resumes
    movie_dict = {movie['Title']: movie['Plot'] for movie in data}
    return movie_dict

mon_dico = json_to_dic('film.json')

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


#sentence = "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency"
#print(nlp(sentence))
def jaccard_similarity(tokens1, tokens2):
    set1 = set(tokens1)
    set2 = set(tokens2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if len(union) > 0 else 0.0

def jaccard_similarity_from_title(title1, title2, movie_dict):
    if title1 not in movie_dict or title2 not in movie_dict:
        print("One or both titles not found in the movie dictionary.")
        return
    
    tokens1 = nlp(movie_dict[title1])
    tokens2 = nlp(movie_dict[title2])
    similarity_score = jaccard_similarity(tokens1, tokens2)
    
    print(f"Le Score de similarité de Jaccard des films `{title1}` et `{title2}` est {similarity_score}")

#jaccard_similarity_from_title("The Godfather", "The Godfather: Part II", mon_dico)
print("quelle doit-être la taille des vecteurs représentant chaque document de notre corpus ?\n")
print("Correspond au nombre total de termes unique sans l'ensemble des résumés des films")
print("\n")
print("Cette valeur peut-elle être obtenue en aditionnant toutes les longueurs des valeurs du dictionnaire obtenu à l'étape 3 du I ?")
print("Non, cette valeur ne peut pas être obtenue en additionnant toutes les longueurs des valeurs du dictionnaire.")
print("Parce que certains termes peuvent apparaître dans plusieurs résumés.")
print("\n")

def tf(token,document):
    #ça fait nb_occurences_token/nb_total_token
    nb_occurences_token = document.count(token)
    nb_total_token = len(document)
    resultat = nb_occurences_token / nb_total_token if nb_total_token > 0 else 0
    return resultat

'''fonction IDF qui prend en argument 
un dictionnaire (clé : titre, valeur : résumé) 
et qui renvoie un dictionnaire dont les clés sont
les tokens du vocabulairede l'ensemble du corpus, 
et les valeurs sont le score idf pour chacun de ces tokens.'''

def IDF(dictionaire):
    #ça fait log10(nb_docs_du_corpus/nb_docs_contenant_token)
    nb_docs_du_corpus = len(dictionaire)
    idf_dict = {}
    for title, plot in dictionaire.items():
        tokens = nlp(plot)
        unique_tokens = set(tokens)
        for token in unique_tokens:
            if token not in idf_dict:
                idf_dict[token] = 0
            idf_dict[token] += 1

    for token, nb_docs_contenant_token in idf_dict.items():
        idf_dict[token] = log10(nb_docs_du_corpus / nb_docs_contenant_token) if nb_docs_contenant_token > 0 else 0

    return idf_dict

#print(IDF(mon_dico))
print("\n")


'''Implémentez une fonction TFIDF qui prend en argument
le dictionnaire des films et qui renvoie un dictionnaire
dont les clés sont le titre des films et les valeurs sont 
le vecteur de coefficients TF-IDF correspondants à ce film.'''
def TFIDF(dictionaire,):
    #ça fait tf(token,document) * idf(token,corpus)
    tfidf_dict = {}
    idf_dict = IDF(dictionaire)
    for title, plot in dictionaire.items():
        tokens = nlp(plot)
        tfidf_vector = {}
        for token in set(tokens):
            tfidf_vector[token] = tf(token, tokens) * idf_dict[token]
        tfidf_dict[title] = tfidf_vector
    return tfidf_dict

#dico_cles = TFIDF(mon_dico)
#print(dico_cles)

def cosine_similarity(list1, list2):
    dot = np.dot(list1, list2)
    norm1 = np.linalg.norm(list1)
    norm2 = np.linalg.norm(list2)
    cos = dot / (norm1 * norm2)
    return(cos)


def recommend_Jaccard(title, movie_dict, n=3):
    if title not in movie_dict:
        print(f"Le film '{title}' n'est pas dans le dictionnaire.")
        return []
    target_tokens = nlp(movie_dict[title])
    similarities = {}
    
    for other_title, plot in movie_dict.items():

def recommend_TFIDF(title, movie_dict, n=3):
    if title not in movie_dict:
        print(f"Le film '{title}' n'est pas dans le dictionnaire.")
        return []
    
    #ça trille les films par similarité décroissante et prendre les n premiers