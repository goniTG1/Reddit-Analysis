
import spacy
import gensim
from gensim.models.phrases import Phrases
from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary
from gensim.models.wrappers.ldamallet import malletmodel2ldamodel
import pyLDAvis.gensim as gensimvis
import pyLDAvis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
from gensim.models.wrappers import LdaMallet
os.environ['MALLET_HOME'] = '/user/Downloads/mallet-2.0.8'
mallet_path = '/user/Downloads/mallet-2.0.8/bin/mallet'

import os
os.environ.update({'MALLET_HOME': r'/user/Downloads/mallet-2.0.8'})
mallet_path = '/user/Downloads/mallet-2.0.8/bin/mallet'


"""
    NATURAL LANGUAGE  PROCESSING
"""

def get_plot_lda(df, subreddit):
    filter_df = df
    nlp = spacy.load('en_core_web_md')
    nlp.max_length = 10000000
    nlp = spacy.load('en_core_web_md')
    nlp.disable_pipe('parser')
    nlp.disable_pipe('ner')
    valid_POS = set(['VERB', 'NOUN', 'ADJ', 'PROPN'])
    filter_df["New_Title"] = filter_df['Eng Title'].apply(lambda x: " ".join([y.lemma_ for y in nlp(x)]))
    filter_df["New_Title"] = filter_df['New_Title'].apply(lambda x: " ".join([token.lemma_ for token in nlp(x) if token.is_alpha and token.pos_ in valid_POS
                              and token.is_stop==False]))
    filter_df['New_Title'] = filter_df.New_Title.astype(str)
    filter_df = filter_df.reset_index(drop=True)
    Final_Title_l = []
    for i in range(len(filter_df['New_Title'])):
      pepe = filter_df['New_Title'][i].lower()
      Final_Title_l.append(pepe)
    filter_df['New_Title'] = Final_Title_l
    # Gensim Part
    # we will start by N-gram detection
    mycorpus = [gensim.utils.simple_preprocess(line, deacc =True) for line in filter_df.New_Title]
    phrase_model = Phrases(mycorpus, min_count=1, threshold=20)
    mycorpus = [el for el in phrase_model[mycorpus]]  # We populate mycorpus again
    Ngrams = {}
    for doc in mycorpus:
        for token in doc:
            if token in Ngrams:
                Ngrams[token] += 1
            else:
                Ngrams[token] = 1

    Ngrams_df = pd.DataFrame(Ngrams.items(), columns = ['Ngrams', 'count'])
    Ngrams_df.set_index('Ngrams')
    Ngrams_df = Ngrams_df.sort_values(by=['count'], ascending=False).reset_index(drop=True)

    topic = []
    for j in range(len(filter_df['New_Title'])):
        counts = []
        if len(filter_df['New_Title'][j].split(' ')) > 0:
            for i in filter_df['New_Title'][j].split(' '):
                if i in Ngrams_df['Ngrams'].to_list():
                    counts.append(int(Ngrams_df[Ngrams_df['Ngrams'] == i]['count']))
            if len(counts) > 0:
                m = counts.index(max(counts))
                topic.append(filter_df['New_Title'][j].split(' ')[m])
            else:
                topic.append('')
        else:
            topic.append('')
    filter_df['Topic'] = topic
    # topic modeling
    # Computing the Dictionary
    mycorpus = filter_df.New_Title
    mycorpus = [el.strip().split() for el in mycorpus]

    #mycorpus = [elem for elem in mycorpus if elem != subreddit.lower()]
    # Create dictionary of tokens
    D = Dictionary(mycorpus)
    no_below = 2  # Minimum number of documents to keep a term in the dictionary
    no_above = .50  # Maximum proportion of documents in which a term can appear to be kept in the dictionary
    D.filter_extremes(no_below=no_below ,no_above=no_above)
    n_tokens = len(D)
    num_topics = 20
    mycorpus_bow = [D.doc2bow(doc) for doc in mycorpus]
    ldag = LdaModel(corpus=mycorpus_bow, id2word=D, num_topics=num_topics)
    #ldag.show_topics(num_topics = -1, num_words = 10, log = False, formatted = True)

    lda_model = gensim.models.ldamodel.LdaModel(corpus=mycorpus_bow,id2word=D,
                                                num_topics=10,random_state=100,
                                                update_every=1,chunksize=10,passes=10,
                                                alpha='symmetric',iterations=100,per_word_topics=True)
    vis = pyLDAvis.gensim.prepare(lda_model, mycorpus_bow, dictionary=lda_model.id2word)

    return vis, ldag
    #pyLDAvis.display(vis_data)


def get_plots(ldag):
    import streamlit as st
    topn = 10
    fig, axes = plt.subplots(2, 5, figsize=(15, 10), sharex=True)
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
    for i in range(5):
        # Build a dataframe with columns 'token' and 'weight' for topic i
        tt = pd.DataFrame(ldag.show_topic(i, topn=topn), columns=['token' ,'weight'])
        sns.barplot(x='weight', y='token', data=tt, color='c', orient='h', ax=axes[0][i])
        axes[0][i].set_title('Topic ' + str(i))
        # Build a dataframe with columns 'token' and 'weight' for topic i + 5
        tt = pd.DataFrame(ldag.show_topic(i+5, topn=topn), columns=['token' ,'weight'])
        sns.barplot(x='weight', y='token', data=tt, color='c', orient='h', ax=axes[1][i])
        axes[1][i].set_title('Topic ' + str(i+5))
    #fig.tight_layout()
    return fig



def get_sentiment_analysis():

    # sentiment analysis: Select good and bad words from the data
    # positive and negative feedback

    def senti(x):
        return TextBlob(x).sentiment

    sentim = []
    for i in  range(len(filter_df)):
        sentim.append(senti(filter_df['Eng Comments'][i]))

    # first number indicates polarity and second subjectivity

    polarity = []
    subjectivity = []

    for i in range(len(sentim)):
        polarity.append(sentim[i][0])
        subjectivity.append(sentim[i][1])

    filter_df['Polarity'] = polarity
    filter_df['Subjectivity'] = subjectivity


    # get most repeated topics

    def most_frequent(List):
        return max(set(List), key = List.count)

    def remove_values_from_list(the_list, val):
        return [value for value in the_list if value != val]

    topic = filter_df['Topic'].to_list()
    n = 5 # number of topics to retrieve
    mf_topics = []

    for i in range((n)):
        mf = most_frequent(topic)
        while len(mf) == 0:
            topic = remove_values_from_list(topic, mf)
            mf = most_frequent(topic)
            mf_topics.append(mf)
            topic = remove_values_from_list(topic, mf)
        else:
            if mf not in mf_topics:
                mf_topics.append(mf)
                topic = remove_values_from_list(topic, mf)

