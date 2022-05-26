import numpy as np # linear algebra
import pandas as pd
import pandas as pd
#! pip install sentence-transformers
#!pip install lexrank
import numpy as np
from sentence_transformers import *
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
import re
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import matplotlib.cm as cm
import torch
import sentence_transformers
#from lexrank_utility import *
#import umap
#import plotly
#plotly.offline.init_notebook_mode (connected = True)

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import glob
import json
import re
import matplotlib.pyplot as plt
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import seaborn as sns
from sklearn.cluster import MiniBatchKMeans


from sklearn.decomposition import PCA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as shc


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from numpy.random import randn, uniform
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler

#!pip install git+https://github.com/boudinfl/pke.git
import numpy as np 
import pandas as pd 
import os
import seaborn as sns
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
import re
import pke

#!pip install git+https://github.com/arvkevi/kneed.git
#!pip install openpyxl
#! pip install scikit-learn-extra
#! pip install pyclustering
#!pip install faiss
#! pip install faiss-cpu --no-cache
import faiss

from tensorflow.keras.models import load_model


import NeighborBlend
from NeighborBlend import *

from datetime import date,timedelta, datetime


# Include standard modules
import argparse

# Initiate the parser
parser = argparse.ArgumentParser(description="Parameters required for the script")

# Add long and short argument
#parser.add_argument("--width", "-w", help="set output width")
parser.add_argument("--thresh_similarity_sigmoid","-tss",default=0.5,type=float,help="Threshold for sigmoid function in order to classify as event and non-event")

parser.add_argument("--thresh_similarity_comp","-tsc",default=0.7,type=float,help="Threshold for comapring it to the previous classes")

parser.add_argument("--neighbor_blend_thresh1","-nbt1",default=0.6,type=float,help="Threshold 1 for Neighbor hood blending")

parser.add_argument("--neighbor_blend_thresh2","-nbt2",default=0.0,type=float,help="Threshold 2 for Neighbor hood blending")

parser.add_argument("--k_nearest_neigh","-knn",default=15,type=int,help="K nearest neighbors to do neighborhood blending")

parser.add_argument("--community_detection","-cd",default='louvain',help="Community detection algorithm")

parser.add_argument("--community_blending_thresh1","-cbt1",default=0.4,type=float,help="Threshold for community blending 1")


parser.add_argument("--community_blending_thresh2","-cbt2",default=0.5,type=float,help="Threshold for community blending 1")


parser.add_argument("--magic_number","-mn",default=40,type=int,help="Minimum members to be present in a community to be considered legit")

parser.add_argument("--keyword_extraction","-key",default="SingleRank",help="Select the keyword extraction technique")




# Read arguments from the command line
args = parser.parse_args()

# Check for --width
print(args.thresh_similarity_sigmoid)
print(args.neighbor_blend_thresh1)
print(args.neighbor_blend_thresh2)
print(args.community_detection)
print(args.community_blending_thresh1)
print(args.community_blending_thresh2)
print(args.keyword_extraction)


today = date.today()


# Month abbreviation, day and year	
date_today = today.strftime("%b%d")
print("date =", date_today)

yesterday = today - timedelta(days = 1)
print("Yesterday was: ", yesterday)

date_yesterday=datetime.strftime(yesterday, '%b%d')
print(date_yesterday)
#-----------------------------------

import os
  
# Directory
directory = date_today
  
# Parent Directory path
parent_dir = "../pipeline_scripts/"
  
# Path
today_path = os.path.join(parent_dir, directory)
path_yesterday=parent_dir+date_yesterday

os.mkdir(today_path)
print("Directory '% s' created" % directory)
  


#print("Directory '% s' created" % directory)
  
#----------------------------------------------
path_to_live_tweets='../pipeline_scripts/LiveTweets/'

tweets_today=pd.read_csv(path_to_live_tweets+date_today+'.csv',nrows=400)

tweets_today.columns=['Date', 'Text', 'Retweet count', 'Followers count', 'Location',
       'Username', 'Statuses count']
##--------------------------------------------

def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    text=emoji_pattern.sub(r'', text) # no emoji
    return text

# Applying the cleaning function to both test and training datasets
#data_event['clean_tweet'] = data_event['tweet'].apply(lambda x: clean_text(x))
#data_non_event['clean_tweet']=data_non_event['tweet'].apply(lambda x: clean_text(x))
tweets_today['clean_tweet']=tweets_today['Text'].apply(lambda x: clean_text(x))

#_-----------------------------------


model = SentenceTransformer('paraphrase-distilroberta-base-v1')
        
#query_words_encode=model.encode(query_string)
print('dcx')

X_liv_test=model.encode(tweets_today['clean_tweet'])



loaded_model=load_model(parent_dir+'model_ann_clean_unique_194k_100.h5')

pred=loaded_model.predict(X_liv_test)
#_______-----------------------------------------------

pred=loaded_model.predict(X_liv_test)

yy_test=pred

output_class_pred=[]
#output_class_pred=[]
for i in range(len(yy_test)):
    #m=max(y_test[i])
    if(yy_test[i][0]<args.thresh_similarity_sigmoid):
        output_class_pred.append(0)
    else:
        output_class_pred.append(1)
tweets_today['tag_from_model']=output_class_pred
tweets_today_eventful=tweets_today[tweets_today['tag_from_model']==1]

tweets_today_eventful.reset_index(drop=True,inplace=True)
#------------------------------------------------------------------------------




yesterday_merged_tweets=pd.read_csv(path_yesterday+'/MergedClusters_'+date_yesterday+'.csv',nrows=100)
classes=yesterday_merged_tweets['Class'].unique()

import spacy 
nlp = spacy.load('en_core_web_lg')

tweets_today_eventful['class']=0



for sno,query in enumerate(tweets_today_eventful['clean_tweet']):
    #print(sno,query)
    similarity_score=0
    for index,class_emb in enumerate(classes):
        #print(index,class_emb)
        sim=nlp(query).similarity(nlp(class_emb))
        print(sim)
        if(sim>similarity_score and sim>args.thresh_similarity_comp):
            similarity_score=sim
            tweets_today_eventful['class'][sno]=classes[index]
        else:
            pass
        
        
tweets_common_from_yesterday=tweets_today_eventful[tweets_today_eventful['class']!=0]

Oldclusters_DayDataset= pd.concat([yesterday_merged_tweets,tweets_common_from_yesterday ], axis=0)
Oldclusters_DayDataset.reset_index(drop=True,inplace=True)


filename='Oldclusters_DayDataset_'+date_today+".csv"
Oldclusters_DayDataset.to_csv(today_path+'/'+filename,header=Oldclusters_DayDataset.columns,columns=Oldclusters_DayDataset.columns)
    
#######_---------------------------------

tweets_unique_from_yesterday=tweets_today_eventful[tweets_today_eventful['class']==0]
live_tweet_eventful=tweets_unique_from_yesterday.copy()
live_tweet_eventful.reset_index(drop=True,inplace=True)


model = SentenceTransformer('paraphrase-distilroberta-base-v1')
        
#query_words_encode=model.encode(query_string)
print('dcx')

X_reduced=model.encode(live_tweet_eventful['clean_tweet'])
#------------------------------------------------------------ Neighborhood blending
"""def neighborhood_search(emb,thresh):
    
    index = faiss.IndexFlatIP(emb.shape[1])
    faiss.normalize_L2(emb)
    index.add(emb)
    #faiss.write_index(index,'yodd')
    
    sim, I = index.search(emb, 100)
    #print(emb)
    #print(I[32])
    print(sim[2])
    
    
    pred_index=[]
    pred_sim=[]
    for i in range(emb.shape[0]):
        cut_index=0
        for j in sim[i]:
            if(j>thresh):
                cut_index+=1
            else:
                break
                
        #cut_index =  np.searchsorted(sim[i],thresh)
        pred_index .append( I[i][:(cut_index)])
        pred_sim .append (sim[i][:(cut_index)])
    #print(pred_index[32])
    #print(pred_sim[32])
        
    return pred_index,pred_sim
    
    
    
    

def blend_neighborhood(emb, match_index_lst, similarities_lst):
    new_emb = emb.copy()
    for i in range(emb.shape[0]):
        cur_emb = emb[match_index_lst[i]]
        #print(cur_emb)
        weights = np.expand_dims(similarities_lst[i], 1)
        #print(weights)
        new_emb[i] = (cur_emb * weights).sum(axis=0)
        #print(new_emb[i])
    new_emb = normalize(new_emb, axis=1)
    #print(weights)
    #print(new_emb[199])
    #print(new_emb)
    return new_emb



#th1=0.6   #########*********
#th2=0.8########*#***********


#ust be a list of thresholds kyunki iteratively similarities badhti jaani  chahiye
def iterative_neighborhood_blending(emb, threshes):
    for thresh in threshes:
        match_index_lst, similarities_lst = neighborhood_search(emb, thresh)
        emb = blend_neighborhood(emb, match_index_lst, similarities_lst)
    return match_index_lst,similarities_lst
"""
#threshes=[args.neighbor_blend_thresh1,args.neighbor_blend_thresh2]#must be a list of thresholds kyunki iteratively similarities badhti jaani  chahiye
threshes=[0]
X_reduced,match_index_lst,similarities_lst=iterative_neighborhood_blending(X_reduced, threshes,args.k_nearest_neigh)

#-------------------------------------------------



import networkx as nx
def create_graph(match_index_lst):
    adjacency_dict = {i: match_index_lst[i] for i in range(0, len( match_index_lst))}
    #print(adjacency_dict)
    graph = nx.from_dict_of_lists(adjacency_dict)
    #graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph

graph=create_graph(match_index_lst)


for i in range(X_reduced.shape[0]):
    #print('hi')
    for j in range(len(match_index_lst[i])):
        graph[i][match_index_lst[i][j]]['weight']=similarities_lst[i][j]


#---------------------------------------------------------------



#import pycombo
import community as community_louvain
def get_communities_in_graph(graph,method):
    memberships = {}
    if method=='pycombo':
        memberships=pycombo.execute(graph,weight='weight')
    if method == "louvain":
        memberships = community_louvain.best_partition(graph)
        #print(memberships)
    elif method=='leidenalg':
        
        r=graph.edges.data('weight')
        G2 = ig.Graph.TupleList(r, directed=False,weights=True)
        weights = np.array(G2.es["weight"]).astype(np.float64)
        results=leidenalg.find_partition(G2, leidenalg.ModularityVertexPartition,n_iterations=-1,weights=weights)
        print(len(list(results)))
        print('hi')
        memberships={}
        communities = list(results)
        for i, single_community in enumerate(list(communities)):
            for member in single_community:
                memberships[int(member)] = i
        '''mm={}
        
        for i in range(len(list(memberships))):
            for j in memberships[i]:
                mm[j]=i
        
        return mm'''
        
    elif method == "connected_components":
        set_of_communities = list(nx.connected_components(graph))
        for label_iter, single_community in enumerate(set_of_communities):
            members = list(single_community)
            for member in members:
                memberships[int(member)] = label_iter
    elif method == "girvan_newman":
        result = nx.algorithms.community.girvan_newman(graph)
        communities = next(result)
        for i, single_community in enumerate(list(communities)):
            for member in single_community:
                memberships[int(member)] = i
    memberships = {k: v for k, v in sorted(list(memberships.items()), key=lambda x: x[0])}
    return memberships
memberships = get_communities_in_graph(graph, method=args.community_detection)
live_tweet_eventful['predicted_community']=list(memberships.values())


print("Total Communities ---->",live_tweet_eventful['predicted_community'].max())


#---------------------------------------------------------------

inn=live_tweet_eventful.index
live_tweet_eventful['Unnamed: 0']=inn  



from collections import OrderedDict
def refine_clustering(X):
    for i in range(0, len(X)):
        X[i] /= np.linalg.norm(X[i])
    print(len(X))
    phrases = live_tweet_eventful['Text'].astype(str).tolist()
    phrase_vs_embeddings = {phrases[i]: X[i] for i in range(len(phrases))}
    
    cluster_label = 0
    label_vs_cluster = {}
    for label, item in live_tweet_eventful.groupby('predicted_community'):
    #print(item)
        label_vs_cluster[cluster_label] = (item['Text'].astype(str).tolist(), item['Unnamed: 0'].tolist())
        cluster_label += 1
        
    current_iter = 0
    while True:
        
        number_of_cluster_label = len(label_vs_cluster.keys())
        print("Number of cluster label before merging: {}".format(number_of_cluster_label))
        label_vs_cluster = {k: v for k, v in sorted(label_vs_cluster.items())}
        label_vs_cluster_center = OrderedDict()
        for label in label_vs_cluster:
            cluster_center = [phrase_vs_embeddings[phrase] for phrase in label_vs_cluster[label][0]]
            cluster_center = np.array(cluster_center, dtype='float32')
            cluster_center = np.mean(cluster_center, axis=0, dtype='float32')
            label_vs_cluster_center[label] = cluster_center
        print('label_vs_cluster_center',len(label_vs_cluster_center))
        
        
        embeddings_matrix = np.array(list(label_vs_cluster_center.values()), dtype='float32')
        embeddings_matrix = np.array([vec/np.linalg.norm(vec) for vec in embeddings_matrix], dtype='float32')
        #print("Embeddings matrix: {}".format(str(embeddings_matrix.shape))
              
        #thr1=0.4
        #thr2=0.5
        threshes=[args.community_blending_thresh1,args.community_blending_thresh2]
        embeddings_matrix,match_index_lst_centers,similarities_lst_centers=iterative_neighborhood_blending(embeddings_matrix, threshes,100)
        
        graph_centers=create_graph(match_index_lst_centers)
        
        for i in range(embeddings_matrix.shape[0]):
            for j in range(len(match_index_lst_centers[i])):
                graph_centers[i][match_index_lst_centers[i][j]]['weight']=similarities_lst_centers[i][j]
              
        memberships_centers = get_communities_in_graph(graph_centers, method='louvain')
        print("Length of memberships: ",len(memberships_centers))
              
        new_label_vs_cluster = {}
        for member, label in memberships_centers.items():
            if label not in new_label_vs_cluster:
                new_label_vs_cluster[label] = ([], [])
    
            new_label_vs_cluster[label][0].extend(label_vs_cluster[member][0].copy())
            new_label_vs_cluster[label][1].extend(label_vs_cluster[member][1].copy())
        current_iter += 1
        print("[iteration: {}]: label difference: {}".format(current_iter, number_of_cluster_label - len(new_label_vs_cluster)))
        print("Number of cluster label after merging: {}".format(len(new_label_vs_cluster)))
        #print("Difference",number_of_cluster_label - len(new_label_vs_cluster))
        #print('After merging',len(new_label_vs_cluster))
        label_vs_cluster = new_label_vs_cluster.copy()
        
        if number_of_cluster_label - len(new_label_vs_cluster) < 50 or current_iter == 5:
            break
        
    return label_vs_cluster
            
final_label_vs_cluster=refine_clustering(X_reduced)



#---------------------------------------------------------------



live_tweet_eventful['final_pred_comunity']=['']*len(live_tweet_eventful)
for i in range(len(final_label_vs_cluster)):
    for j in final_label_vs_cluster[i][1]:
        live_tweet_eventful['final_pred_comunity'][j]=i
coun=live_tweet_eventful['final_pred_comunity'].value_counts()
live_tweet_eventful['prdicted_count']=[coun[j] for j in list(live_tweet_eventful['final_pred_comunity'])]      

#_----------------------



live_tweet_eventful_majority=(live_tweet_eventful[live_tweet_eventful['prdicted_count']>args.magic_number]).sort_values(by = 'prdicted_count')


no_of_classes_today=live_tweet_eventful_majority['final_pred_comunity'].nunique()
classes_today=live_tweet_eventful_majority['final_pred_comunity'].unique()
#------------------------------------------


###---------------------------------------------------------------
extraction_tech=args.keyword_extraction
def get_keywords(no_of_clusters):
    tf=[]
    yake=[]
    textrank=[]
    singlerank=[]
    topicrank=[]
    topicalpagerank=[]
    positionrank=[]
    multipartite=[]
    kea=[]
    wingnus=[]
    for i in range(no_of_clusters):
        
        df=live_tweet_eventful_majority[live_tweet_eventful_majority['final_pred_comunity']==classes_today[i]]['Text']
        #df=data_train['text'][cluster.labels_==i]
        df=df.to_frame()
        df=df.reset_index()
        textt=""
        
        for j in range(len(df)):
            textt+=df['Text'][j]
        if extraction_tech=="TfIdf":
            extractor = pke.unsupervised.TfIdf() 
            extractor.load_document(textt)
            extractor.candidate_selection()
            extractor.candidate_weighting()
            keyphrases = extractor.get_n_best(n=5, stemming=False)
            print("Cluster", i)
            #print(keyphrases,"\n")
            tf.append(keyphrases)
        
        if extraction_tech=="Yake":
            extractor = pke.unsupervised.YAKE() 
            extractor.load_document(textt)
            extractor.candidate_selection()
            extractor.candidate_weighting()
            keyphrases = extractor.get_n_best(n=5, stemming=False)
            print("Cluster", i)
            #print(keyphrases,"\n")
            yake.append(keyphrases)


        if extraction_tech=="TextRank":
            extractor = pke.unsupervised.TextRank () 
            extractor.load_document(textt)
            extractor.candidate_selection()
            extractor.candidate_weighting()
            keyphrases = extractor.get_n_best(n=5, stemming=False)
            print("Cluster", i)
            #print(keyphrases,"\n")
            textrank.append(keyphrases)

        if extraction_tech=="SingleRank":
            extractor = pke.unsupervised.SingleRank () 
            extractor.load_document(textt)
            extractor.candidate_selection()
            extractor.candidate_weighting()
            keyphrases = extractor.get_n_best(n=5, stemming=False)
            print("Cluster", i)
            #print(keyphrases,"\n")
            singlerank.append(keyphrases)
        
        if extraction_tech=="TopicRank":
            extractor = pke.unsupervised.TopicRank () 
            extractor.load_document(textt)
            extractor.candidate_selection()
            extractor.candidate_weighting()
            keyphrases = extractor.get_n_best(n=2, stemming=False)
            print("Cluster", i)
            #print(keyphrases,"\n")
            topicrank.append(keyphrases)


        if extraction_tech=="TopicalPageRank":
            extractor = pke.unsupervised.TopicalPageRank () 
            extractor.load_document(textt)
            extractor.candidate_selection()
            extractor.candidate_weighting()
            keyphrases = extractor.get_n_best(n=2, stemming=False)
            print("Cluster", i)
            #print(keyphrases,"\n")
            topicalpagerank.append(keyphrases)
        
        
        
        if extraction_tech=="MultipartiteRank":
            extractor = pke.unsupervised.MultipartiteRank () 
            extractor.load_document(textt)
            extractor.candidate_selection()
            extractor.candidate_weighting()
            keyphrases = extractor.get_n_best(n=2, stemming=False)
            print("Cluster", i)
            #print(keyphrases,"\n")
            multipartite.append(keyphrases)
        
        
        if extraction_tech=="Kea":
            extractor = pke.supervised.Kea () 
            extractor.load_document(textt)
            extractor.candidate_selection()
            extractor.candidate_weighting()
            keyphrases = extractor.get_n_best(n=5, stemming=False)
            print("Cluster", i)
            #print(keyphrases,"\n")
            kea.append(keyphrases)
        
        if extraction_tech=="Wingnus":
            extractor = pke.supervised.WINGNUS () 
            extractor.load_document(textt)
            extractor.candidate_selection()
            extractor.candidate_weighting()
            keyphrases = extractor.get_n_best(n=5, stemming=False)
            print("Cluster", i)
            #print(keyphrases,"\n")
            wingnus.append(keyphrases)
        
        
        
        
        
        
        
    #data = {'TFIDF':tf,'Yake':yake,"TextRank":textrank,"SingleRank":singlerank,"kea":kea,"WINGNUS":wingnus}
    data = {"extraction_tech":extraction_tech}
    
    key=pd.DataFrame(data)
    return key
        

n_optimum_clusters=no_of_classes_today
key=get_keywords(n_optimum_clusters)

###---------------------------------------------------------------

live_tweet_eventful_majority.reset_index(inplace=True,drop=True)

live_tweet_eventful_majority['Class']=0
for i in range(no_of_classes_today):
    
    a=live_tweet_eventful_majority[live_tweet_eventful_majority['final_pred_comunity']==classes_today[i]].index
    live_tweet_eventful_majority.loc[a,['Class']]=key['extraction_tech'][i][0][0]


###---------------------------------------------------------------
live_tweet_eventful_majority.drop(['predicted_community','Unnamed: 0','final_pred_comunity','prdicted_count'],axis=1,inplace=True)


new_name='Newclusters_DayDataset_'+date_today+'.csv'
live_tweet_eventful_majority.to_csv(today_path+'/'+new_name,columns=live_tweet_eventful_majority.columns,header=live_tweet_eventful_majority.columns)



file1=pd.read_csv(today_path+'Oldclusters_DayDataset_'+date_today+'.csv')
file2=pd.read_csv(today_path+'Newclusters_DayDataset_'+date_today+'.csv')

merged_cluster=pd.concat([file1,file2],axis=0)
merged_cluster.reset_index(drop=True,inplace=True)

mergedname='MergedClusters_'+date_today+'.csv'
merged_cluster.to_csv(today_path+'/'+mergedname,header=merged_cluster.columns,columns=merged_cluster.columns)

    