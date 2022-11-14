from django.shortcuts import render
import requests
import pickle
from . import preprocess

import pandas as pd
from transformers import BertTokenizer, BertModel
import numpy as np
import nltk
import torch
from sklearn.metrics.pairwise import cosine_similarity
from nltk import sent_tokenize
from fuzzywuzzy import process
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import fuzz




Processes=pd.read_csv(r'C:\Users\MSI\Desktop\nlp\PROCESSES.csv')
del Processes['Unnamed: 0']
Processes["Line"] = Processes["process"] + Processes["sub process"]
Processes_nouns=Processes['process'].tolist()
Subprocess=Processes['sub process'].tolist()
Processes_Nouns=[]
for process1 in Processes_nouns:
  if process1 not in Processes_Nouns:
    Processes_Nouns.append(process1)
details=Processes['details'].tolist()
nltk.download('punkt')
nltk.download('wordnet')
####################################
def Process_inputs(process1):
  inputs=''
  i=0
  for index,row in Processes.iterrows():
    sub_process=row['sub process']
    process_noun=row['process']
    Type=row['type']
    if (str(process_noun)==process1) & (str(Type)==" Inputs"):
          inputs=inputs+','+sub_process
  return inputs
##################################
def SubProcess_is_Input_of(subprocess):
  inputs=''
  i=0
  for index,row in Processes.iterrows():
    sub_process=row['sub process']
    process_noun=row['process']
    Type=row['type']
    if (str(sub_process)==subprocess) & (str(Type)==" Inputs"):
          inputs=inputs+','+process_noun
  return inputs
  ########################################
def Process_outputs(process1):  #retourne the output of process
  outputs=''
  i=0
  for index,row in Processes.iterrows():
    sub_process=row['sub process']
    process_noun=row['process']
    Type=row['type']
    if (str(process_noun)==process1) & (str(Type)==" outputs"):
          outputs=outputs+','+sub_process
          # i=i+1
    # print(i)
  return outputs
########################################
def SubProcess_is_Output_of(subprocess):  #retourne nom_process/ subprocess is the ouput of nom_process
  outputs=''
  i=0
  for index,row in Processes.iterrows():
    sub_process=row['sub process']
    process_noun=row['process']
    Type=row['type']
    if (str(sub_process)==subprocess) & (str(Type)==" outputs"):
          outputs=outputs+','+process_noun
  return outputs
########################################
def Process_tools(process1):   # return the tools of process
  tools=''
  i=0
  for index,row in Processes.iterrows():
    sub_process=row['sub process']
    process_noun=row['process']
    Type=row['type']
    if (str(process_noun)==process1) & (str(Type)==" tools and techniques"):
          tools=tools+','+sub_process
          # i=i+1
    # print(i)
  return tools
########################################
def SubProcess_is_tool_of(subprocess):   #retourne nom_process/ subprocess is the tool of nom_process
  tools=''
  i=0
  for index,row in Processes.iterrows():
    sub_process=row['sub process']
    process_noun=row['process']
    Type=row['type']
    if (str(sub_process)==subprocess) & (str(Type)==" tools and techniques"):
          tools=tools+','+process_noun
          # i=i+1
    # print(i)
  return tools
########################################
def Process_details(process1):  #en entrée : esm subprocess
  print('hello')
  for index,row in Processes.iterrows():
    sub_process=row['sub process']
    if sub_process==process1:
      return row['details']
########################################
def process_bert_similarity(base_document,documents):

	# This will download and load the pretrained model offered by UKPLab.
	model = SentenceTransformer('bert-base-nli-mean-tokens')

	# Although it is not explicitly stated in the official document of sentence transformer, the original BERT is meant for a shorter sentence. We will feed the model by sentences instead of the whole documents.
	sentences = sent_tokenize(base_document)
	base_embeddings_sentences = model.encode(sentences)
	base_embeddings = np.mean(np.array(base_embeddings_sentences), axis=0)

	vectors = []
	for i, document in enumerate(documents):

		sentences = sent_tokenize(document)
		embeddings_sentences = model.encode(sentences)
		embeddings = np.mean(np.array(embeddings_sentences), axis=0)

		vectors.append(embeddings)

		#print("making vector at index:", i)

	scores = cosine_similarity([base_embeddings], vectors).flatten()

	highest_score = 0
	highest_score_index = 0
	for i, score in enumerate(scores):
		if highest_score < score:
			highest_score = score
			highest_score_index = i

	most_similar_document = documents[highest_score_index]
	return(most_similar_document)
########################################
def response_subprocess(requete):
  subprocess=process_bert_similarity(requete,Subprocess)
  S2='The subprocess is : '+subprocess+','+subprocess+' is the input of :'+SubProcess_is_Input_of(subprocess)+','+subprocess+' is the output of :'+SubProcess_is_Output_of(subprocess)+','+subprocess+' is the tool of :'+SubProcess_is_tool_of(subprocess)
  L2=S2.split(',')
  return L2
  #return('The subprocess is : '+subprocess,subprocess+' is the input of : ',SubProcess_is_Input_of(subprocess),subprocess+' is the output of : ',SubProcess_is_Output_of(subprocess),subprocess+' is the tool of : ',SubProcess_is_tool_of(subprocess))
########################################
def highest_score(requete,List_processes):
  highest = process.extractOne(requete,List_processes)
  return highest[1] #return the highest score
########################################
def similar_process(requete,List_processes):
  highest = process.extractOne(requete,List_processes)
  return highest[0]  # return the most similar process(compared to requete)
########################################
def response_requete(requete):
  if requete in Processes_Nouns:
   index=Processes_Nouns.index(requete)
   Noun_process=Processes_Nouns[index]
   S='The process is : '+Noun_process+','+Noun_process+' inputs :'+Process_inputs(Noun_process)+','+Noun_process+' outputs :'+Process_outputs(Noun_process)+','+Noun_process+' tools and techniques :'+Process_tools(Noun_process)
   L=S.split(',')
   return L
   #return ('The process is :'+Noun_process,Noun_process+' inputs :',Process_inputs(Noun_process),Noun_process+' outputs :',Process_outputs(Noun_process),Noun_process+' tools and techniques :',Process_tools(Noun_process))
  elif (highest_score(requete,Processes_Nouns)> 80):
    matched_process=similar_process(requete,Processes_Nouns)
    S1='The process is :'+matched_process+','+matched_process+' inputs :'+Process_inputs(matched_process)+','+matched_process+' outputs :'+Process_outputs(matched_process)+','+matched_process+' tools and techniques :'+Process_tools(matched_process)
    L1=S1.split(',')
    return L1
    #return ('The process is :'+matched_process,matched_process+' inputs :',Process_inputs(matched_process),matched_process+' outputs :',Process_outputs(matched_process),matched_process+' tools and techniques :',Process_tools(matched_process))
  else :
    return (response_subprocess(requete))
########################################



def predict(request):

    if request.method=='POST':
        message=request.POST.get('message',None)
        print(message)
        result={'res':response_requete(message)}
        ## trying to split from source
        return render(request,'ContactFrom/results.html',result)

    else:
        return render(request,'ContactFrom/index.html')

def getDetails(request):
    title=request.GET['title']
    result={'res':Process_details(title) , 'title':title }
    return render(request,'ContactFrom/details.html' , result)

