
# Importing Dependencies

print("Importing Dependencies...")
# Imports:
from PIL import Image 
import sys 
from pdf2image import convert_from_path 
import os
import PyPDF2
# Imports
import gensim
import numpy as np
np.random.seed(2018)
from gensim.summarization import keywords
from gensim.summarization.summarizer import summarize
import pandas as pd
import spacy
from scipy import spatial
from sentence_transformers import SentenceTransformer
sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
import en_core_web_lg
import docx
import os
import warnings
import requests
from striprtf.striprtf import rtf_to_text
import xml.etree.ElementTree as ET
import time
warnings.filterwarnings("ignore")

nlp = en_core_web_lg.load()


# Reading text from .docx document:
def getText(filename):
  doc = docx.Document(filename)
  fullText = []
  for para in doc.paragraphs:
      fullText.append(para.text.replace('\n',' ').replace('\t', ' '))
  return ' '.join(fullText)


# Extracting Keywords within a given content:
def keywords_from_summary(summary):
  keywords_set = []
  for des in [summary]:
    keywords_set.append(keywords(des).split('\n'))
  return keywords_set[0]


# Cleaning the Extracted Keywords to remove symbols or stopwords:
def clean_keywords(keywords):
  cleaned_keywords = []
  for keyword in keywords:
    if len(keyword) < 20 or len(keyword) > 3:
      cleaned_keywords.append(keyword)
  return cleaned_keywords


# Scanning through the entire Repo to find files:
def find_files():
  print("Scaning Repository for Files...")
  filepaths = []
  for subdir, dirs, files in os.walk(r'.'):
      for filename in files:
          filepath = subdir + os.sep + filename
          if filepath.endswith(".pdf") or filepath.endswith(".rtf") or filepath.endswith(".docx") or filepath.endswith(".xlsx"):
              filepaths.append(filepath)
  return filepaths


# Controller function for extracting contents of files:
def find_content(filepaths):
  print('Reading files within the Repository for content ...')
  documents = []
  for fp in filepaths:
      # Split the extension from the path and normalise it to lowercase.
      ext = os.path.splitext(fp)[-1].lower()
      # Now we can simply use == to check for equality, no need for wildcards.
      if ext == ".pdf":
        document = read_pdf_data(fp)
      elif ext == '.rtf':
        with open(fp, 'r') as file:
          text = file.read()
          document = rtf_to_text(text).replace('\n', ' ').replace('\t', ' ')
      elif ext == '.docx':
        document = getText(fp)
      else:
        meta_path = os.path.dirname(fp) + 'metadata.csv'
        des = pd.read_csv(meta_path)
        try:
          description = des['Description'][0]
        except:
          description = des['Title'][0]
        document = description
      documents.append(document)
  return documents


# Controller Function to find keywords Based upon File-extensions.
def find_keywords(filepaths):
  print("Finding Keywords...")
  file_keywords = []
  files = []
  documents = []
  for fp in filepaths:
      ext = os.path.splitext(fp)[-1].lower()
      if ext == ".pdf":
        # keywords_set = clean_keywords(read_pdf(fp))
        text = ''
        pdfFileObj = open(fp, 'rb') 
        pdfReader = PyPDF2.PdfFileReader(pdfFileObj) 
        num = pdfReader.numPages 
        for i in range(num):
          pageObj = pdfReader.getPage(i) 
          text += pageObj.extractText()
        pdfFileObj.close()
        if text != '':
          document = summarize(text)
        else:
          meta_path = os.path.dirname(fp) + '\metadata.csv'
          des = pd.read_csv(meta_path, encoding = 'unicode_escape')
          try:
            description = des['Description'][0]
          except:
            description = des['Title'][0]
        document = description
        keywords_set = clean_keywords(keywords_from_summary(document))
        files.append(fp)
      elif ext == '.rtf':
        files.append(fp)
        with open(fp, 'r') as file:
          text = file.read()
          document_t = rtf_to_text(text).replace('\n', ' ').replace('\t', ' ')
          keywords_set = clean_keywords(keywords_from_summary(summarize(document_t)))
          document = document_t
      elif ext == '.docx':
        text = getText(fp)
        document = text
        keywords_set = clean_keywords(keywords_from_summary(summarize(text)))
        files.append(fp)
      else:
        files.append(fp)
        meta_path = os.path.dirname(fp) + '\metadata.csv'
        des = pd.read_csv(meta_path, encoding = 'unicode_escape')
        try:
          description = des['Description'][0]
        except:
          description = des['Title'][0]
        document = description
        keywords_set = clean_keywords(keywords_from_summary(description))
      file_keywords.append(keywords_set)
      documents.append(document)
  # print(documents)
  return file_keywords, documents


# Finding similarities between file-keywords and the intent:
def keyword_similarity(file_keywords, term):
  print("Finding Keyword Similarity...")
  track = []
  for keyword_set in file_keywords:
    temp = {}
    sum = 0
    count = 0
    token1 = nlp(term)
    for key in keyword_set:
      if len(key) < 20 and len(key) > 3 and token1.has_vector: 
        token2 = nlp(key)
        if token2.has_vector and token1.has_vector:
          cal = token1.similarity(token2)
          if cal != 0:
            sum += cal
            count += 1
        else:
          pass
    if count != 0:
      track.append(sum/count)
  return track


# Controller function to execute faster solution:
def faster_solution(track, file_paths):
  print("Sorting through files...")
  scores = []
  for i in range(len(track)):
    scores.append([track[i], file_paths[i]])
  scores.sort(key=lambda x: x[0], reverse=True)
  maximum_score_item = scores[0]
  maximum_score = maximum_score_item[0]
  files = []
  # print(scores)
  for i in scores:
    if maximum_score > 0.5:
      if i[0] > maximum_score - 0.4 * maximum_score:
        files.append(i[1])
    elif maximum_score > 0.3 and maximum_score < 0.5:
      if i[0] > maximum_score - 0.15 * maximum_score:
        files.append(i[1])
    else:
      if i[0] > maximum_score - 0.12 * maximum_score:
        files.append(i[1])
  return files


# Summarize Text:
def get_summaries(documents):
  print('Summarizing the contents of the files...')
  summaries = []
  for doc in documents:
    if len(doc.split('. ')) > 15:
      summary = summarize(doc)
    else:
      summary = doc
    if summary == '':
      summaries.append(doc)
    else:
      summaries.append(summary)
  return summaries


# Get sentences from Text
def get_sentences(summaries):
  print('Extracting sentences from summaries...')
  files_sent = []
  for summary in summaries:
    temp = []
    temp2 = []
    for s in summary.split('.'):
      s = s.replace('\n', '').replace('\t', ' ')
      temp.append(s)
      temp = list(set(temp))
      for t in temp:
        if t != '':
          temp2.append(t)
    files_sent.append(temp2)
  return files_sent


# Fetching the articles:
def loadRSS(url): 

  # creating HTTP response object from given url 
  resp = requests.get(url) 

  # saving the xml file 
  with open('intent.xml', 'wb') as f: 
      f.write(resp.content) 


def parseXML(xmlfile): 
  
  # create element tree object 
  tree = ET.parse(xmlfile) 

  # get root element 
  root = tree.getroot() 
  summary_tag = root.tag.split('}')[0] + '}summary'
  
  docs = []
  for x in tree.iter():
    if x.tag == summary_tag:
      docs.append(x.text.replace('\n', ' ').replace('\t', ' '))
  return docs


def quote_plus(word):
  s = ''
  for w in word.split(' '):
    s += w.lower() + '+'
  return s[:-1]


def get_dataset(term):
  print('Fetching data related to the intent entered...')
  url = 'http://export.arxiv.org/api/query?search_query=all:' + term
  loadRSS(url)
  dataset = parseXML('intent.xml')
  return dataset


# Cleaning a set of sentences:
def cleaned_dataset(dataset):
  print('Cleaning fetched data...')
  split_dataset = []
  temp = []
  for j in dataset:
    if len(j.split(' ')) > 5:
      temp.extend(j.split('. '))
  temp2 = []
  for t in temp:
    if t != '':
      temp2.append(t)
  split_dataset.append(temp2)
  return split_dataset


# Calculating the sentence similarities
def sentence_similarity(files_sent, dataset, filepaths):
  print('Calculating Sentence Similarities...')
  sentences = dataset[0]
  intent_vect = sbert_model.encode(sentences[:10])
  track = []
  index = 0
  for file in files_sent:
    file_vect = sbert_model.encode(file[:10])
    sum = 0
    count = 0
    for i in file_vect:
      for j in intent_vect:
        dis =  spatial.distance.cosine(i, j)
        sim = 1 - dis
        if sim != 0:
          sum += sim
          count += 1
    if count != 0:
      cal = sum / count
      track.append([cal, filepaths[index]])
      index += 1
  return track


# Controller Function for Deeper Solution:
def deeper_solution(scores):
  print("Sorting through files...")
  scores.sort(key=lambda x: x[0], reverse=True)
  # print(scores)
  maximum_score_item = scores[0]
  maximum_score = maximum_score_item[0]
  # print(maximum_score)
  files = []
  for i in scores:
    if maximum_score > 0.5:
      if i[0] > maximum_score - 0.05:
        files.append(i[1])
    elif maximum_score > 0.3 and maximum_score < 0.5:
      if i[0] > maximum_score - 0.04:
        files.append(i[1])
    else:
      if i[0] > maximum_score - 0.02:
        files.append(i[1])
  return files


def print_locations(files):
  for i in files:
    print(i)
  return


# Demo with Search Terms:
continue_on = True
filepaths = find_files()
file_keywords, file_sent = find_keywords(filepaths)

while continue_on:
  term = input('Enter Search Term: ')
  print('Enter the type of search to perform: ')
  print('1. Quick Search')
  print('2. Deep Search')
  choice = int(input('Your Choice: '))

  if choice == 1:
    print('Running Quick Search for ', term)
    start = time.time()
    similarities = keyword_similarity(file_keywords, term)
    result = faster_solution(similarities, filepaths)
    print()
    print('-----------OUTPUT-----------')
    print_locations(result)
    end = time.time()
    print("Search took ", end - start, " seconds.")
  else:
    print('Running Deep Search for ', term)
    start = time.time()
    summaries = get_summaries(file_sent)
    sentences = get_sentences(summaries)
    dataset = get_dataset(term)
    cleaned_dataset_item = cleaned_dataset(dataset)
    similarities = sentence_similarity(sentences, cleaned_dataset_item, filepaths)
    print()
    print('-----------OUTPUT-----------')
    if similarities:
      result = deeper_solution(similarities)
      print_locations(result)
    else:
      print('None of the files are similar to the given intent...')
    end = time.time()
    print("Search took ", end - start, " seconds.")

  option = int(input('Press 1 to continue...'))
  if option == 1:
    continue_on = True
  else:
    continue_on = False




