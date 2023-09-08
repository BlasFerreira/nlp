import torch
from transformers import BertTokenizer, BertForTokenClassification
from transformers import RobertaTokenizer, RobertaForQuestionAnswering
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline   # Pipeline for easy-to-use model interfaces
from tqdm import tqdm
import nltk
nltk.download('punkt')
import sentencepiece
from transformers import T5Tokenizer, AutoModelWithLMHead
from bs4 import BeautifulSoup
import requests
import streamlit as st

session = requests.Session()

headers = {
    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8,es;q=0.7",
    "Cookie": "GU_mvt_id=546190; bwid=idFromPV_tGNE1Y4ziW6RF9ZU7oKWAQ; bwid_withoutSameSiteForIncompatibleClients=idFromPV_tGNE1Y4ziW6RF9ZU7oKWAQ; consentUUID=7e54c557-2b08-429f-9b3a-62d04932b3aa_22; consentDate=2023-08-15T12:41:50.817Z; _ga=GA1.2.1086360360.1692103312; _gid=GA1.2.362089074.1692103312; permutive-id=e6896ed3-6a89-426c-bced-1b3e2f395993; _cc_id=6b76286e9308ea51392d6993ac96cd0b; panoramaId_expiry=1692708112890; panoramaId=8b4cbd9cd4e1289b855afdf3abb74945a7027a222951e8665464c8751b3a5aeb; panoramaIdType=panoIndiv",
    "Referer": "https://www.theguardian.com/books/2022/nov/05/i-want-to-open-a-window-in-their-souls-haruki-murakami-on-the-power-of-writing-simply",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
}

# Update the session with the headers
session.headers.update(headers)

# Now you can make requests with this session and the headers will be used automatically
response = session.get("https://www.theguardian.com/books/2022/nov/05/i-want-to-open-a-window-in-their-souls-haruki-murakami-on-the-power-of-writing-simply")

# Parse the content with BeautifulSoup
soup = BeautifulSoup(response.content, 'html.parser')

# Initialize the variable that will contain the full content
paragraphs = []

# Extract the title and add it to the content
title = soup.find('h1')
if title:
    paragraphs.append(title.get_text())

# Iterate through the article content and add each element in the order it appears
for element in soup.find_all(['h2', 'blockquote', 'p']):
    paragraphs.append(element.get_text())

print(paragraphs)

paragraphs = paragraphs[:-3]
print('\n'.join( paragraphs))


model_bert_large_name  = "dbmdz/bert-large-cased-finetuned-conll03-english"

model_model_bert_large = BertForTokenClassification.from_pretrained(model_bert_large_name )
tokenizer_bert_large   = BertTokenizer.from_pretrained(model_bert_large_name )
nlp_ner = pipeline("ner", model=model_model_bert_large, aggregation_strategy="simple", tokenizer = tokenizer_bert_large)
print(f'model_max_length : {tokenizer_bert_large.model_max_length}')

def extract_entity(text_list):
  # A NER pipeline is set up, and entities from text_list are added to the entities list.
  entities = []
  for paragraph in text_list:
    entity = nlp_ner(paragraph)
    entities.extend(entity)

  # Delete duplicates
  seen_words = set()
  unique_entities = [entry for entry in entities if entry['word'] not in seen_words and not seen_words.add(entry['word'])]

  return unique_entities

if __name__=='__main__':
    
    unique_data = extract_entity(paragraphs)



    for entity_i in unique_data :
        st.write(f" Entity: {entity_i['word']}, Label: {entity_i['entity_group']}\n")


# entities_short = [item['word'] for item in unique_data]
# print(entities_short)

"""### **2 Point**

**Extract key elements and discern themes from the content of the article.**

The **chunks_creator function** takes a **text** content and **breaks** it into **chunks** using a **tokenizer**, ensuring that no chunk exceeds a specific number of tokens **(max_len)**. These chunks are formed based on the original content's sentences and are returned as a list.
"""

def chunks_creator(tokenizer,FileContent, max_len ) :


  # extract the sentences from the document
  sentences = nltk.tokenize.sent_tokenize(FileContent)

  # Create the chunks
  # initialize
  length = 0
  chunk = ""
  chunks = []
  count = -1
  for sentence in sentences:
    count += 1
    combined_length = len(tokenizer.tokenize(sentence)) + length # add the no. of sentence tokens to the length counter

    if combined_length  <= max_len : # if it doesn't exceed
      chunk += sentence + " " # add the sentence to the chunk
      length = combined_length # update the length counter

      # if it is the last sentence
      if count == len(sentences) - 1:
        chunks.append(chunk.strip()) # save the chunk

    else:
      chunks.append(chunk.strip()) # save the chunk

      # reset
      length = 0
      chunk = ""

      # take care of the overflow sentence
      chunk += sentence + " "
      length = len(tokenizer.tokenize(sentence))
  print(f'the function return : {len(chunks)} chunks.\n')

  # I divide the text into 4 chunks to be processed by the model
  print([len(tokenizer.tokenize(c)) for c in chunks ])

  return chunks

# Load a pretrained model and tokenizer designed for summarization
checkpoint_summ = "sshleifer/distilbart-cnn-12-6"

# Cargar el modelo y el tokenizador
tokenizer_summ  = AutoTokenizer.from_pretrained(checkpoint_summ)
model_summ  = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_summ)

# Display some stats related to the tokenizer
print("Maximum token length for the model:", tokenizer_summ.model_max_length)
print("Maximum token length for a single sentence:", tokenizer_summ.max_len_single_sentence)
print("Number of special tokens to add:", tokenizer_summ.num_special_tokens_to_add())

# I call the chunks_creator() to create chunks
chunks_summ = chunks_creator( tokenizer_summ , '\n'.join(paragraphs), 1022 )

# Convert chunks into inputs for the model
inputs_summ = [tokenizer_summ(chunk, return_tensors="pt") for chunk in chunks_summ]

"""#### Result"""

# Use the model to generate summaries for each chunk and display them
# Here is where we make the change, using max_new_tokens instead of max_length
summary = []
for input in inputs_summ:
    output_summary = model_summ.generate(**input, max_new_tokens=1022)
    summary.append(tokenizer_summ.decode(*output_summary, skip_special_tokens=True))
print('\n'.join(summary))

# """### **3 Point**

# **Create a basic recommendation system using Hugging Face's model to generate innovative ideas based on the extracted elements.**
# """

# # Loading a Question Answering model and tokenizer based on the "deepset/roberta-base-squad2" checkpoint.
# checkpoint_qa = 'distilbert-base-cased-distilled-squad'

# # Cargar el modelo y el tokenizador
# tokenizer_qa = RobertaTokenizer.from_pretrained("deepset/roberta-base-squad2")
# model_qa = RobertaForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")

# # Display some stats related to the tokenizer
# print("Maximum token length for the model:", tokenizer_qa.model_max_length)
# print("Maximum token length for a single sentence:", tokenizer_qa.max_len_single_sentence)
# print("Number of special tokens to add:", tokenizer_qa.num_special_tokens_to_add())

# # I call the chunks_creator() to create chunks
# # increase the margin from chunk size to token limit because, I have to add text and question tokens
# chunks_qa = chunks_creator( tokenizer_qa , '\n'.join(paragraphs), 460 )

# # Function to use the loaded QA model and tokenizer to find answers to a given question within a list of contexts.
# def question_answer(contexts,question):

#   print(f'Question : {question}')

#   for context in contexts:
#       inputs = tokenizer_qa.encode_plus(question, context, return_tensors="pt", truncation=True, max_length=512)
#       outputs = model_qa(**inputs)
#       answer_start_scores = outputs.start_logits
#       answer_end_scores = outputs.end_logits
#       answer_start = answer_start_scores.argmax()
#       answer_end = answer_end_scores.argmax() + 1
#       answer = tokenizer_qa.convert_tokens_to_string(tokenizer_qa.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
#       # print(f"Context: {context}\n")
#       if len(answer) > 3 :
#         print(f"\n Answer: {answer}\n")

# """#### **Result**"""

# question_answer(chunks_qa, 'Where born Murakami ?' )

# """## **Step 2: Sentiment Analysis and Emotional Assessment**

# ### **1 Point**

# **Employ a Hugging Face sentiment analysis model to ascertain the sentiment conveyed in the provided article.**
# """

# # Define the name of the sentiment analysis model from CardiffNLP
# name_model_sentiment = 'cardiffnlp/twitter-roberta-base-sentiment-latest'

# # Load the tokenizer associated with the sentiment model
# tokenizer_sentiment = AutoTokenizer.from_pretrained(name_model_sentiment)

# # Initialize a sentiment analysis pipeline using the specified model and tokenizer
# sentiment_task = pipeline("sentiment-analysis", model=name_model_sentiment, tokenizer=name_model_sentiment)

# # I call the chunks_creator() to create chunks
# chunks_sentiment = chunks_creator( tokenizer_sentiment , '\n'.join(paragraphs), 510 )

# """#### **Result**"""

# for chunk in chunks_sentiment:
#     try:
#         print(f"{sentiment_task(chunk)[0]['label'].upper()}", '\n')
#         print(chunk, '\n')
#     except Exception as e:
#         pass

# """**Emotional Assessment**"""

# # Load model directly
# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# # Use a pipeline as a high-level helper
# from transformers import pipeline

# pipe_emotions = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions")

# for chunk in chunks_sentiment :
#   try:
#     print(pipe_emotions(chunk)[0]['label'].upper(),'\n')
#     print(chunk , '\n')
#   except Exception as e:
#       pass

# """## **Step 3: Characters and Details Extraction**

# **Leverage Hugging Face's named entity recognition model to identify and extract character names and details from the article.**
# """

# print(f'model_max_length dbmdz/bert-large-cased-finetuned-conll03-english  : {tokenizer_bert_large.model_max_length}')

# # I call the chunks_creator() to create chunks
# chunks_entities = chunks_creator( tokenizer_bert_large , '\n'.join(paragraphs), 510 )

# # I have used the before function to extract entities from chunks
# # I use dbmdz/bert-large-cased-finetuned-conll03-english model
# unique_data = extract_entity(chunks_entities)

# # Label to filter
# ner_labels = ['PER', 'ORG', 'LOC', 'DATE', 'TIME', 'MISC', 'GPE', 'MONEY', 'PERCENT', 'QUANTITY', 'ORDINAL', 'CARDINAL']

# # filter relevant entities
# selected_entities = list(filter(lambda x: x['entity_group'] in ner_labels, unique_data ))

# """### **Result**"""

# # Show entities and labels
# for entity_i in selected_entities :
#     print(f" Entity: {entity_i['word']}, Label: {entity_i['entity_group']}\n")

# """## **Step 6: Question Generation**

# **Integrate a Hugging Face model tailored for question generation.**

# **Utilize the provided article as input to automatically generate pertinent questions that align with its content.**
# """

# model_name_qg = "mrm8488/t5-base-finetuned-question-generation-ap"
# tokenizer_qg = T5Tokenizer.from_pretrained(model_name_qg)
# model_qg = AutoModelWithLMHead.from_pretrained(model_name_qg)
# print(tokenizer_qg.max_len_single_sentence)

# # I call the chunks_creator() to create chunks
# chunks_qg = chunks_creator( tokenizer_qg , '\n'.join(paragraphs), 511 )

# # I set up a Hugging Face model to turn text chunks into questions.
# def generate_question(context, max_length=64):
#     input_text = "context: %s " % context
#     features = tokenizer_qg([input_text], return_tensors='pt')

#     output = model_qg.generate(input_ids=features['input_ids'],
#                attention_mask=features['attention_mask'],
#                max_length=max_length)

#     return tokenizer_qg.decode(output[0]).replace('<pad>', '').replace('</s>', '').strip()

# """### **Result**"""

# # Show Questions Generated
# for idx, chunk in enumerate(chunks_qg, 1):  # Comienza la enumeración desde 1
#     try:
#         print(f'{idx} {generate_question(chunk, max_length=64)}\n')
#     except Exception as e:
#         print(f'Error in question {idx}: {e}\n')