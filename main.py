import streamlit as st
from bs4 import BeautifulSoup
import requests
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from transformers import AutoTokenizer,BertTokenizer,AutoModelForSeq2SeqLM, BertForTokenClassification

from transformers import pipeline 
import pandas as pd
from tqdm import tqdm
import spacy
# Descargar los recursos necesarios
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
# Descargar recursos necesarios
nltk.download('punkt')
nltk.download('stopwords')


def presentation(text_result):

	# Title
	st.title("NLP Entity Recognition and Summarization")

	## Subtitle
	st.header("Using NLTK, spaCy, and Hugging Face")

	# Introduction
	st.write("In today's information-rich landscape, online magazines provide a wealth of knowledge and trends, but the sheer volume can be overwhelming. This project uses NLTK, spaCy, and Hugging Face to perform entity recognition and automatic summarization of online magazine articles, simplifying complex content.")

	## Objectives
	st.subheader("Objectives:")
	st.write("1. **Data Extraction:** We scrape article contents, including titles and metadata.")
	st.write("2. **Text Preprocessing:** Prepare the data with tokenization and cleaning.")
	st.write("3. **Entity Recognition:** Identify names, locations, and organizations.")
	st.write("4. **Automatic Summarization:** These tools create concise summaries for easier comprehension.")

	## Benefits
	st.subheader("Benefits:")
	st.write("By leveraging NLTK, spaCy, and Hugging Face, we enhance article analysis, extract key entities, and simplify complex content for informed decision-making.")

	# Conclusion
	st.subheader("Conclusion:")
	st.write("Through entity recognition and summarization, NLTK, spaCy, and Hugging Face streamline understanding and knowledge dissemination from online magazine articles, empowering efficient research.")


# scrapy
def scrapy():
	# Start a session
	session = requests.Session()

	# Define the headers
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

	# print(paragraphs)

	paragraphs = paragraphs[:-3]


	return paragraphs


@st.cache_data()
def reconocedor_de_entidades(texto):
    # Tokenización
    palabras = word_tokenize(texto)

    # Etiquetado gramatical (POS tagging)
    palabras_etiquetadas = pos_tag(palabras)

    # Reconocimiento de entidades nombradas
    arbol_entidades = ne_chunk(palabras_etiquetadas)

    # Extraer entidades del árbol
    entidades = []
    for subtree in arbol_entidades:
        if isinstance(subtree, nltk.Tree):
            entidad = " ".join([word for word, tag in subtree.leaves()])
            etiqueta = subtree.label()
            entidades.append((entidad, etiqueta))

    return entidades


def extract_entity(text_list):
	# A NER pipeline is set up, and entities from text_list are added to the entities list.
	entities = []
	for paragraph in tqdm(text_list):
		entity = nlp_ner(paragraph)
		entities.extend(entity)

		# Delete duplicates
		seen_words = set()
		unique_entities = [entry for entry in entities if entry['word'] not in seen_words and not seen_words.add(entry['word'])]

	return unique_entities


def summarize(text, num_of_sentences=5):
    # Tokenizar oraciones y palabras
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())

    # Eliminar palabras vacías
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words and word.isalnum()]

    # Calcular frecuencia de palabras
    freq = nltk.FreqDist(words)

    # Puntuar oraciones basadas en la frecuencia de palabras
    sentence_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in freq:
                if sentence not in sentence_scores:
                    sentence_scores[sentence] = freq[word]
                else:
                    sentence_scores[sentence] += freq[word]

    # Seleccionar las oraciones con las puntuaciones más altas
    summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_of_sentences]
    return ' '.join(summary_sentences)


# @st.cache_data()
def download_bert() :
	model_bert_large_name  = "dbmdz/bert-large-cased-finetuned-conll03-english"

	model_model_bert_large = BertForTokenClassification.from_pretrained(model_bert_large_name )
	tokenizer_bert_large   = BertTokenizer.from_pretrained(model_bert_large_name )
	nlp_ner = pipeline("ner", model=model_model_bert_large, aggregation_strategy="simple", tokenizer = tokenizer_bert_large)

	return nlp_ner


@st.cache_data()
def entity_spacy(text_input) :

	st.subheader("Entity Recognition")

	# Carga el modelo en inglés
	nlp = spacy.load("en_core_web_sm")
	doc = nlp(text_input)

	# Extrae y muestra las entidades nombradas
	entities = []
	for ent in doc.ents:
	    entities.append((ent.text, ent.label_))

	return entities




def summarize_spacy(text, num_sentences=5):
	st.subheader("Spacy - Summarization of Magazine Articles: Unveiling Insights with Spacy")
	# Carga el modelo en inglés de spaCy
	nlp = spacy.load("en_core_web_sm")
	# Divide el texto en oraciones
	doc = nlp(text)
	sentences = [sent.text.strip() for sent in doc.sents]

	# Si el número de oraciones es menor o igual al número deseado de oraciones, devuelve el texto original
	if len(sentences) <= num_sentences:
		return text

	# Calcula la matriz TF-IDF para las oraciones
	vectorizer = TfidfVectorizer().fit_transform(sentences)
	vectors = vectorizer.toarray()

	# Calcula la similitud de términos para cada oración y ordena las oraciones por importancia
	rankings = np.argsort(np.sum(vectors, axis=1))[::-1]

	# Selecciona las oraciones más importantes
	top_sentences = [sentences[rank] for rank in rankings[:num_sentences]]

	# Ordena las oraciones seleccionadas en el orden en que aparecen en el texto original y las devuelve
	st.write(' '.join(sorted(top_sentences, key=sentences.index)))


def block_text(text_input):

	texto_con_enlace = "¡Extract the content of this post with scrapy > [I want to open a window in their souls’: Haruki Murakami on the power of writing simply](https://www.theguardian.com/books/2022/nov/05/i-want-to-open-a-window-in-their-souls-haruki-murakami-on-the-power-of-writing-simply)"
	st.markdown(texto_con_enlace)
	st.text_area("Text:", value = text_input, height = 200 )

# @st.cache_data()
def extract_entity(text_list):
	st.markdown(" [Analysis with Hugging Face in this link (COLAB)](https://colab.research.google.com/drive/1J6R20SSRdx9y8GMyiayYlaMnrQVBOvaa#scrollTo=RviFJwTTVid7)")

	nlp_ner= download_bert()
	# A NER pipeline is set up, and entities from text_list are added to the entities list.
	entities = []
	st.write('Extracting entities...')
	progress_bar = st.progress(0)
	for i, paragraphh in enumerate(text_list):
	    entity = nlp_ner(paragraphh)
	    entities.extend(entity)
	    
	    # Actualizar la barra de progreso
	    progress_bar.progress((i + 1) / len(text_list))


	# Delete duplicates
	seen_words = set()
	unique_entities = [entry for entry in entities if entry['word'] not in seen_words and not seen_words.add(entry['word'])]


	return unique_entities



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


@st.cache_data()
def distilbart_download():
	# Load a pretrained model and tokenizer designed for summarization
	checkpoint_summ = "sshleifer/distilbart-cnn-12-6"
	# Cargar el modelo y el tokenizador
	tokenizer  = AutoTokenizer.from_pretrained(checkpoint_summ)
	model  = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_summ)

	return tokenizer, model 


def summarize_hf(text_input): 

	tokenizer_summ,model_summ  = distilbart_download()

	# I call the chunks_creator() to create chunks
	chunks_summ = chunks_creator( tokenizer_summ , '\n'.join(text_input), 1022 )

	# Convert chunks into inputs for the model
	inputs_summ = [tokenizer_summ(chunk, return_tensors="pt") for chunk in chunks_summ]



	# Use the model to generate summaries for each chunk and display them
	# Here is where we make the change, using max_new_tokens instead of max_length
	summary = []
	for input in inputs_summ:
	    output_summary = model_summ.generate(**input, max_new_tokens=1022)
	    summary.append(tokenizer_summ.decode(*output_summary, skip_special_tokens=True))

	return summary



if __name__=='__main__':

	st.set_page_config(page_title="Blas", 
	                   page_icon=":robot_face:",
	                   layout="wide",
	                   initial_sidebar_state="expanded"
	                   )


	st.sidebar.header('Hi, Select tools')
	nav = st.sidebar.radio('',['Go to homepage', 'NLTK','spaCy', 'Hugging Face'])
	st.sidebar.write('')
	st.sidebar.write('')
	st.sidebar.write('')
	st.sidebar.write('')
	st.sidebar.write('')


	text = scrapy()

	if nav == 'Go to homepage':
		presentation(text)
		block_text(text)

	if nav == 'NLTK':
		st.title("NLTK (Natural Language Toolkit) ")   
		block_text(text)
		# nltk_analysis(text)
		entidades_reconocidas = reconocedor_de_entidades(' '.join(text))

		st.subheader("Entity Recognition")

		for ent in set(entidades_reconocidas):
			st.write(f'**Entity:** {ent[0]} **Label: {ent[1]}**')

		# Diccionario de descripciones de etiquetas
		label_descriptions = {
		"GPE": "Geopolitical Entity",
		"ORGANIZATION": "A formal group, such as companies, governments, or educational institutions.",
		"PERSON": "Refers to individuals, including fictional and real people.",
		"DATE": "Refers to calendar dates, such as year, month, day, etc.",
		"ORDINAL": "Represents rank or position in an ordered sequence (e.g., 'first', 'second').",
		"CARDINAL": "Represents numerical quantities (e.g., 'one', 'two', 'three')."
		}

		# Título de la página
		st.subheader("Entity Label Descriptions")

		# Mostrar descripciones de etiquetas
		for label, description in label_descriptions.items():
			st.write(f"**{label}:** {description}")

		st.subheader("NLP - Summarization of Magazine Articles: Unveiling Insights with NLTK")

		st.write(summarize(' '.join(text) ))


	if nav == 'spaCy':   
		st.title("spaCy")
		block_text(text)
		entities_spacy = entity_spacy(' '.join(text))

		for ent in set(entities_spacy):
			st.write(f'**Entity:** {ent[0]} **Label: {ent[1]}**')

		# Title
		st.subheader("Entity Label Descriptions")

		# Descriptions of entity labels
		descriptions = {
		"DATE": "Refers to calendar dates, such as year, month, day, etc.",
		"ORG": "Represents organizations, companies, institutions, etc.",
		"LANGUAGE": "Denotes the language mentioned in the text.",
		"TIME": "Represents times of day or clock time.",
		"GPE": "Stands for Geopolitical Entity, referring to countries, cities, states, etc.",
		"ORDINAL": "Represents rank or position in an ordered sequence (e.g., 'first', 'second').",
		"WORK_OF_ART": "Refers to titles of books, songs, movies, artworks, etc.",
		"NORP": "Stands for Nationalities, Religious, or Political groups.",
		"CARDINAL": "Represents numerical quantities (e.g., 'one', 'two', 'three')."
		}

		# Display descriptions of labels
		for labell, descriptionn in descriptions.items():
			st.write(f"**{labell}:** {descriptionn}")



		summarize_spacy(' '.join(text), num_sentences=5)

	if nav == 'Hugging Face':  
		st.title("Hugging Face")
		block_text(text)
		entities_hf = extract_entity(text)


		for entity_i in entities_hf :
		    st.write(f" Entity: {entity_i['word']}, Label: {entity_i['entity_group']}\n")   

		# st.subheader(" \n Hugging Face - Summarization of Magazine Articles: Unveiling Insights with Hugging Face \n")
		# summary_hf = summarize_hf(text)


		# st.write('\n'.join(summary_hf))

