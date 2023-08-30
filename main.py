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
# Descargar los recursos necesarios
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
# Descargar recursos necesarios
nltk.download('punkt')
nltk.download('stopwords')
import pandas as pd
from tqdm import tqdm
import spacy

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


	return '\n'.join( paragraphs)


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




if __name__=='__main__':

	text = scrapy()

	# Título de la página
	st.title("NLP Entity Recognition and Summarization of Magazine Articles using NLTK, spaCy, and Hugging Face.")

	introduction = """

	**Introduction:**

	In today's information-rich landscape, magazines and online publications are abundant sources of knowledge and trends. However, the sheer volume of data can be overwhelming. This is where Natural Language Processing (NLP) comes into play, a field of artificial intelligence focused on enabling computers to understand and process human language.

	This project revolves around the combined application of NLTK (Natural Language Toolkit), spaCy, and Hugging Face to perform entity recognition and automatic summarization of articles from an online magazine. These libraries offer advanced tools and techniques for processing, analyzing, and deriving insights from textual data.

	**Objectives:**

	1. **Data Extraction:** The first step involves utilizing web scraping techniques to extract article contents from the online magazine. This includes capturing titles, content, and relevant metadata.

	2. **Text Preprocessing:** Before analysis, data must be cleaned and prepared. NLTK, spaCy, and Hugging Face offer functionalities for tasks like tokenization, stemming, stopwords removal, and more, ensuring the text is in a suitable format for analysis.

	3. **Entity Recognition:** Both NLTK and spaCy provide named entity recognition capabilities to identify specific names, locations, and organizations mentioned in the text. This helps extract key entities that play a significant role in the articles.

	4. **Automatic Summarization:** NLTK, spaCy, and Hugging Face's summarization tools assist in distilling key ideas from articles, generating coherent and concise summaries. This simplifies the comprehension of extensive textual content.

	**Benefits:**

	Applying NLTK's, spaCy's, and Hugging Face's entity recognition and summarization capabilities to magazine articles provides valuable insights. By extracting key entities and generating summaries, articles' core information becomes easily accessible, aiding in efficient analysis and understanding.

	**Conclusion:**

	Leveraging the combined prowess of NLTK, spaCy, and Hugging Face in entity recognition and automatic summarization empowers researchers to unlock essential insights from online magazine articles. Through data extraction, preprocessing, entity recognition, and summarization, these libraries streamline the process of understanding and summarizing extensive textual content, thereby facilitating informed decision-making and knowledge dissemination.
	"""



	# Mostrar la introducción en Streamlit
	st.markdown(introduction)

	texto_con_enlace = "¡Extract the content of this post with scrapy > [I want to open a window in their souls’: Haruki Murakami on the power of writing simply](https://www.theguardian.com/books/2022/nov/05/i-want-to-open-a-window-in-their-souls-haruki-murakami-on-the-power-of-writing-simply)"
	st.markdown(texto_con_enlace)
	st.text_area("Text:", value = text, height = 300 )
	
	# ENTITY RECOGNITION


	# Título de la página
	st.title("Entity Recognition with NLP Libraries")

	selected_library = st.selectbox("Select the library to do entity recognition and summary ", ["NLTK", "spaCy","Hugging Face"],index=0)


	if selected_library == "NLTK":
		st.write("NLTK (Natural Language Toolkit) is a comprehensive library for natural language processing.")


		entidades_reconocidas = reconocedor_de_entidades(text)

		st.subheader("Entities")

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


		st.write(summarize(text))


	elif selected_library == "spaCy":
		st.write("spaCy is a modern NLP library that excels in processing large volumes of text efficiently.")

		# Carga el modelo en inglés
		nlp = spacy.load("en_core_web_sm")

		st.title("SPACY")
		def extract_named_entities(text):
		    # Procesa el texto con spaCy
		    doc = nlp(text)

		    # Extrae y muestra las entidades nombradas
		    entities = []
		    for ent in doc.ents:
		        entities.append((ent.text, ent.label_))

		    return entities

		# Ejemplo de uso

		entitis = extract_named_entities(text)

		for ent in set(entitis):
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




		st.subheader("NLP - Summarization of Magazine Articles: Unveiling Insights with Spacy")
		# Carga el modelo en inglés de spaCy
		nlp = spacy.load("en_core_web_sm")

		def summarize_spacy(text, num_sentences=5):
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
			return ' '.join(sorted(top_sentences, key=sentences.index))


		st.write(summarize_spacy(text))

	elif selected_library ==  "Hugging Face" :
	
		st.markdown(" [Analysis with Hugging Face in this link (COLAB)](https://colab.research.google.com/drive/1J6R20SSRdx9y8GMyiayYlaMnrQVBOvaa#scrollTo=RviFJwTTVid7)")



