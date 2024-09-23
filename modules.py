import re
import os
import faiss
import time
import warnings
import requests
import html5lib
import numpy as np
import google.generativeai as genai
import logging

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from requests.exceptions import RequestException

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')
load_dotenv()

# API setup
 # Ensure you have the API key in your .env file
genai.configure(api_key='AIzaSyCFXeORI_4dVplI0Jj-S0f6i3NRNKDVHUU')
MODEL_ID = "models/text-embedding-004"
MAX_LINKS_PER_PAGE = 10
DEFAULT_CHUNK_SIZE = 500

link_count = 0

# Function to crawl the search results and scrape the sites
def googlebot(url, sites_required):
    global link_count
    scraped_text = ""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36'
    }
    extracted_links = {}
    sites_to_scrape = sites_required + 30
    page_count = 0

    # Extracting the search links pagewise
    while len(extracted_links) < sites_to_scrape:
        page_temp = page_count * MAX_LINKS_PER_PAGE
        url_temp = url[:-2] if page_count > 1 else url
        url = f"{url_temp}{page_temp}"

        try:
            url_open = requests.get(url, headers=headers)
            url_open.raise_for_status()  # Check for HTTP errors
            soup = BeautifulSoup(url_open.content, 'html5lib')
            for result in soup.find_all('div', class_='g'):
                heading = result.find('h3')
                link = result.find('a', href=True)
                if heading and link:
                    text = heading.get_text(strip=True)
                    text_url = link['href']
                    extracted_links[link_count + 1] = (text, text_url)
                    link_count += 1
            page_count += 1
        except RequestException as e:
            logger.error(f"Error fetching URL {url}: {e}")
            break

    logger.info("Extracted links:")
    for x, (y, l) in extracted_links.items():
        logger.info(f"{x}. {y} - {l}")

    # Scraping the required number of links
    sites = 0
    for i in range(1, sites_required + 1):
        link = extracted_links.get(i, (None, None))[1]
        if not link or any(x in link for x in ['/maps', '/search?', 'www.instagram', '/shop/', "myntra", "play.google", "flipkart.com", "amazon.in"]):
            logger.warning(f"Skipping site no. {i} due to invalid link.")
            continue
        try:
            url_open2 = requests.get(link)
            url_open2.raise_for_status()
            soup2 = BeautifulSoup(url_open2.content, 'html5lib')
            for j in soup2.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                if j.name not in ['a', 'img']:
                    text = j.get_text(strip=False)
                    if text:
                        scraped_text += text
            sites += 1
            logger.info(f"Scraped: {i}, TOTAL SCRAPED SITES: {sites}")
        except RequestException as e:
            logger.error(f"Error scraping site no. {i}: {e}")

    logger.info(f"TOTAL SCRAPED SITES: {sites}")
    return scraped_text

def urlbot(url):
    ans = ""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        soup = BeautifulSoup(response.content, 'html5lib')
        for i in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            if i.name not in ['a', 'img']:
                text = i.get_text(strip=False)
                if text:
                    ans += text
        return ans
    except RequestException as e:
        logger.error(f"Request error for URL {url}: {e}")
    except Exception as e:
        logger.error(f"Error while processing URL {url}: {e}")

# Check query and initiate scraping
def search_and_scrape(q, https_match, sites_required):
    logger.info("Initializing search and scrape.")
    if re.match(https_match, q):
        url = q
        return urlbot(url)
    else:
        lists = q.split()
        word = "+".join(lists)
        url = "https://www.google.com/search?q=" + word + "&start="
        logger.info("Calling web scraper.")
        return googlebot(url, sites_required)

# Speeding up embeddings by use of thread management
def parallel_embed_text_chunks(chunks):
    with ThreadPoolExecutor(max_workers=5) as executor:  # Limit number of threads
        embeddings = list(executor.map(embed_text_chunk, chunks))
    return embeddings

# Creating embeddings
def embed_text_chunk(content, retries=3, backoff_factor=1):
    for i in range(retries):
        try:
            result = genai.embed_content(
                model=MODEL_ID,
                content=content,
                task_type="retrieval_document",
                title="Embedding of single string"
            )
            return np.array(result['embedding'], dtype='float32')
        except Exception as e:
            logger.error(f"Error embedding text chunk ({e}). Retrying in {backoff_factor * (2 ** i)} seconds...")
            time.sleep(backoff_factor * (2 ** i))
    logger.warning("Max retries exceeded for embedding text chunk.")
    return np.zeros((1, 512), dtype='float32')  # Adjust dimensions if necessary

# Creating embeddings for the query
def embed_query(query):
    logger.info("Creating embedding for the query.")
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=query,
        task_type="retrieval_document",
        title="Embedding of user query"
    )
    return result['embedding']

# Finding the relevant chunks out of total chunks
def retrieve_relevant_chunks(query_embedding, index, text_chunks, sites_required):
    no_of_relevant_chunks = max(20, sites_required * 2)
    n = len(text_chunks)
    no_of_relevant_chunks = min(no_of_relevant_chunks, n)
    logger.info("Retrieving relevant chunks.")
    query_vector = np.array([query_embedding], dtype='float32')
    _, indices = index.search(query_vector, no_of_relevant_chunks)
    relevant_chunks = [text_chunks[i] for i in indices[0]]
    return relevant_chunks

# Creating chunks from the scraped text
def chunk_text(text, chunk_size):
    logger.info("Creating chunks.")
    chunk_list = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    return chunk_list

# Indexing and storing the embeddings
def store_embeds(embeddings):
    logger.info("Storing embeddings.")
    faiss_embeddings = np.array(embeddings, dtype='float32')
    dimension = faiss_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(faiss_embeddings)
    return index

# Initializing geminiAPI to generate response based on relevant chunks and the query provided
def generateResponse(query, relevant_chunks):
    logger.info("Formulating response.")
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    context = " ".join(relevant_chunks)
    prompt = (f"The query given by the user follows after the colon: {query}\n"
              "Strictly use the following text as the database only to generate responses for the previous query. "
              "Strictly do not use your own database or knowledge about the query. "
              "The response length should be directly proportional to the number of relevant chunks. "
              "The text will follow after this colon:\n" + context)

    response = model.generate_content(prompt)
    return response.text

def main(q, https_match, sites_required):
    global link_count
    link_count = 0
    text = search_and_scrape(q, https_match, sites_required)  # Scrape the web
    chunks = chunk_text(text, DEFAULT_CHUNK_SIZE)  # Divide into chunks
    logger.info(f"Total number of chunks: {len(chunks)}")
    logger.info("Creating embeddings.")
    embeddings = parallel_embed_text_chunks(chunks)  # Make embeddings for the chunks
    index = store_embeds(embeddings)  # Indexing of embeds
    q_embedding = embed_query(q)  # Embedding the query
    relevant_chunks = retrieve_relevant_chunks(q_embedding, index, chunks, sites_required)
    return generateResponse(q, relevant_chunks)

