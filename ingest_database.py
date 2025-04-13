from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from uuid import uuid4
import re
import unicodedata  # Import the unicodedata module

# import the .env file
from dotenv import load_dotenv
load_dotenv()

# configuration
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

# initiate the embeddings model
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

# initiate the vector store
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

# loading the PDF document
loader = PyPDFDirectoryLoader(DATA_PATH)

raw_documents = loader.load()

# splitting the document
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)

# creating the chunks
chunks = text_splitter.split_documents(raw_documents)

def clean_chunk(chunk):
    """
    Cleans a chunk by removing extra whitespace, control characters, and ensuring it's not empty.
    """
    text = chunk.page_content

    # Normalize Unicode characters (e.g., combining diacritics)
    text = unicodedata.normalize("NFKD", text)

    # Replace problematic characters and patterns with spaces
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]", " ", text) # Remove control characters and extended ASCII
    text = re.sub(r"ﬁ|ﬂ|ﬀ|ﬃ|ﬄ", "ff", text)  # Replace ligatures
    text = re.sub(r"Table1", "", text) # remove "Table1" pattern in text

    # Replace multiple whitespace characters with a single space
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Filter out problematic chunks using the cleaning function
filtered_chunks = [chunk for chunk in chunks if clean_chunk(chunk)]

# creating unique ID's
uuids = [str(uuid4()) for _ in range(len(filtered_chunks))]

# adding chunks to vector store
vector_store.add_documents(documents=filtered_chunks, ids=uuids)

print(f"Added {len(filtered_chunks)} chunks to ChromaDB")
