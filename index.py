from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
from PyPDF2 import PdfReader
import os

apiKey = "sk-proj-jxyZAem6t60o6Wcw5fwdT3BlbkFJgb6m74eAQTZXgY3qv1V4"


def index(path):
    # read the pdf file content and store as a txt file
    reader = PdfReader(path)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    with open('content.txt', 'w') as f:
        f.write(text)

    # create a loader obj to load the txt file
    loader = TextLoader("content.txt")
    data = loader.load()

    # create a text splitting obj for processing/training.. having 10 chunks
    textSplitter = CharacterTextSplitter(separator = '\n', chunk_size = 512, chunk_overlap = 128)
    chunks = textSplitter.split_documents(data)

    # vector embeddings of the chunks
    embeddings = OpenAIEmbeddings(openai_api_key = apiKey)
    pd = 'db'  # to avoid creating it again and again
    vectordb = Chroma.from_documents(documents = chunks, embedding = embeddings, persist_directory = pd)
    vectordb.persist()

index('sample2.pdf')
