import gradio as gr
import openai
import json
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
import os

apiKey = input("Please enter your OpenAI API key: ")
os.environ["OPENAI_API_KEY"] = apiKey
embeddings = OpenAIEmbeddings(openai_api_key=apiKey)
db = Chroma(embedding_function=embeddings)

with open('content_dict.json', 'r') as f:
    content_dict = json.load(f)

for page, page_data in content_dict.items():
    page_text = page_data['text']
    page_embedding = embeddings.embed_documents([page_text])[0]
    db.add_texts([page_text], metadatas=[{"page": page}], embeddings=[page_embedding])

def add_text(history, text):
    history = history + [(text, None)]
    return history, ""

def bot(history):
    question = history[-1][0]
    results = db.similarity_search_with_score(question, k=3)
    sorted_pages = sorted(results, key=lambda x: x[1])
    relevant_content = "\n\n".join([f"Content from {doc.metadata['page']}:\n{doc.page_content}" for doc, score in sorted_pages])
    # print(relevant_content)

    system_prompt = (
        """You are a data structures and algorithms assistant who answers questions only related to the given content.
        If the question is not related to the given content, it should be replied with I dont know and dont give any pages.
        Otherwise use the following content to answer the question:\n\n"""
        f"{relevant_content}"
    )
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        temperature=0.1,
        api_key=apiKey
    )

    answer = response['choices'][0]['message']['content'].strip()
    # print(answer)
    if (answer != "I don't know."):
        history[-1][1] = f"Answer: {answer} (Pages: {', '.join([doc.metadata['page'] for doc, score in sorted_pages])})"
    else:
        history[-1][1] = f"Answer: {answer}"

    return history

def conversation():
    retr_chain = ConversationalRetrievalChain.from_llm(
        llm = ChatOpenAI(openai_api_key=apiKey, temperature = 0.1),
        chain_type = 'stuff',
        retriever = db.as_retriever(),
        get_chat_history = lambda h: h,
    )
    return retr_chain


retrChain = conversation()
with gr.Blocks() as chatbot:
    chatbot = gr.Chatbot([], elem_id="chatbot", label='DSA').style(height=680)
    with gr.Row():
        with gr.Column(scale=0.80):
            txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter")
        with gr.Column(scale=0.10):
            submitButton = gr.Button('Submit', variant='primary')
        with gr.Column(scale=0.10):
            clearButton = gr.Button('Clear', variant='stop')

    txt.submit(add_text, [chatbot, txt], [chatbot, txt]).then(bot, chatbot, chatbot)
    submitButton.click(add_text, [chatbot, txt], [chatbot, txt]).then(bot, chatbot, chatbot)
    clearButton.click(lambda: None, None, chatbot, queue=False)


chatbot.queue(concurrency_count=3)
chatbot.launch()
