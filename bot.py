import gradio as gr
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import openai
import os

apiKey = "sk-proj-jxyZAem6t60o6Wcw5fwdT3BlbkFJgb6m74eAQTZXgY3qv1V4"
os.environ["OPENAI_API_KEY"] = apiKey
with open('content.txt', 'r') as f:
    pdf_content = f.read()


def classify_topic(paragraph):
    print("PARA -", paragraph)

    if paragraph in pdf_content:
        prompt = f"Classify the following paragraph into a topic:\n\n{paragraph}"
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": prompt}]
        )
        topic = response['choices'][0]['message']['content'].strip()
        return topic
    else:
        return "I don't know"


def answer_question(question):
    return "Placeholder answer to PDF content question"


def add_text(history, text):
    history = history + [(text, None)]
    return history, ""


def bot(history):
    question = history[-1][0]

    if "classify" in question.lower() and "topic" in question.lower():
        topic = classify_topic(question)
        history[-1][1] = f"Topic: {topic}"
    else:
        res = retrChain({'question': history[-1][0], 'chat_history': history[:-1]})
        answer = res['answer']
        history[-1][1] = f"Answer: {answer}"

    return history


def conversation():
    pd = 'db'
    embeddings = OpenAIEmbeddings(openai_api_key=apiKey)

    db = Chroma(persist_directory=pd, embedding_function=embeddings)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    retrChain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(openai_api_key=apiKey),
        chain_type='stuff',
        retriever=db.as_retriever(),
        memory=memory,
        get_chat_history=lambda h: h,
    )

    return retrChain


retrChain = conversation()


with gr.Blocks() as trial:
    chatbot = gr.Chatbot([], elem_id="chatbot", label='DSA').style(height=500)
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


if __name__ == '__main__':
    trial.queue(concurrency_count=3)
    trial.launch()
