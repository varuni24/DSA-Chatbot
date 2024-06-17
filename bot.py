import gradio as gr
import openai
import json
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os

def initialize_chatbot(api_key):
    os.environ["OPENAI_API_KEY"] = api_key
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
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

        system_prompt = (
            """You are a data structures and algorithms assistant who answers questions only related to the given content.
            If the question is not related to the given content, it should be replied with "I don't know" and don't give any pages.
            Otherwise, use the following content to answer the question:\n\n"""
            f"{relevant_content}"
        )
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            temperature=0.1,
            api_key=api_key
        )

        answer = response['choices'][0]['message']['content'].strip()
        if answer != "I don't know.":
            history[-1][1] = f"Answer: {answer} (Pages: {', '.join([doc.metadata['page'] for doc, score in sorted_pages])})"
        else:
            history[-1][1] = f"Answer: {answer}"

        return history
    return add_text, bot


def start_chatbot():
    with gr.Blocks() as trial:
        api_key_input = gr.Textbox(label="Enter API Key", type="password")
        submit_button = gr.Button("Submit")
        chatbot = gr.Chatbot([], elem_id="chatbot", label='DSA').style(height=680)
        txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter")
        submitButton = gr.Button('Submit', variant='primary')
        clearButton = gr.Button('Clear', variant='stop')

        state = gr.State()

        def add_text_fn(history, text):
            add_text, _ = state.value
            return add_text(history, text)

        def bot_fn(history):
            _, bot = state.value
            return bot(history)

        def setup_chatbot(api_key):
            add_text, bot = initialize_chatbot(api_key)
            state.value = (add_text, bot)
            return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)

        submit_button.click(
            setup_chatbot,
            inputs=[api_key_input],
            outputs=[chatbot, txt, submitButton, clearButton]
        )
        txt.submit(add_text_fn, [chatbot, txt], [chatbot, txt]).then(bot_fn, chatbot, chatbot)
        submitButton.click(add_text_fn, [chatbot, txt], [chatbot, txt]).then(bot_fn, chatbot, chatbot)
        clearButton.click(lambda: None, None, chatbot, queue=False)
    trial.launch()

start_chatbot()
