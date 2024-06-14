# DSA Chatbot

This project consists of a chatbot designed to assist with questions specifically related to data structures and algorithms. The chatbot uses the OpenAI GPT-4 model to generate responses based on the content extracted from a PDF file. The project leverages NLP techniques such as tokenization, lemmatization, and removal of stopwords to process the text and plot its n-gram frequencies, and it utilizes Gradio for the user interface.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)

## Features

- Extracts text from a PDF document.
- Processes text to tokenize, lemmatize, and remove stopwords and numbers.
- Plots top 20 n-grams for each page in the document.
- Uses OpenAI's GPT-4 to generate answers based on the content.
- Provides a Gradio-based chatbot interface for user interaction.

## Prerequisites

Ensure you have the following installed:

- Python 3.7+
- pip (Python package installer)
- [OpenAI API Key](https://beta.openai.com/signup/)

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/varuni24/DSA-Chatbot.git
    cd DSA-Chatbot
    ```

2. Create and activate a virtual environment (optional but recommended):

    ```sh
    virtualenv -p /usr/local/bin/python3.11 venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

4. Download the necessary NLTK data:

    ```sh
    python -m nltk.downloader punkt stopwords wordnet
    ```

5. Place your PDF file (e.g., `Dsa.pdf`) in the project directory.

## Usage

1. **Set Up OpenAI API Key**:

    The script is designed to prompt the user for their OpenAI API key.

2. **Run the Indexing Script**:

    This script processes the PDF file, extracts text, and plots the n-grams.

    ```sh
    python index.py
    ```

    The script will generate and save JSON files (`content_dict.json` and `page_classification.json`) and n-gram bar plots for each page.

3. **Run the Chatbot Interface**:

    Start the Gradio interface to interact with the chatbot.

    ```sh
    python bot.py
    ```

    Open your browser and navigate to the provided URL to interact with the chatbot.



