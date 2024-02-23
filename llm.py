from dotenv import find_dotenv, load_dotenv
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

load_dotenv(find_dotenv())

llm = OpenAI(temperature=0.5)
embeddings = OpenAIEmbeddings()


def create_db_from_youtube_video_url(video_url):
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)
    return db


def get_response_from_query(db, query, k=4):
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    chat = OpenAI(model_name="gpt-3.5-turbo", temperature=0.5)

    # system message prompt
    template = """
        you are a helpful assistant that can answer questions based on given YouTube video's transcript {docs}

        only use the factual information from the transcript to answer the questions.

        If you feel like you don't have enough knowledge to answer the questions, say "I don't know about your topic. I'm sorry!'

        your answer should be verbose and detailed
    """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # human question prompt
    human_template = """Answer the following question : {question}"""
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(
        llm=llm,
        prompt=chat_prompt,
    )

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    response = response.replace("System: ", "")
    return response, docs
