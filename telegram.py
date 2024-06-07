import json
from flask import Flask, request
from telegram import Bot, Update
from telegram.ext import Dispatcher, CommandHandler, MessageHandler, Filters, CallbackContext
from langchain.chains import StuffDocumentsChain
from langchain.llms import YandexLLM
from langchain.embeddings import YandexEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.prompts import PromptTemplate

app = Flask(__name__)

# Загружаем конфигурацию
with open('config.json') as f:
    config = json.load(f)

self_url = config['self_url']
api_key = config['api_key']
telegram_token = config['telegram_token']
folder_id = config['folder_id']
source_dir = config['source_dir']

# Инициализируем Telegram Bot
bot = Bot(token=telegram_token)
embeddings = YandexEmbeddings(folder_id=folder_id, api_key=api_key)
docs = DirectoryLoader(source_dir).load()
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever()

# Инициализируем LLM и цепочку
instructions = """
Представь себе, что ты ассистент студента Московского Авиационного Института.
"""
llm = YandexLLM(api_key=api_key, folder_id=folder_id, instruction_text=instructions)
document_prompt = PromptTemplate(
    input_variables=["page_content"], template="{page_content}"
)
document_variable_name = "context"
stuff_prompt_override = """
Пожалуйста, посмотри на текст ниже и ответь на вопрос, используя информацию из этого текста.
Текст:
-----
{context}
-----
Вопрос:
{query}
"""
prompt = PromptTemplate(
    template=stuff_prompt_override, input_variables=["context", "query"]
)
llm_chain = StuffDocumentsChain(
    llm_chain=llm,
    document_prompt=document_prompt,
    document_variable_name=document_variable_name,
)

# Обработчик команды /start
def start(update: Update, context: CallbackContext):
    update.message.send_message('Доброго времени суток, я с радостью помогу Вам найти ответы на Ваши вопросы!')

# Обработчик текстовых сообщений
def respond(update: Update, context: CallbackContext):
    user_id = update.message.from_user.id
    if user_id not in context.user_data:
        update.message.reply_text("Для начала работы с ботом нажмите /start")
        return
    question = update.message.text
    docs = retriever.get_relevant_documents(question)
    response = llm_chain.run(question, docs)
    update.message.send_message(response)

# Добавляем обработчики команд и сообщений
dispatcher.add_handler(CommandHandler("start", start))
dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, respond))

@app.route('/telegram', methods=['POST'])
def telegram_hook():
    update = Update.de_json(request.get_json(force=True), bot)
    dispatcher.process_update(update)
    return 'ok'

if __name__ == '__main__':
    app.run(port=8000)
