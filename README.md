# Разработка чат-бота ассистента студента

## rag.ipynb

Основной код для вопрос-ответного бота на основе LangChain. Здесь текстовые материалы разбиваются на фрагменты, индексируются с помощью эмбеддингов и сохраняются в векторную базу данных. Затем по запросу из базы данных извлекаются релевантные документы, и подаются на вход модели Yandex GPT

## telegram.py
Код телеграм-бота на основе фреймворка flask
