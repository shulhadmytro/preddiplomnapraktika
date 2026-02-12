import streamlit as st
import numpy as np
import pickle
import json
import time
import random
import re
import tensorflow as tf
import pymorphy3
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Ініціалізація морфологічного аналізатора
morph = pymorphy3.MorphAnalyzer(lang='uk')

st.set_page_config(page_title="Python AI Tutor", page_icon="🐍", layout="centered")

@st.cache_resource
def load_assets():
    try:
        model = load_model('chatbot_model.h5')
        tokenizer = pickle.load(open('words.pkl', 'rb'))
        classes = pickle.load(open('classes.pkl', 'rb'))
        max_len = pickle.load(open('max_len.pkl', 'rb'))
        with open('intents.json', encoding='utf-8') as f:
            intents = json.load(f)
        return model, tokenizer, classes, max_len, intents
    except Exception as e:
        st.error(f"Помилка: {e}. Спершу запустіть train.py!")
        st.stop()

model, tokenizer, classes, max_len, intents = load_assets()

def get_best_match(user_query, responses):
    """Шукає відповідь. Якщо лем не знайдено, повертає випадкову з тегу."""
    user_words = re.findall(r'\w+', user_query.lower())
    user_lemmas = {morph.parse(word)[0].normal_form for word in user_words}
    
    best_res = None
    max_score = 0
    
    for resp in responses:
        clean_resp = re.sub(r'<[^>]+>', '', resp)
        resp_words = re.findall(r'\w+', clean_resp.lower())
        resp_lemmas = {morph.parse(word)[0].normal_form for word in resp_words}
        score = len(user_lemmas.intersection(resp_lemmas))
        
        if score > max_score:
            max_score = score
            best_res = resp
            
    return best_res if best_res else random.choice(responses)

def predict_intent(user_input):
    sequence = tokenizer.texts_to_sequences([user_input.lower()])
    padded = pad_sequences(sequence, maxlen=max_len, padding='post')
    prediction = model.predict(padded, verbose=0)[0]
    
    ERROR_THRESHOLD = 0.3 # Знижено поріг для кращої реакції
    results = [[i, r] for i, r in enumerate(prediction) if r > ERROR_THRESHOLD]
    
    if not results:
        return None
    results.sort(key=lambda x: x[1], reverse=True)
    return classes[results[0][0]]

def get_response_by_tag(tag):
    for i in intents['intents']:
        if i['tag'] == tag:
            return i['responses']
    return None

def get_intent_logic(user_input):
    user_input_low = user_input.lower().strip()
    
    # Автоматичний мапінг на основі ТЕГІВ з вашого JSON
    for intent in intents['intents']:
        tag = intent['tag']
        # Якщо назва тегу (наприклад "тип_int") або ключові слова є в запиті
        keyword = tag.replace("_", " ")
        if keyword in user_input_low or tag in user_input_low:
            return intent['responses'], tag
            
    ai_tag = predict_intent(user_input_low)
    if ai_tag:
        return get_response_by_tag(ai_tag), ai_tag
    return None, None

# --- UI ---
st.title("🐍 Python AI Tutor")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_topic_responses" not in st.session_state:
    st.session_state.current_topic_responses = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Запитайте про Python (напр. 'що таке float' або 'про списки')..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    resp_list, tag = get_intent_logic(prompt)
    
    with st.chat_message("assistant"):
        if resp_list:
            raw_answer = get_best_match(prompt, resp_list)
            st.session_state.current_topic_responses = [r for r in resp_list if r != raw_answer]
            final_answer = raw_answer
        else:
            final_answer = "Я поки не маю інформації про це у своїй базі знань. Спробуйте запитати про типи даних або оператори Python."

        placeholder = st.empty()
        full_response = ""
        for char in final_answer:
            full_response += char
            placeholder.markdown(full_response + "▌")
            time.sleep(0.005)
        placeholder.markdown(final_answer)
        st.session_state.messages.append({"role": "assistant", "content": final_answer})

if st.session_state.current_topic_responses:
    if st.button("📖 Дізнатися більше"):
        next_answer = st.session_state.current_topic_responses.pop(0)
        st.session_state.messages.append({"role": "assistant", "content": next_answer})
        st.rerun()