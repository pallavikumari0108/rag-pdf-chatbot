from dotenv import load_dotenv
import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def get_llm():
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        api_key = st.secrets["GOOGLE_API_KEY"]

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0,
    )