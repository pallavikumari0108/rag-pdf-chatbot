from dotenv import load_dotenv
import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def get_llm():
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        api_key = st.secrets.get("GOOGLE_API_KEY")

    st.write("API Key Exists:", bool(api_key))

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=api_key,
            temperature=0,
        )

        # Test the API immediately
        llm.invoke("Hello")

        st.success("Gemini API Connected Successfully!")
        return llm

    except Exception as e:
        st.error(f"Actual Error: {e}")
        raise