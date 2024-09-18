import streamlit as st
from dotenv import load_dotenv
from api_handlers import OllamaHandler, PerplexityHandler, GroqHandler
from utils import generate_response, load_env_vars, final_answer  # Updated import
import json

# Load environment variables and configuration
load_dotenv()
config = load_env_vars()

def setup_page():
    st.set_page_config(page_title="multi1 - Unified AI Reasoning Chains", page_icon="🧠", layout="wide")
    st.markdown("""
    <h1 style='text-align: center; font-family: -apple-system, BlinkMacSystemFont, sans-serif;'>
        🧠 multi1 - Unified AI Reasoning Chains
    </h1>
    """, unsafe_allow_html=True)
    st.markdown("""
    <p style='text-align: center; font-family: -apple-system, BlinkMacSystemFont, sans-serif; font-size: 1.1em;'>
        This app demonstrates AI reasoning chains using different backends: Ollama, Perplexity AI, and Groq.
        Choose a backend and enter your query to see the step-by-step reasoning process.
    </p>
    """, unsafe_allow_html=True)

def get_api_handler(backend):
    if backend == "Ollama":
        return OllamaHandler(config['OLLAMA_URL'], config['OLLAMA_MODEL'])
    elif backend == "Perplexity AI":
        return PerplexityHandler(config['PERPLEXITY_API_KEY'], config['PERPLEXITY_MODEL'])
    else:  # Groq
        return GroqHandler(config['GROQ_MODEL'])

def display_config(backend):
    st.sidebar.markdown("## 🛠️ Current Configuration")
    if backend == "Ollama":
        st.sidebar.markdown(f"- 🖥️ Ollama URL: `{config['OLLAMA_URL']}`")
        st.sidebar.markdown(f"- 🤖 Ollama Model: `{config['OLLAMA_MODEL']}`")
    elif backend == "Perplexity AI":
        st.sidebar.markdown(f"- 🧠 Perplexity AI Model: `{config['PERPLEXITY_MODEL']}`")
    else:  # Groq
        st.sidebar.markdown(f"- ⚡ Groq Model: `{config['GROQ_MODEL']}`")

def main():
    setup_page()

    st.sidebar.markdown("<h3 style='font-family: -apple-system, BlinkMacSystemFont, sans-serif;'>⚙️ Settings</h3>", unsafe_allow_html=True)
    backend = st.sidebar.selectbox("Choose AI Backend", ["Groq", "Ollama", "Perplexity AI"])
    num_generations = st.sidebar.slider("Number of sequential generations", min_value=1, max_value=5, value=2)
    display_config(backend)
    api_handler = get_api_handler(backend)

    user_query = st.text_input("💬 Enter your query:", placeholder="e.g., How many 'R's are in the word strawberry?")

    if user_query:
        all_generations = []
        for generation in range(num_generations):
            st.write(f"🔍 Generating response GEN {generation + 1}")
            response_container = st.empty()
            time_container = st.empty()
            generation = []
            for steps, total_thinking_time in generate_response(user_query, api_handler):
                generation.append(steps)
                with response_container.container():
                    for title, content, _ in steps:
                        if title.startswith("Final Answer"):
                            st.markdown(f"<h3 style='font-family: -apple-system, BlinkMacSystemFont, sans-serif;'>🎯 {title}</h3>", unsafe_allow_html=True)
                            st.markdown(f"<div style='font-family: -apple-system, BlinkMacSystemFont, sans-serif;'>{content}</div>", unsafe_allow_html=True)
                        else:
                            with st.expander(f"📝 {title}", expanded=False):
                                st.markdown(f"<div style='font-family: -apple-system, BlinkMacSystemFont, sans-serif;'>{content}</div>", unsafe_allow_html=True)
                            
                if total_thinking_time is not None:
                    time_container.markdown(f"<p style='font-family: -apple-system, BlinkMacSystemFont, sans-serif;'><strong>⏱️ Total thinking time: {total_thinking_time:.2f} seconds</strong></p>", unsafe_allow_html=True)
            all_generations.append(generation)
        

        # Extract the last step from each result
        last_steps = json.dumps([result[-1][-1][1] for result in all_generations])
        summary = final_answer(user_query, last_steps, api_handler)  # Updated function call
        st.markdown(f"<h2 style='font-family: -apple-system, BlinkMacSystemFont, sans-serif;'>🎯 Summary</h3>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-family: -apple-system, BlinkMacSystemFont, sans-serif;'>{summary}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()