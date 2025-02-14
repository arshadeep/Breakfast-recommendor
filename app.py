# app.py

import streamlit as st
#from langchain.llms import Ollama
from langchain import PromptTemplate, LLMChain
import os
from langchain.llms import HuggingFaceHub  # Free option
hugging_face_token = os.getenv("HUGGINGFACE_API_TOKEN")
# Modify your code to use HuggingFace:
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=hugging_face_token  # Sign up at HuggingFace
)

def main():
    st.title("Breakfast Recommendation App")

    # Collect user inputs
    ingredients = st.text_input("Ingredients you have:")
    time = st.number_input("Time available (in minutes):", min_value=1, step=1)

    # Define prompt template
    prompt_template = """
    You are an expert nutritionist.
    Given the following inputs, suggest a breakfast recipe:
    - Ingredients: {ingredients}
    - Time Available: {time} minutes

    Constraints:
    1. Must use the provided ingredients when possible.
    2. Offer a quick summary of preparation steps within the provided time.
    3. Provide an approximate calorie count and any notable nutritional benefits.

    Output Format:
    Recipe Name:
    Ingredients Needed:
    Steps to Prepare:
    Approximate Calories:
    Nutritional Highlights:
    """

    template = PromptTemplate(
        template=prompt_template,
        input_variables=["ingredients", "time"]
    )

    # Initialize Ollama LLM (make sure Ollama server is running: ollama serve)
    #ollama_llm = Ollama(
     #   model="mistral-nemo"  # or "mistral", whichever you have pulled
        # You can also pass extra params (e.g. f16_kv=True, num_gpu=1, etc.) if desired
    #)

    # Create the LangChain LLMChain
    # Option A: Using classic approach
    chain = LLMChain(llm=llm, prompt=template)

    # Option B: If you have a newer LangChain and want to try pipe syntax:
    # chain = template | ollama_llm

    # When user clicks "Get Recommendation"
    if st.button("Get Recommendation"):
        if ingredients and time > 0:
            response = chain.run({"ingredients": ingredients, "time": time})
            st.write(response)
        else:
            st.warning("Please provide both ingredients and a valid time.")

if __name__ == "__main__":
    main()
