import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import LLMChain
import os

# Page configuration
st.set_page_config(
    page_title="Breakfast Recommendation App",
    page_icon="🍳",
    layout="centered"
)

# Initialize HuggingFace
@st.cache_resource
def init_llm():
    hugging_face_token = os.getenv("HUGGINGFACE_API_TOKEN")
    if not hugging_face_token:
        st.error("Please set the HUGGINGFACE_API_TOKEN environment variable")
        st.stop()
    
    return HuggingFaceEndpoint(
        endpoint_url="https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct",
        huggingfacehub_api_token=hugging_face_token,
        task="text-generation",
        model_kwargs={
            "temperature": 0.7,
            "max_new_tokens": 256,
            "return_full_text": False
        }
    )

def main():
    st.title("🍳 Breakfast Recommendation App")
    st.markdown("""
    Get personalized breakfast recommendations based on your available ingredients and time!
    """)

    # Create two columns for inputs
    col1, col2 = st.columns(2)
    
    with col1:
        ingredients = st.text_area(
            "What ingredients do you have?",
            placeholder="e.g., eggs, bread, milk, butter"
        )
    
    with col2:
        time = st.number_input(
            "How much time do you have? (minutes)",
            min_value=5,
            max_value=60,
            value=15,
            step=5
        )

    # Define prompt template
    prompt_template = """You are a helpful chef. Create a breakfast recipe using only these ingredients: {ingredients}. The recipe must take {time} minutes or less to prepare.

Recipe format:
Name: (name of the dish)
Time: (cooking time in minutes)
Steps:
1. (step)
2. (step)
3. (step)
Nutrition: (brief nutritional info)"""

    template = PromptTemplate(
        template=prompt_template,
        input_variables=["ingredients", "time"]
    )

    # Initialize LLM and chain
    try:
        llm = init_llm()
        chain = LLMChain(llm=llm, prompt=template)

        # When user clicks "Get Recommendation"
        if st.button("Get Recommendation", type="primary"):
            if ingredients and time > 0:
                with st.spinner("Creating your breakfast recipe..."):
                    try:
                        # Convert inputs to strings
                        response = chain.invoke({"ingredients": str(ingredients), "time": str(time)})
                        
                        # Display response in a nice format
                        st.success("Here's your personalized breakfast recipe!")
                        st.markdown(response['text'])
                    except Exception as e:
                        st.error("Sorry, couldn't generate a recipe at the moment. Please try again.")
                        st.error(f"Error details: {str(e)}")
                        st.info("If the error persists, try refreshing the page or using different ingredients.")
            else:
                st.warning("Please provide both ingredients and time.")

    except Exception as e:
        st.error("Error connecting to the recipe service. Please try again later.")
        st.error(f"Error details: {str(e)}")

    # Add footer with disclaimer
    st.markdown("---")
    st.markdown(
        """
        Made with ❤️ using Streamlit and HuggingFace
        
        *Note: Please verify ingredients and cooking times for food safety.*
        """
    )

if __name__ == "__main__":
    main()
