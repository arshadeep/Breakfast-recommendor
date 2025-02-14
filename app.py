import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import HuggingFaceHub
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
    
    return HuggingFaceHub(
        repo_id="facebook/opt-1.3b",  # Changed to a more reliable model
        huggingfacehub_api_token=hugging_face_token,
        model_kwargs={
            "temperature": 0.7,
            "max_new_tokens": 512,
            "top_p": 0.9,
            "repetition_penalty": 1.1
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
    prompt_template = """Create a breakfast recipe using these ingredients: {ingredients}. The recipe must take {time} minutes or less to prepare.

Please format the response exactly like this:
Recipe Name: [short descriptive name]
Ingredients Needed: [list only the ingredients that were provided]
Steps to Prepare: [numbered steps, be concise]
Approximate Calories: [estimate]
Nutritional Highlights: [brief nutritional benefits]

Keep the recipe simple and practical."""

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
                        response = chain.run(ingredients=ingredients, time=time)
                        
                        # Display response in a nice format
                        st.success("Here's your personalized breakfast recipe!")
                        st.markdown(response)
                    except Exception as e:
                        st.error("Sorry, couldn't generate a recipe at the moment. Please try again.")
                        st.error(f"Error details: {str(e)}")
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
