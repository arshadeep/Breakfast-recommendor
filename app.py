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
        endpoint_url="https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1",
        huggingfacehub_api_token=hugging_face_token,
        task="text-generation",
        model_kwargs={
            "temperature": 0.7,
            "max_new_tokens": 512,
            "top_p": 0.95,
            "return_full_text": False
        }
    )

def main():
    st.title("🍳 Smart Breakfast Recommendation App")
    st.markdown("""
    Get personalized breakfast recommendations based on your available ingredients and time!
    """)

    # Create two columns for inputs
    col1, col2 = st.columns(2)
    
    with col1:
        ingredients = st.text_area(
            "What ingredients do you have?",
            placeholder="e.g., eggs, bread, butter (separate with commas)"
        )
    
    with col2:
        time = st.number_input(
            "How much time do you have? (minutes)",
            min_value=5,
            max_value=60,
            value=15,
            step=5
        )

    # Define prompt template with nutrition calculation
    prompt_template = """You are a professional chef and nutritionist specializing in breakfast recipes. Create a recipe using ONLY the ingredients listed below and calculate its nutritional value.

Available ingredients: {ingredients}
Time limit: {time} minutes

Important rules:
- Use ONLY the ingredients listed above
- Do not suggest or mention any additional ingredients
- Recipe must be completable within the time limit
- Specify exact quantities for each ingredient
- Calculate nutritional information based on the specified quantities

Please provide the recipe in this exact format:

Recipe Name: (create a name using only the available ingredients)

Ingredients Used:
- (list each ingredient with exact measurements)

Instructions:
1. (step with timing)
2. (step with timing)
3. (step with timing)

Total Time: (must be less than or equal to the time limit)

Nutritional Information (per serving):
- Calories: (calculate based on ingredients and portions)
- Protein: (in grams)
- Carbohydrates: (in grams)
- Fat: (in grams)
- Fiber: (in grams)
- Key Vitamins/Minerals: (list main nutrients)

Tips: (optional tips for better results using ONLY the listed ingredients)

Remember: 
1. Calculate nutrition values based on the EXACT quantities specified in the recipe
2. Consider standard serving sizes when calculating nutrition
3. Break down the nutritional calculation if relevant
4. Stick STRICTLY to the provided ingredients"""

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
                with st.spinner("Creating your personalized breakfast recipe with nutritional analysis..."):
                    try:
                        # Get recipe recommendation with nutrition info
                        response = chain.invoke({"ingredients": str(ingredients), "time": str(time)})
                        
                        # Display response in a nice format
                        st.success("Here's your personalized breakfast recipe!")
                        
                        # Recipe card
                        with st.container():
                            st.markdown(response['text'])
                            
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
        Made with ❤️ using Streamlit and Mixtral-8x7B
        
        *Note: Please verify ingredients, cooking times, and nutritional information for food safety.*
        """
    )

if __name__ == "__main__":
    main()
