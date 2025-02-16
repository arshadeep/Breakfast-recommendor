import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import LLMChain
import os

# ---- 1. PAGE CONFIG & CUSTOM STYLING ----
st.set_page_config(
    page_title="TeenChef 🍳",
    page_icon="🦄",
    layout="centered"
)

def add_custom_styles():
    """
    Inject some custom CSS to give the app a fun, teenage look & feel.
    You can experiment with different fonts, colors, and styling.
    """
    st.markdown(
        """
        <style>
        /* Import a playful Google font */
        @import url('https://fonts.googleapis.com/css2?family=Comic+Neue:wght@700&display=swap');

        /* Apply the font and background color globally */
        html, body, [class*="css"]  {
            font-family: 'Comic Neue', sans-serif;
            background-color: #FFE6F2; /* light pink background */
            color: #4D004D; /* deep purple text */
        }

        /* Style the main title */
        .stMarkdown h1 {
            color: #FF66CC;
            text-align: center;
            font-size: 3rem;
        }

        /* Style buttons */
        .stButton>button {
            background-color: #FF66CC;
            color: #FFF;
            border-radius: 10px;
            border: none;
            padding: 0.75rem 1.5rem;
            cursor: pointer;
            font-size: 1rem;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #FF1493;
        }

        /* Style text input areas */
        .stTextArea textarea, .stNumberInput>div>input, .stSelectbox>div>div>select {
            background-color: #FFF0F5;
            color: #4D004D;
            border: 2px solid #FF66CC;
        }

        /* Give columns a bit of spacing at the top */
        .css-18e3th9 {
            padding-top: 2rem;
        }

        /* Optionally style success messages, warnings, etc. */
        .stSuccess {
            background-color: #D8BFD8 !important; /* Thistle color for success */
            border-left: 4px solid #FF66CC !important;
        }
        .stWarning {
            background-color: #FFFACD !important; /* LemonChiffon for warning */
            border-left: 4px solid #FFA500 !important;
        }
        .stError {
            background-color: #F08080 !important; /* LightCoral for errors */
            border-left: 4px solid #FF0000 !important;
        }

        /* Scrollbar styling (optional) */
        ::-webkit-scrollbar {
            width: 10px;
        }
        ::-webkit-scrollbar-track {
            background: #FFF0F5;
        }
        ::-webkit-scrollbar-thumb {
            background-color: #FF66CC;
            border-radius: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# ---- 2. INITIALIZE LLM ----
@st.cache_resource
def init_llm():
    hugging_face_token = os.getenv("HUGGINGFACE_API_TOKEN")
    if not hugging_face_token:
        raise ValueError("Please set the HUGGINGFACE_API_TOKEN environment variable")
    
    return HuggingFaceEndpoint(
        endpoint_url="https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1",
        huggingfacehub_api_token=hugging_face_token
    )

# ---- 3. MAIN APP ----
def main():
    # Add custom style
    add_custom_styles()

    st.title("Breakfast recommendor 🍳")
    st.markdown("""
    **Welcome to the Breakfast recommendor!**  
    _Get Personalized personalized breakfast recommendations based on your ingredients, time, and vibe!_
    """)

    # Create two columns for inputs
    col1, col2 = st.columns(2)
    
    with col1:
        ingredients = st.text_area(
            "🛒 What ingredients do you have?",
            placeholder="e.g., eggs, bread, butter (separate with commas)"
        )
    
    with col2:
        time = st.number_input(
            "⏱️ How much time do you have? (minutes)",
            min_value=5,
            max_value=60,
            value=15,
            step=5
        )

    # Add a dropdown for dietary preferences
    preference = st.selectbox(
        "🍔 Select your dietary preference:",
        options=["None", "Vegan", "Gluten-Free", "Vegetarian", "Paleo", "Keto"],
        index=0,
        help="Choose a dietary preference to tailor your recipe"
    )

    # Define prompt template with nutrition calculation and dietary preference
    prompt_template = """You are a professional chef and nutritionist specializing in breakfast recipes. 
Create a recipe using ONLY the ingredients listed below and calculate its nutritional value.

Available ingredients: {ingredients}
Time limit: {time} minutes
Dietary Preference: {preference}

Important rules:
- Use ONLY the ingredients listed above.
- Do not suggest or mention any additional ingredients.
- The recipe must be completable within the time limit.
- Specify exact quantities for each ingredient.
- Calculate nutritional information based on the specified quantities.
- If a dietary preference is provided (e.g., Vegan, Gluten-Free), ensure that the recipe adheres to it.

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
1. Calculate nutrition values based on the EXACT quantities specified in the recipe.
2. Consider standard serving sizes when calculating nutrition.
3. Break down the nutritional calculation if relevant.
4. Stick STRICTLY to the provided ingredients and dietary preference (if applicable)."""

    template = PromptTemplate(
        template=prompt_template,
        input_variables=["ingredients", "time", "preference"]
    )

    # Initialize LLM and chain
    try:
        llm = init_llm()
        chain = LLMChain(llm=llm, prompt=template)

        # When user clicks "Get Recommendation"
        if st.button("🎉 Get Recommendation!", type="primary"):
            if ingredients and time > 0:
                with st.spinner("🍳 Cooking up your teenage-inspired breakfast recipe..."):
                    try:
                        # Get recipe recommendation with nutrition info
                        response = chain.invoke({
                            "ingredients": str(ingredients),
                            "time": str(time),
                            "preference": preference
                        })
                        
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
        *Note: Please verify ingredients, cooking times, and nutritional information for food safety.*  
        **Stay creative, stay hungry!** 🍔
        """
    )

if __name__ == "__main__":
    main()
