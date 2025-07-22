import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI  # <-- 1. Import Gemini
from langchain_core.messages import HumanMessage
import argparse
from pathlib import Path
import base64

# Load the API key from the .env file
load_dotenv() # <-- 2. Load environment variables

# This function is unchanged
def encode_image(image_path):
    with open(image_path, "rb") as fridge_image:
        return base64.b64encode(fridge_image.read()).decode('utf-8')


def analyze_fridge(image_path):
    # <-- 3. Instantiate the Gemini model
    # We use a vision model like gemini-1.5-flash-latest
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
    
    encoded_image = encode_image(image_path)
    
    prompt_text = """You are a helpful AI assistant that analyzes
the contents of the insides of a fridge. Please analyze this image and:
1. list all the visible food items
2. Give healthy and tasty recipes which can be made from the items given"""

    # <-- 4. The message structure is identical! No changes needed here.
    # LangChain abstracts this so you can swap models easily.
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": prompt_text,
            },
            {
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{encoded_image}"
            },
        ]
    )

    # The invocation is also the same
    response = llm.invoke([message])
    return response.content


def main():
    parser = argparse.ArgumentParser(description="Analyze contents in fridge using Gemini")
    parser.add_argument(
        "--image",
        type=str,
        default="/Users/nebulo/Study-Space/JUPYTER_STUFF/fridge.jpg",
        help="Path to fridge image"
    )
    
    args = parser.parse_args()
    image_path = Path(args.image)
    
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY environment variable not set.")
        print("Please create a .env file with your API key.")
        return

    try:
        result = analyze_fridge(image_path)
        print("\nFridge Analysis Results:")
        print(result)
        
    except Exception as e:
        print(f"Error analyzing image: {str(e)}")


if __name__ == "__main__":
    main()