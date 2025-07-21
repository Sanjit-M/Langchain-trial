from dotenv import load_dotenv
import getpass
import os

load_dotenv()

if not os.environ.get("GOOGLE_API_KEY"):
  os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

from langchain.chat_models import init_chat_model

model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

from langchain_core.prompts import ChatPromptTemplate

system_template = "Translate the following from English into French: "

prompt_template = ChatPromptTemplate.from_messages(
  [("system", system_template), ("user", "{text}")]
)

prompt = prompt_template.invoke({"text": "Hello, how are you?"})

response = model.invoke(prompt)
print(response.content)