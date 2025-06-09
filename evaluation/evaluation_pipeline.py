from dotenv import load_dotenv
import os

load_dotenv() # This will load environment variables from a .env file into the environment

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")