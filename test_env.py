from dotenv import load_dotenv
import os

load_dotenv()
print(f"LM_STUDIO_API_URL = {os.getenv('LM_STUDIO_API_URL', 'NOT SET')}")
print(f"Current working directory = {os.getcwd()}")
