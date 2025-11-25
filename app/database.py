import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

# Fetch variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize Supabase Client
supabase: Client = None

if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("Supabase Client initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize Supabase Client: {e}")
else:
    print("Warning: SUPABASE_URL or SUPABASE_KEY not found in environment variables.")

def get_supabase_client() -> Client:
    return supabase
