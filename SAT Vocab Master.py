import json
import time
import random
import os
import base64
import re
import io
import tempfile
import threading
from typing import List, Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor

import streamlit as st
from pydantic import BaseModel, Field

# --- EXTERNAL API IMPORTS ---
try:
    # Supabase Client Import (REMAINS)
    from supabase import create_client, Client as SupabaseClient
except ImportError:
    st.error("SUPABASE ERROR: The required library 'supabase-py' is likely missing. Please install it.")
    st.stop()

try:
    from gtts import gTTS
except ImportError:
    st.error("ERROR: The 'gtts' library is required for open-source TTS. Please install it.")
    st.stop()
    
try:
    from google import genai
    from google.genai import types
except ImportError:
    st.error("ERROR: The 'google-genai' and 'pydantic' libraries are required. Please install them.")
    st.stop()


# ======================================================================
# 1. CONFIGURATION & MODELS
# ======================================================================

# --- Supabase Credentials & Constants (REMAINS) ---
SUPABASE_URL: str = "https://hcmoeljjxlpcgoelyjqh.supabase.co"
SUPABASE_KEY: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImhjbW9lbGpqeGxwY2dvZWx5anFoIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM5NTI1ODUsImV4cCI6MjA3OTUyODU4NX0.YYDFnDMfn9UvRlJHs1nozVTe-qkOIY_habwixcCI6KM"
TABLE_NAME: str = "sat_vocabulary" 

# --- App State and Constants (UNCHANGED) ---
REQUIRED_WORD_COUNT = 2000 
LOAD_BATCH_SIZE = 10         
QUIZ_SIZE = 5 
AUTO_FETCH_THRESHOLD = 50 
AUTO_FETCH_BATCH = 25 
BRIEFING_BATCH_SIZE = 10 
MANUAL_BRIEFING_BATCH = 50 
MANUAL_EXTRACT_BATCH = 50 

# Admin Configuration (Mock Login)
ADMIN_EMAIL = "roy.jamshaid@gmail.com" 
ADMIN_PASSWORD = "Jamshaid,1981" 

# Pydantic Schema for Vocabulary Word (UNCHANGED)
class SatWord(BaseModel):
    word: str = Field(description="The SAT-level word.")
    pronunciation: str = Field(description="Simple, hyphenated phonetic pronunciation (e.g., eh-FEM-er-al).")
    definition: str = Field(description="The concise dictionary definition.")
    tip: str = Field(description="A short, catchy mnemonic memory tip.")
    usage: str = Field(description="A professional sample usage sentence.")
    sat_level: str = Field(default="High", description="Should always be 'High'.")
    audio_base64: Optional[str] = Field(default=None, description="Base64 encoded audio data for word pronunciation.")
    created_at: float = Field(default_factory=time.time)
    briefing_text: Optional[str] = Field(default=None, description="The extended AI-generated briefing text.")
    briefing_audio_base64: Optional[str] = Field(default=None, description="Base64 encoded audio data for the briefing.")

# ======================================================================
# 2. SETUP & INITIALIZATION (Database and AI Clients)
# ======================================================================

# --- DATABASE CLIENT INITIALIZATION (Supabase Implementation) ---
@st.cache_resource
def initialize_db_client() -> SupabaseClient:
    """Initializes the database client (Supabase)."""
    try:
        client: SupabaseClient = create_client(SUPABASE_URL, SUPABASE_KEY)
        client.from_(TABLE_NAME).select("word").limit(1).execute() 
        st.success("✅ Database client (Supabase) initialized and connected.")
        return client
    except Exception as e:
        st.error(f"🔴 DATABASE INITIALIZATION FAILED. Root Cause: {e}. Check your URL, Key, and table name ('{TABLE_NAME}').")
        st.stop()

try:
    # RENAMED VARIABLE
    db_client = initialize_db_client()
except Exception:
    st.stop()

# --- GEMINI CLIENT INITIALIZATION (UNCHANGED) ---
if "GEMINI_API_KEY" not in st.secrets:
    st.error("🔴 GEMINI_API_KEY is missing! Please set it in your Streamlit Secrets.")
    st.stop()

try:
    gemini_client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
except Exception as e:
    st.error(f"🔴 Failed to initialize Gemini Client: {e}")
    st.stop()


# ======================================================================
# 3. CORE UTILITIES & LAZY LOADING (Supabase Implementation)
# ======================================================================

def initialize_session_state():
    # ... (Initialization logic - unchanged) ...
    if 'current_user_email' not in st.session_state: st.session_state.current_user_email = None
    if 'is_auth' not in st.session_state: st.session_state.is_auth = False
    if 'vocab_data' not in st.session_state: st.session_state.vocab_data = None 
    if 'quiz_active' not in st.session_state: st.session_state.quiz_active = False
    if 'current_page_index' not in st.session_state: st.session_state.current_page_index = 0
    if 'quiz_start_index' not in st.session_state: st.session_state.quiz_start_index = 0
    if 'is_admin' not in st.session_state: st.session_state.is_admin = False
    if 'drill_word_index' not in st.session_state: st.session_state.drill_word_index = 0
    if 'data_refresh_key' not in st.session_state: st.session_state.data_refresh_key = 0
    if 'initial_load_done' not in st.session_state: st.session_state.initial_load_done = False
    
    # Task Management
    if 'autotask_running' not in st.session_state: st.session_state.autotask_running = False
    if 'autotask_message' not in st.session_state: st.session_state.autotask_message = None
    if 'autotask_status' not in st.session_state: st.session_state.autotask_status = 'Idle' 

    # LAZY LOADING STATE MANAGEMENT
    if 'has_more_data' not in st.session_state: st.session_state.has_more_data = True
    if 'total_word_count' not in st.session_state: st.session_state.total_word_count = 0


# RENAMED FUNCTION
def get_total_word_count() -> int:
    """Fetches the total document count using the database client."""
    try:
        # Supabase implementation of total count
        response = db_client.from_(TABLE_NAME).select("count", count="exact").limit(0).execute()
        return response.count if response and response.count is not None else 0
    except Exception as e:
        print(f"🔴 DB Count Failed: {e}")
        return len(st.session_state.vocab_data) if st.session_state.vocab_data else 0


# RENAMED FUNCTION
def fetch_vocabulary_batch(offset: int) -> List[Dict]:
    """Fetches the next batch of words using offset-based pagination."""
    start_index = offset
    end_index = offset + LOAD_BATCH_SIZE - 1
    
    try:
        # Supabase implementation of batch fetch
        response = (
            db_client.from_(TABLE_NAME)
            .select("*")
            .order('created_at', desc=False)
            .range(start_index, end_index)
            .execute()
        )
        return response.data
    except Exception as e:
        print(f"🔴 DB Batch Load Failed: {e}")
        return []

def load_and_update_vocabulary_data():
    """Loads the INITIAL batch of data and calculates the total count."""
    if not st.session_state.is_auth or st.session_state.vocab_data is not None: return

    # 1. Fetch Total Count (RENAMED CALL)
    st.session_state.total_word_count = get_total_word_count()
    
    # 2. Fetch Initial Batch (RENAMED CALL)
    vocab_list = fetch_vocabulary_batch(offset=0)
    
    # 3. Update State
    st.session_state.vocab_data = vocab_list
    st.session_state.initial_load_done = True
    
    if vocab_list:
        st.session_state.has_more_data = len(vocab_list) == LOAD_BATCH_SIZE
    else:
        st.session_state.has_more_data = False
    
    if len(vocab_list) > 0:
        st.info(f"✅ Loaded initial {len(vocab_list)} words from DB (Total: {st.session_state.total_word_count}).")
    elif st.session_state.is_auth:
        st.info(f"Database is empty. Total count: {st.session_state.total_word_count}.")


def fetch_and_append_next_batch():
    """Fetches the next batch and appends it to session state."""
    if not st.session_state.has_more_data:
        st.warning("No more data to load from the database.")
        return

    offset = len(st.session_state.vocab_data)
    # RENAMED CALL
    next_batch = fetch_vocabulary_batch(offset=offset)
    
    if next_batch:
        st.session_state.vocab_data.extend(next_batch)
        st.session_state.has_more_data = len(next_batch) == LOAD_BATCH_SIZE
        # RENAMED CALL
        st.session_state.total_word_count = get_total_word_count() 
        st.success(f"Loaded {len(next_batch)} more words.")
    else:
        st.session_state.has_more_data = False
        st.info("Reached the end of the vocabulary list.")
        
    st.rerun()

# --- Pagination Logic (UNCHANGED) ---
def go_to_next_page():
    # ... (Logic unchanged) ...
    total_loaded = len(st.session_state.vocab_data)
    max_index = (total_loaded // LOAD_BATCH_SIZE) - 1
    current_index = st.session_state.current_page_index
    
    if current_index == max_index and st.session_state.has_more_data:
        fetch_and_append_next_batch() 
    
    st.session_state.current_page_index += 1
    st.rerun()

def go_to_prev_page():
    st.session_state.current_page_index -= 1
    st.rerun()

# --- Database Write Operations ---

# RENAMED FUNCTION
def save_word_to_db(word_data: Dict) -> bool:
    """Adds a single word document to the database."""
    try:
        # Supabase implementation of save
        db_client.from_(TABLE_NAME).insert(word_data).execute()
        return True
    except Exception as e:
        print(f"🔴 DB Save Failed for {word_data['word']}: {e}")
        return False
        
# RENAMED FUNCTION
def update_word_in_db(word_data: Dict, fields_to_update: Dict) -> bool:
    """Updates specific fields of a word document in the database by word name."""
    try:
        # Supabase implementation of update
        db_client.from_(TABLE_NAME).update(fields_to_update).eq('word', word_data['word']).execute()
        return True
    except Exception as e:
        print(f"🔴 DB Update Failed for {word_data['word']}: {e}")
        return False

# --- Core Utilities (UNCHANGED) ---
def generate_tts_audio(text: str) -> Optional[str]:
    # ... (TTS Generation - unchanged) ...
    if not text: return None
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        base64_data = base64.b64encode(mp3_fp.read()).decode('utf-8')
        return base64_data
    except Exception as e:
        print(f"🔴 gTTS Generation failed for text segment '{text[:20]}...'. Error: {e}")
        return None

def generate_full_briefing_content(word_data: Dict) -> Optional[Dict]:
    # ... (Briefing LLM/TTS Generation - unchanged) ...
    word = word_data.get('word', 'a high-level word')
    definition = word_data.get('definition', 'a complex meaning')
    
    prompt = f"""
    You are a vocabulary tutor. Write a **short, memorable, and concise briefing (5-6 sentences maximum, about 60-80 words)** on the word '{word}'. 
    The briefing must seamlessly include: 1. The core definition: {definition}. 2. A brief note on its origin or etymology (1 sentence). 3. One compelling example sentence demonstrating high-level usage. 4. A final, memorable takeaway.
    Ensure the entire text is conversational and suitable for speech synthesis. Do not use bullet points or lists; write it as a continuous, flowing speech.
    """
    
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.7) 
        )
        briefing_text = response.text.strip()
        audio_data = generate_tts_audio(briefing_text)
        
        if not audio_data: return None 

        return {
            "briefing_text": briefing_text,
            "briefing_audio_base64": audio_data 
        }
        
    except Exception as e:
        print(f"🔴 Gemini/Briefing Generation Failed for '{word}': {e}")
        return None


# ======================================================================
# 4. ASYNCHRONOUS TASK CONTROLLER (Uses Renamed DB Functions)
# ======================================================================

class LongRunningTaskController:
    # ... (Logic unchanged) ...
    def __init__(self):
        if 'task_thread' not in st.session_state:
            st.session_state.task_thread = None

    def _update_session_state(self, status: str, message: str, running: bool):
        st.session_state.autotask_status = status
        st.session_state.autotask_message = message
        st.session_state.autotask_running = running
        st.rerun()

    def run_task_in_thread(self, target_function, *args, **kwargs):
        if st.session_state.autotask_running: return False 
        st.session_state.task_thread = threading.Thread(
            target=target_function, args=args, kwargs=kwargs, daemon=True
        )
        self._update_session_state('Running', 'Task initiated...', True)
        st.session_state.task_thread.start()
        return True

    def check_task_status(self):
        if st.session_state.autotask_running and st.session_state.task_thread and not st.session_state.task_thread.is_alive():
            
            # RENAMED CALL
            st.session_state.total_word_count = get_total_word_count()
            st.session_state.vocab_data = None 
            st.session_state.has_more_data = True 
            
            self._update_session_state('Complete', st.session_state.autotask_message or 'Task complete. Reloading data.', False)
            st.rerun() 
        elif st.session_state.autotask_running:
             st.rerun()

    # --- THREAD TARGET FUNCTIONS ---
    def _extract_and_save_batch(self, num_words: int, existing_words: List[str], auto_fetch: bool):
        try:
            # ... (LLM Extraction) ...
            st.session_state.autotask_message = f"LLM Task: Generating {num_words} structured words..."
            prompt = f"Generate {num_words} unique, extremely high-level SAT vocabulary words. The words must NOT be any of the following: {', '.join(existing_words) if existing_words else 'none'}."
            list_schema = {"type": "array", "items": SatWord.model_json_schema()}
            config = types.GenerateContentConfig(response_mime_type="application/json", response_json_schema=list_schema)
            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash", contents=prompt, config=config
            )
            new_data_list = json.loads(response.text)
            validated_words = [SatWord(**item).model_dump() for item in new_data_list if 'word' in item]

            successful_saves = 0
            
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_word = {executor.submit(self._enrich_word, word_data): word_data for word_data in validated_words}
                enriched_words = [future.result() for future in future_to_word]

            st.session_state.autotask_message = f"Saving {len(enriched_words)} words to DB..."
            
            for word_data in enriched_words:
                # RENAMED CALL
                if save_word_to_db(word_data):
                    successful_saves += 1
            
            st.session_state.autotask_message = f"✅ Extracted and saved {successful_saves} words."
            
        except Exception as e:
            st.session_state.autotask_status = 'Error'
            st.session_state.autotask_message = f"🔴 Extraction Failed: {e}"
        finally:
            st.session_state.autotask_running = False
            
    def _enrich_word(self, word_data: Dict) -> Dict:
        # ... (Helper for word enrichment - unchanged) ...
        pronunciation_audio = generate_tts_audio(word_data['word'])
        word_data['audio_base64'] = pronunciation_audio if pronunciation_audio else None
        
        briefing_content = generate_full_briefing_content(word_data)
        if briefing_content:
            word_data.update(briefing_content)
        return word_data
        
    def _generate_briefing_batch(self, batch_indices: List[int], batch_size: int):
        try:
            generated_count = 0
            words_to_process = [st.session_state.vocab_data[i] for i in batch_indices]

            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_word = {executor.submit(self._enrich_briefing, word_data): word_data for word_data in words_to_process}

                for future in future_to_word:
                    result = future.result()
                    if result:
                        word_data = future_to_word[future]
                        # RENAMED CALL
                        if update_word_in_db(word_data, result):
                             st.session_state.vocab_data[st.session_state.vocab_data.index(word_data)].update(result)
                             generated_count += 1

            remaining = len([d for d in st.session_state.vocab_data if not d.get('briefing_audio_base64')])
            
            if remaining > 0:
                 st.session_state.autotask_message = f"✅ Auto-Briefing completed a batch of {generated_count}. {remaining} remaining. Processing next LEGACY batch..."
            else:
                 st.session_state.autotask_message = f"✅ Auto-Briefing complete: All words now have briefings."
                 
        except Exception as e:
            st.session_state.autotask_status = 'Error'
            st.session_state.autotask_message = f"🔴 Briefing Generation Failed: {e}"
        finally:
            st.session_state.autotask_running = False

    def _enrich_briefing(self, word_data: Dict) -> Optional[Dict]:
        # ... (Helper for briefing enrichment - unchanged) ...
        briefing_content = generate_full_briefing_content(word_data)
        if briefing_content:
            return briefing_content
        return None

task_controller = LongRunningTaskController()


# ======================================================================
# 5. HANDLERS (Adjusted for DB Abstraction)
# ======================================================================

# ... (All other handlers and UI components use the new DB-agnostic names) ...

def handle_manual_word_entry(word: str):
    """Generates all content for a single word and saves it to the database."""
    if not word or st.session_state.autotask_running: st.error("Please enter a word or wait for the current task to finish."); return
        
    st.info(f"Generating content for '{word}'...")
    
    try:
        # 1. Get Base Word Data via LLM (UNCHANGED)
        prompt = f"Generate the pronunciation, definition, mnemonic tip, and a usage sentence for the high-level SAT word: {word}. Return only the JSON object."
        list_schema = {"type": "array", "items": SatWord.model_json_schema()}
        config = types.GenerateContentConfig(response_mime_type="application/json", response_json_schema=list_schema)
        response = gemini_client.models.generate_content(model="gemini-2.5-flash", contents=prompt, config=config)
        data_list = json.loads(response.text)
        new_word_data = SatWord(**data_list[0]).model_dump()
        
        # 2. Enrich: Pronunciation & Briefing (UNCHANGED)
        new_word_data = task_controller._enrich_word(new_word_data)
        
    except Exception as e:
        st.error(f"🔴 Failed to generate content for '{word}'. Error: {e}"); return

    # RENAMED CALL
    if save_word_to_db(new_word_data):
        st.session_state.vocab_data = None 
        st.session_state.last_word_timestamp = None
        st.session_state.has_more_data = True
        st.session_state.total_word_count = 0
        st.success(f"✅ Successfully added '{new_word_data['word']}' with ALL content to DB! Reloading data...")
        st.rerun()
    else:
        st.error("🔴 Failed to save to DB.")
        
# ... (Other UI and Handler functions omitted for brevity, but they would use 
# the renamed functions like `get_total_word_count`, `save_word_to_db`, etc.)
# The main() function and all UI rendering logic remain structurally identical.

# ... (End of script)
