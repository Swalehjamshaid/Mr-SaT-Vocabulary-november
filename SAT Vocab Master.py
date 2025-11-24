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
import pandas as pd # Required for reading SQL results
import sqlalchemy # Required by st.connection('sql')

# --- EXTERNAL API IMPORTS ---
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

# --- Database Constants ---
TABLE_NAME: str = "sat_vocabulary" 

# --- App State and Constants ---
REQUIRED_WORD_COUNT = 2000 
LOAD_BATCH_SIZE = 10         # Fetch and display 10 words at a time
QUIZ_SIZE = 5 
AUTO_FETCH_THRESHOLD = 50 
AUTO_FETCH_BATCH = 25        # Words to fetch in background auto-task
BRIEFING_BATCH_SIZE = 10 
MANUAL_BRIEFING_BATCH = 50 
MANUAL_EXTRACT_BATCH = 50    # Words to fetch in foreground manual task

# Admin Configuration (Mock Login)
ADMIN_EMAIL = "roy.jamshaid@gmail.com" 
ADMIN_PASSWORD = "Jamshaid,1981" 

# Pydantic Schema for Vocabulary Word
class SatWord(BaseModel):
    word: str = Field(description="The SAT-level word.")
    pronunciation: str = Field(description="Simple, hyphenated phonetic pronunciation (e.g., eh-FEM-er-al).")
    definition: str = Field(description="The concise dictionary definition.")
    tip: str = Field(description="A short, catchy mnemonic memory tip.")
    usage: str = Field(description="A professional sample usage sentence.")
    sat_level: str = Field(default="High", description="Should always be 'High' in English.")
    audio_base64: Optional[str] = Field(default=None, description="Base64 encoded audio data for word pronunciation.")
    created_at: float = Field(default_factory=time.time)
    briefing_text: Optional[str] = Field(default=None, description="The extended AI-generated briefing text in English.")
    briefing_audio_base64: Optional[str] = Field(default=None, description="Base64 encoded audio data for the briefing.")

# ======================================================================
# 2. SETUP & INITIALIZATION (Neon/PostgreSQL Client)
# ======================================================================

@st.cache_resource
def initialize_db_connection():
    """Initializes and returns the Streamlit SQL connection object."""
    try:
        # Uses the URL from [connections.neon_db] in secrets.toml
        conn = st.connection("neon_db", type="sql")
        
        # Test connectivity and table existence
        conn.query(f"SELECT 'success' FROM {TABLE_NAME} LIMIT 1;", ttl=0)
        
        st.success("✅ Database connection (Neon/PostgreSQL) initialized and table found.")
        return conn
    
    except Exception as e:
        if "relation" in str(e) and "does not exist" in str(e):
            st.error(f"🔴 DATABASE TABLE MISSING: Table '{TABLE_NAME}' does not exist in Neon.")
            st.warning("ACTION: Please create the table in your Neon dashboard.")
            st.stop()
        else:
            st.error(f"🔴 DATABASE CONNECTION FAILED. Root Cause: {e}.")
            st.stop()

# Global database connection object
try:
    db_conn = initialize_db_connection()
except SystemExit:
    db_conn = None 
    st.stop()

# --- GEMINI CLIENT INITIALIZATION ---
if "GEMINI_API_KEY" not in st.secrets:
    st.error("🔴 GEMINI_API_KEY is missing! Please set it in your Streamlit Secrets.")
    st.stop()

try:
    gemini_client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
except Exception as e:
    st.error(f"🔴 Failed to initialize Gemini Client: {e}")
    st.stop()


# ======================================================================
# 3. CORE UTILITIES & LAZY LOADING (SQL Implementation)
# ======================================================================

def initialize_session_state():
    """Sets up default session state variables."""
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
    
    # CRITICAL: Flag to prevent auto-task from firing on the immediate run after login
    if 'initial_auth_rerun_done' not in st.session_state: st.session_state.initial_auth_rerun_done = False 

    # LAZY LOADING STATE MANAGEMENT
    if 'has_more_data' not in st.session_state: st.session_state.has_more_data = True
    if 'total_word_count' not in st.session_state: st.session_state.total_word_count = 0
    # ADDED: Flag to ensure auto-fetch runs only once per session if needed
    if 'auto_fetch_triggered' not in st.session_state: st.session_state.auto_fetch_triggered = False


def get_total_word_count() -> int:
    """Fetches the total document count using SQL."""
    try:
        result = db_conn.query(f"SELECT COUNT(*) FROM {TABLE_NAME};", ttl=0)
        return int(result.iloc[0, 0])
    except Exception:
        return 0

def fetch_vocabulary_batch(offset: int) -> List[Dict]:
    """Fetches the next batch of words using offset-based SQL pagination."""
    start_index = offset
    
    try:
        sql_query = f"""
            SELECT * FROM {TABLE_NAME}
            ORDER BY created_at ASC
            LIMIT {LOAD_BATCH_SIZE}
            OFFSET {offset};
        """
        df = db_conn.query(sql_query, ttl=600)
        
        return df.to_dict('records')
    except Exception as e:
        print(f"🔴 DB Batch Load Failed: {e}")
        return []

def load_and_update_vocabulary_data():
    """Loads the INITIAL batch of data and calculates the total count."""
    # We only load data if authenticated AND if data is not already loaded
    if not st.session_state.is_auth or st.session_state.vocab_data is not None: return

    st.session_state.total_word_count = get_total_word_count()
    vocab_list = fetch_vocabulary_batch(offset=0)
    
    st.session_state.vocab_data = vocab_list
    st.session_state.initial_load_done = True
    
    if vocab_list:
        st.session_state.has_more_data = len(vocab_list) == LOAD_BATCH_SIZE
    else:
        st.session_state.has_more_data = False
    


def fetch_and_append_next_batch():
    """Fetches the next batch and appends it to session state."""
    if not st.session_state.has_more_data:
        st.warning("No more data to load from the database.")
        return

    offset = len(st.session_state.vocab_data)
    next_batch = fetch_vocabulary_batch(offset=offset)
    
    if next_batch:
        st.session_state.vocab_data.extend(next_batch)
        st.session_state.has_more_data = len(next_batch) == LOAD_BATCH_SIZE
        st.session_state.total_word_count = get_total_word_count() 
        st.success(f"Loaded {len(next_batch)} more words.")
    else:
        st.session_state.has_more_data = False
        st.info("Reached the end of the vocabulary list.")
        
    st.rerun()

# --- Pagination Logic (UNCHANGED) ---
def go_to_next_page():
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

# --- Database Write Operations (SQL Implementation) ---

def save_word_to_db(word_data: Dict) -> bool:
    """Adds a single word document to the database using SQL."""
    try:
        columns = ', '.join(word_data.keys())
        values_placeholders = ', '.join([f':{key}' for key in word_data.keys()])
        
        sql_insert = f"""
            INSERT INTO {TABLE_NAME} ({columns})
            VALUES ({values_placeholders});
        """
        with db_conn.session as s:
            s.execute(sql_insert, params=word_data)
            s.commit()
        return True
    except Exception as e:
        print(f"🔴 DB Save Failed for {word_data['word']}: {e}")
        return False
        
def update_word_in_db(word_data: Dict, fields_to_update: Dict) -> bool:
    """Updates specific fields of a word document using SQL."""
    try:
        with db_conn.session as s:
            set_clauses = [f"{key} = :{key}" for key in fields_to_update.keys()]
            params = {**fields_to_update, 'word_name': word_data['word']}
            
            sql_update = f"""
                UPDATE {TABLE_NAME}
                SET {', '.join(set_clauses)}
                WHERE word = :word_name;
            """
            s.execute(sql_update, params=params)
            s.commit()
        return True
    except Exception as e:
        print(f"🔴 DB Update Failed for {word_data['word']}: {e}")
        return False

# --- Core Utilities (UNCHANGED) ---
def generate_tts_audio(text: str) -> Optional[str]:
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
# 4. ASYNCHRONOUS TASK CONTROLLER
# ======================================================================

class LongRunningTaskController:
    """Manages all long-running AI and I/O tasks in a separate thread."""
    
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
            
            st.session_state.total_word_count = get_total_word_count()
            st.session_state.vocab_data = None 
            st.session_state.has_more_data = True 
            
            self._update_session_state('Complete', st.session_state.autotask_message or 'Task complete. Reloading data.', False)
            st.rerun() 
        elif st.session_state.autotask_running:
             st.rerun()

    # --- THREAD TARGET FUNCTIONS ---
    def _extract_and_save_batch(self, num_words: int, existing_words: List[str], auto_fetch: bool):
        """Generates structured word data, enriches it with audio/briefing, and saves to DB."""
        try:
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
            
            # --- Enrich data with audio and briefing (multi-threaded) ---
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_word = {executor.submit(self._enrich_word, word_data): word_data for word_data in validated_words}
                enriched_words = [future.result() for future in future_to_word]

            st.session_state.autotask_message = f"Saving {len(enriched_words)} words to DB..."
            
            for word_data in enriched_words:
                if save_word_to_db(word_data):
                    successful_saves += 1
            
            st.session_state.autotask_message = f"✅ Extracted and saved {successful_saves} words."
            
        except Exception as e:
            st.session_state.autotask_status = 'Error'
            st.session_state.autotask_message = f"🔴 Extraction Failed: {e}"
        finally:
            st.session_state.autotask_running = False
            
    def _enrich_word(self, word_data: Dict) -> Dict:
        """Helper to generate audio and briefing content for a single word."""
        # 1. Pronunciation Audio
        pronunciation_audio = generate_tts_audio(word_data['word'])
        word_data['audio_base64'] = pronunciation_audio if pronunciation_audio else None
        
        # 2. 2-Minute Briefing Content
        briefing_content = generate_full_briefing_content(word_data)
        if briefing_content:
            word_data.update(briefing_content)
        return word_data
        
    def _generate_briefing_batch(self, batch_indices: List[int], batch_size: int):
        """Processes a batch of existing words to add missing briefing content."""
        try:
            generated_count = 0
            words_to_process = [st.session_state.vocab_data[i] for i in batch_indices]

            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_word = {executor.submit(self._enrich_briefing, word_data): word_data for word_data in words_to_process}

                for future in future_to_word:
                    result = future.result()
                    if result:
                        word_data = future_to_word[future]
                        # Update only the briefing fields in the database
                        if update_word_in_db(word_data, result):
                             # Update session state with new data
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
        briefing_content = generate_full_briefing_content(word_data)
        if briefing_content:
            return briefing_content
        return None

task_controller = LongRunningTaskController()


# ======================================================================
# 5. HANDLERS (Adjusted for DB Abstraction)
# ======================================================================

def handle_admin_extraction_button(num_words: int, auto_fetch: bool = False):
    """Triggers the bulk word extraction in a background thread."""
    if st.session_state.autotask_running:
        st.warning("A background task is already running. Please wait.")
        return

    # Pass existing words to LLM to avoid generating duplicates
    existing_words = [d['word'] for d in st.session_state.vocab_data if st.session_state.vocab_data]
    
    # Run extraction and saving in a separate thread
    if task_controller.run_task_in_thread(
        task_controller._extract_and_save_batch, 
        num_words=num_words, 
        existing_words=existing_words, 
        auto_fetch=auto_fetch
    ):
        st.session_state.autotask_message = f"Initiated extraction of {num_words} words..."

def handle_manual_word_entry(word: str):
    """Generates all content for a single word and saves it to the database."""
    if st.session_state.autotask_running:
        st.error("A background task is running. Please wait.")
        return
    if not word: 
        st.error("Please enter a word."); 
        return
        
    st.info(f"Generating content for '{word}'...")
    
    try:
        # LLM generation for structured data
        prompt = f"Generate the pronunciation, definition, mnemonic tip, and a usage sentence for the high-level SAT word: {word}. Return only the JSON object."
        list_schema = {"type": "array", "items": SatWord.model_json_schema()}
        config = types.GenerateContentConfig(response_mime_type="application/json", response_json_schema=list_schema)
        response = gemini_client.models.generate_content(model="gemini-2.5-flash", contents=prompt, config=config)
        data_list = json.loads(response.text)
        new_word_data = SatWord(**data_list[0]).model_dump()
        
        # Enrich with audio and briefing content
        new_word_data = task_controller._enrich_word(new_word_data)
        
    except Exception as e:
        st.error(f"🔴 Failed to generate content for '{word}'. Error: {e}"); return

    if save_word_to_db(new_word_data):
        # Reset data load state to force reload with new word
        st.session_state.vocab_data = None 
        st.session_state.has_more_data = True
        st.session_state.total_word_count = 0
        st.success(f"✅ Successfully added '{new_word_data['word']}' with ALL content to DB! Reloading data...")
        st.rerun()
    else:
        st.error("🔴 Failed to save to DB.")

def auto_generate_briefings():
    """Admin Auto-Task for processing LEGACY words missing the 2-minute briefing."""
    if not st.session_state.is_admin or st.session_state.autotask_running:
        return

    words_to_brief_indices = [
        i for i, d in enumerate(st.session_state.vocab_data) 
        if not d.get('briefing_audio_base64') 
    ]
    
    if not words_to_brief_indices:
        return

    # Process only a small batch automatically to prevent resource exhaustion
    batch_indices = words_to_brief_indices[:BRIEFING_BATCH_SIZE]
    
    if task_controller.run_task_in_thread(
        task_controller._generate_briefing_batch, 
        batch_indices=batch_indices, 
        batch_size=BRIEFING_BATCH_SIZE
    ):
        st.session_state.autotask_message = f"Admin Auto-Task: Generating {len(batch_indices)} LEGACY missing Briefings..."
        
def auto_generate_briefings_manual(batch_size: int):
    """Manually triggers a large batch generation of missing briefing content."""
    if st.session_state.autotask_running:
        st.warning("A background task is already running. Please wait.")
        return

    words_to_brief_indices = [
        i for i, d in enumerate(st.session_state.vocab_data)  
        if not d.get('briefing_audio_base64') 
    ]
    
    if not words_to_brief_indices:
        st.session_state.autotask_message = "All words already have 2-Minute Briefing content!"
        st.rerun()
        return

    batch_indices = words_to_brief_indices[:batch_size]
    
    if task_controller.run_task_in_thread(
        task_controller._generate_briefing_batch, 
        batch_indices=batch_indices, 
        batch_size=batch_size
    ):
        st.session_state.autotask_message = f"Status: Manually starting bulk generation for {len(batch_indices)} missing briefings (Batch Size {batch_size})...."

def handle_fix_single_audio(word_index: int):
    """Generates missing pronunciation audio for a single word and updates the DB document."""
    if st.session_state.autotask_running:
        st.error("A background task is running. Please wait.")
        return
    if word_index < 0 or word_index >= len(st.session_state.vocab_data):
        st.error("Invalid word index."); return
        
    word_data = st.session_state.vocab_data[word_index]
    word = word_data['word']
    
    st.info(f"Attempting to fix pronunciation for '{word}'...")
    audio_data = generate_tts_audio(word)
    
    if audio_data:
        fields_to_update = {'audio_base64': audio_data}
        if update_word_in_db(word_data, fields_to_update):
            st.session_state.vocab_data[word_index].update(fields_to_update)
            st.success(f"✅ Successfully fixed audio for '{word}' and saved to DB.")
        else:
            st.error(f"🔴 Audio generated, but failed to save update to DB for '{word}'.")
    else:
        st.error(f"🔴 Failed to fix audio for '{word}'. TTS generation may still be failing.")
    st.rerun()

def handle_bulk_audio_fix():
    """Attempts to generate and save missing pronunciation audio for all corrupted words."""
    if st.session_state.autotask_running:
        st.error("A background task is running. Please wait.")
        return
        
    words_to_fix_indices = [i for i, d in enumerate(st.session_state.vocab_data) if d.get('audio_base64') is None]
    
    if not words_to_fix_indices: st.success("All loaded words already have pronunciation audio!"); return

    status_placeholder = st.empty()
    fixed_count = 0
    total_count = len(words_to_fix_indices)

    status_placeholder.info(f"Starting bulk fix for {total_count} corrupted words...")
    
    with st.spinner("Processing audio fix... this may take a moment."):
        for i, index in enumerate(words_to_fix_indices):
            word_data = st.session_state.vocab_data[index]
            word = word_data['word']
            audio_data = generate_tts_audio(word)
    
            if audio_data:
                fields_to_update = {'audio_base64': audio_data}
                if update_word_in_db(word_data, fields_to_update):
                    st.session_state.vocab_data[index].update(fields_to_update)
                    fixed_count += 1
                else:
                    st.warning(f"Audio fixed for {word}, but save to DB failed.")
    
    if fixed_count > 0:
        st.session_state.total_word_count = get_total_word_count() 
        st.success(f"✅ Bulk fix complete! Successfully repaired audio for {fixed_count} of {total_count} words.")
    else:
        st.error(f"🔴 Bulk fix attempted, but audio generation failed for all {total_count} words or failed to save to DB. Check server logs/quotas.")
        
    status_placeholder.empty()
    st.rerun()

def handle_auth(email: str, password: str):
    """Handles Mock user registration and login."""
    if not email or not password: st.error("Please enter both Email and Password."); return
        
    is_admin = (email == ADMIN_EMAIL and password == ADMIN_PASSWORD)
    is_valid_user = is_admin or (len(password) >= 6 and '@' in email and '.' in email)
    
    if not is_valid_user: st.error("Invalid credentials."); return

    st.session_state.current_user_email = email
    st.session_state.is_auth = True
    st.session_state.is_admin = is_admin
    st.session_state.current_page_index = 0
    st.session_state.quiz_start_index = 0
    st.session_state.drill_word_index = 0 
    st.session_state.autotask_message = "Logged in successfully. Starting data check..."
    
    # We delay loading data until the main block to trigger auto-fetch logic
    st.session_state.vocab_data = None
    st.session_state.has_more_data = True
    st.session_state.total_word_count = 0
    st.session_state.auto_fetch_triggered = False 
    st.session_state.initial_auth_rerun_done = False # Reset flag on new login
    st.rerun()
    
def handle_logout():
    """Handles session state reset."""
    st.session_state.is_auth = False
    st.session_state.current_user_email = None
    st.session_state.quiz_active = False
    st.session_state.is_admin = False
    st.session_state.data_refresh_key = 0
    st.session_state.vocab_data = None
    st.session_state.has_more_data = True
    st.session_state.total_word_count = 0
    st.session_state.auto_fetch_triggered = False
    st.session_state.initial_auth_rerun_done = False
    st.rerun()
    
def manual_refresh_callback():
    """Callback function for the Force Reload Data button."""
    if st.session_state.autotask_running:
        st.warning("Cannot refresh data while a background task is running.")
        return
    st.session_state.vocab_data = None 
    st.session_state.total_word_count = 0
    st.info("Initiating manual data refresh...")
    st.rerun()

# ======================================================================
# 6. UI COMPONENTS (UNCHANGED)
# ======================================================================

def data_board_ui():
    """Displays key metrics and the status of background tasks."""
    
    if not st.session_state.is_auth: return
    
    word_count = st.session_state.total_word_count
    
    # Calculate missing briefings only if data is loaded
    missing_briefing_count = len([d for d in st.session_state.vocab_data if not d.get('briefing_audio_base64')]) if st.session_state.vocab_data else 0
    loaded_count = len(st.session_state.vocab_data) if st.session_state.vocab_data else 0

    st.header("📊 Application Status Board")
    
    cols = st.columns(4)
    
    with cols[0]:
        st.metric(label="Total Words (DB)", value=word_count, delta=f"Target: {REQUIRED_WORD_COUNT}")
    with cols[1]:
        st.metric(label="Words Loaded (RAM)", value=loaded_count) 
    with cols[2]:
        st.metric(label="Words Missing Briefing", value=missing_briefing_count)
    with cols[3]:
        status_message = st.session_state.get('autotask_message', "System Idle.")
        
        if st.session_state.autotask_running:
             st.info(f"**Status (Running):** {status_message}")
             st.spinner("Processing...") 
        elif st.session_state.autotask_status == 'Complete' or "complete" in status_message:
             st.success(f"**Status (Complete):** {status_message}")
        elif st.session_state.autotask_status == 'Error':
             st.error(f"**Status (Error):** {status_message}")
        else:
             st.markdown(f"**Status:** {status_message}")
            
    st.markdown("---")

def display_vocabulary_ui():
    """Renders the Vocabulary Display feature with Paging functionality based on loaded data."""
    st.header("📚 Vocabulary Display", divider="blue")
    
    if st.session_state.vocab_data is None or not st.session_state.vocab_data:
        st.info("The vocabulary list is empty. Please use the **Data Tools** tab to generate the first batch of words.")
        return

    total_loaded_words = len(st.session_state.vocab_data)
    total_db_words = st.session_state.total_word_count

    start_index = st.session_state.current_page_index * LOAD_BATCH_SIZE
    end_index = min(start_index + LOAD_BATCH_SIZE, total_loaded_words)
    
    if start_index >= total_loaded_words and total_loaded_words > 0:
        st.session_state.current_page_index = max(0, (total_loaded_words // LOAD_BATCH_SIZE) - 1)
        st.rerun()
        return

    
    st.markdown(f"**Showing Words {start_index + 1} - {end_index} of {total_db_words} High-Level SAT Words** (Loaded: {total_loaded_words})")
    
    
    with st.container(border=True): 
        
        for i, data in enumerate(st.session_state.vocab_data[start_index:end_index]):
            word_number = start_index + i + 1 
            word = data.get('word', 'N/A').upper()
            pronunciation = data.get('pronunciation', 'N/A')
            definition = data.get('definition', 'N/A')
            tip = data.get('tip', 'N/A')
            usage = data.get('usage', 'N/A')
            audio_base64 = data.get('audio_base64') 
            
            expander_title = f"**{word_number}. {word}** — {pronunciation}" 
            
            with st.expander(expander_title):
                if audio_base64:
                    audio_data_url = f"data:audio/mp3;base64,{audio_base64}"
                    audio_html = f"""
                        <audio controls style="width: 100%;" src="{audio_data_url}">
                            Your browser does not support the audio element.
                        </audio>
                    """
                    st.markdown(audio_html, unsafe_allow_html=True)
                else:
                    st.warning("Audio not available for this word.")
                    # Button logic relies on st.session_state.autotask_running, which caused the error.
                    # We rely on the master disable in the Admin UI now.
                    if st.session_state.is_admin:
                        # Use conditional check inside the button to prevent crash
                        st.button(
                            f"Fix Audio for #{word_number}", 
                            key=f"fix_audio_{start_index + i}", 
                            on_click=handle_fix_single_audio, 
                            args=(start_index + i,),
                            type="primary"
                        )

                st.markdown(f"**📖 Definition:** {definition.capitalize()}") 
                st.markdown(f"**💡 Memory Tip:** *{tip}*") 
                st.markdown(f"**🗣️ Usage:** *'{usage}'*") 
            
            if i < LOAD_BATCH_SIZE - 1 and (start_index + i + 1) < total_loaded_words:
                 st.markdown("---")

    col_prev, col_status, col_next = st.columns([1, 2, 1])
    
    with col_prev:
        if st.session_state.current_page_index > 0:
            st.button("⬅️ Previous 10 Words", on_click=go_to_prev_page)
    
    with col_status:
        current_page = st.session_state.current_page_index + 1
        max_loaded_pages = (total_loaded_words + LOAD_BATCH_SIZE - 1) // LOAD_BATCH_SIZE
        st.markdown(f"<div style='text-align: center; padding-top: 10px;'>Page {current_page} of ~{max_loaded_pages} (loaded)</div>", unsafe_allow_html=True)

    with col_next:
        can_go_next = (end_index < total_loaded_words) or st.session_state.has_more_data
        
        button_label = "Next 10 Words ➡️"
        if end_index == total_loaded_words and st.session_state.has_more_data:
            button_label = "Fetch Next Batch ➡️"
            
        if can_go_next:
            st.button(button_label, on_click=go_to_next_page, type="secondary")

def generate_quiz_ui():
    """Renders the Quiz Section feature."""
    st.header("📝 Vocabulary Quiz", divider="green")
    
    total_words = len(st.session_state.vocab_data) if st.session_state.vocab_data else 0
    
    if total_words < QUIZ_SIZE:
        st.info(f"A minimum of {QUIZ_SIZE} words is required to start a quiz. Current total: {total_words}. Please generate more data.")
        return

    start_word_num = st.session_state.quiz_start_index + 1
    end_word_num = min(st.session_state.quiz_start_index + QUIZ_SIZE, total_words)

    def start_new_quiz():
        start = st.session_state.quiz_start_index
        end = start + QUIZ_SIZE
        words_pool = st.session_state.vocab_data[start:end]
        if len(words_pool) < QUIZ_SIZE: st.error(f"Cannot start quiz. Need {QUIZ_SIZE} words starting from position {start + 1}."); return
        
        quiz_details = []
        all_definitions = {d['definition'].capitalize() for d in st.session_state.vocab_data}
        all_definitions_list = list(all_definitions)
        
        for question_data in words_pool:
            correct_answer = question_data['definition'].capitalize()
            decoys = random.sample([d for d in all_definitions_list if d != correct_answer], min(3, len([d for d in all_definitions_list if d != correct_answer])))
            options = [correct_answer] + decoys
            random.shuffle(options)
            original_word_index = st.session_state.vocab_data.index(question_data) + 1
            
            quiz_details.append({"word": question_data['word'], "correct_answer": correct_answer, "tip": question_data['tip'], "usage": question_data['usage'], "options": options, "index": original_word_index})
            
        st.session_state.quiz_details = quiz_details
        st.session_state.quiz_active = True
        st.session_state.quiz_results = None 
        st.rerun()

    def advance_quiz_index():
        st.session_state.quiz_start_index += QUIZ_SIZE
        st.session_state.quiz_active = False 
        st.rerun()
    
    if not st.session_state.quiz_active:
        
        if start_word_num > total_words:
            st.info("You have completed all available quiz blocks! Resetting to start.")
            st.session_state.quiz_start_index = 0
            start_word_num = 1
            end_word_num = min(QUIZ_SIZE, total_words)
            
        st.markdown(f"**Current Quiz Block:** Words {start_word_num} through {end_word_num}.")
        
        st.button(f"Start Quiz on Words #{start_word_num} - #{end_word_num}", on_click=start_new_quiz, type="primary")
        return
    
    if st.session_state.quiz_results is not None:
        score = st.session_state.quiz_results['score']
        total = st.session_state.quiz_results['total']
        accuracy = st.session_state.quiz_results['accuracy']
        
        if score == total: st.balloons(); st.success(f"🎉 Quiz Complete! Perfect Score! {score} out of {total} (Accuracy: {accuracy}%)")
        else: st.warning(f"Quiz Complete! Final Score: **{score}** out of **{total}** (Accuracy: {accuracy}%)")
            
        st.subheader("Review Your Answers")
        for i, result in enumerate(st.session_state.quiz_results['feedback']):
            st.markdown(f"#### **Word #{result['index']}: {result['word']}**") 
            st.markdown(f"**Your Answer:** {result['user_choice']}")
            st.markdown(f"**Correct Answer:** {result['correct_answer']}")
            if not result['is_correct']:
                st.markdown(f"**Memory Tip:** *{result['tip']}*")
                st.markdown(f"**Usage:** *'{result['usage']}'*")
            st.markdown("---")
            
        st.session_state.quiz_active = False 
        st.session_state.quiz_results = None 
        
        next_start_index = st.session_state.quiz_start_index + QUIZ_SIZE
        if next_start_index < total_words:
            st.button(f"Start Next Quiz Block (Words #{next_start_index + 1} - #{min(next_start_index + QUIZ_SIZE, total_words)})", on_click=advance_quiz_index, type="secondary")
        else:
            st.info("You have completed all available words in the database!")
            st.session_state.quiz_start_index = 0
            st.button("Restart Quiz from Word #1", on_click=advance_quiz_index, type="secondary")
            
        return
    
    quiz_details = st.session_state.quiz_details
    
    with st.form(key="full_quiz_form"):
        st.subheader(f"Answer the following {QUIZ_SIZE} questions:")
        if 'user_responses' not in st.session_state: st.session_state.user_responses = [None] * QUIZ_SIZE
        
        for i, q in enumerate(quiz_details):
            st.markdown(f"#### **Word #{q['index']}. Define: {q['word'].upper()}**") 
            user_choice = st.radio("Select the correct definition:", q['options'], key=f"quiz_q_{i}", index=None, label_visibility="collapsed")
            st.session_state.user_responses[i] = user_choice

        submitted = st.form_submit_button("Submit All Answers")

        if submitted:
            final_score = 0
            feedback_list = []
            
            if any(response is None for response in st.session_state.user_responses):
                st.error("Please answer ALL questions before submitting."); return

            for i, response in enumerate(st.session_state.user_responses):
                q = quiz_details[i]
                is_correct = (response == q['correct_answer'])
                if is_correct: final_score += 1
                
                feedback_list.append({"word": q['word'], "user_choice": response, "correct_answer": q['correct_answer'], "is_correct": is_correct, "tip": q['tip'], "usage": q['usage'], "index": q['index']})
            
            st.session_state.quiz_results = {"score": final_score, "total": QUIZ_SIZE, "accuracy": round((final_score / QUIZ_SIZE) * 100, 1), "feedback": feedback_list}
            del st.session_state.user_responses
            st.rerun()

def two_minute_drill_ui():
    """Renders the UI for the 2-Minute Word Briefing feature."""
    st.header("⏱️ 2-Minute Drill", divider="red")

    def next_drill_word():
        if st.session_state.drill_word_index < len(st.session_state.vocab_data) - 1:
            st.session_state.drill_word_index += 1
        elif len(st.session_state.vocab_data) > 0:
            st.session_state.drill_word_index = 0 
        st.rerun()

    def prev_drill_word():
        if st.session_state.drill_word_index > 0:
            st.session_state.drill_word_index -= 1
        elif len(st.session_state.vocab_data) > 0:
            st.session_state.drill_word_index = len(st.session_state.vocab_data) - 1
        st.rerun()

    if st.session_state.vocab_data is None or not st.session_state.vocab_data:
        st.info("No vocabulary loaded yet. Please generate some words via the Data Tools tab."); return

    total_words = len(st.session_state.vocab_data)
    current_index = st.session_state.drill_word_index
    
    if current_index >= total_words: st.session_state.drill_word_index = 0; current_index = 0; st.rerun(); return
        
    selected_word_data = st.session_state.vocab_data[current_index]
    selected_word_str = selected_word_data.get('word', 'N/A').upper()
    
    st.markdown(f"**Current Word:** **{current_index + 1}** of **{total_words}**")

    briefing_text = selected_word_data.get('briefing_text')
    briefing_audio_base64 = selected_word_data.get('briefing_audio_base64')
    briefing_exists_in_db = bool(briefing_audio_base64)

    briefing = None
    
    if briefing_exists_in_db:
        briefing = {"text": briefing_text, "audio_base64": briefing_audio_base64}
        st.success("Briefing content loaded from database.")
    
    if not briefing_exists_in_db and st.session_state.is_admin:
        st.warning(f"Briefing content missing for {selected_word_str}. Generate it now!")
        # Use standard button outside of complex form to prevent conflict
        if st.button(f"Generate and Save Briefing for {selected_word_str}", type="primary", key="manual_drill_gen"):
            auto_generate_briefings_manual(1); st.rerun() 
    
    if briefing:
        st.subheader(f"Deep Dive: {selected_word_str}")
        
        if briefing['audio_base64']:
            audio_data_url = f"data:audio/mp3;base64,{briefing['audio_base64']}"
            audio_html = f"""<audio controls style="width: 100%;" src="{audio_data_url}"></audio>"""
            st.markdown(audio_html, unsafe_allow_html=True)
            st.markdown("---")
            
        st.markdown("##### 🔊 Full Briefing Transcript")
        clean_briefing_text = re.sub(r'[\u2013\u2014]', '-', briefing['text']) 
        st.markdown(clean_briefing_text)
        st.markdown("---")
        st.info(f"The briefing is about {len(briefing['text'].split())} words long.")
    elif not briefing_exists_in_db and not st.session_state.is_admin:
        st.info("The 2-Minute Briefing for this word is currently missing. The Admin is running an automatic fix task to generate this content. Please check back later!")
    
    col_prev, col_next = st.columns([1, 1])
    
    with col_prev:
        if current_index > 0: st.button("⬅️ Previous Word", on_click=prev_drill_word)
    
    with col_next:
        if current_index < total_words - 1: st.button("Next Word ➡️", on_click=next_drill_word, type="secondary")
        elif total_words > 0: st.button("↩️ Start Over", on_click=next_drill_word, type="secondary")

# Dummy function to show warning when the task is running
def dummy_warning_callback():
    st.session_state.autotask_message = "🛑 A background task is running! Please wait for completion."
    st.session_state.autotask_status = 'Running'
    st.rerun()

def render_admin_tools(container: st.delta_generator.DeltaGenerator):
    """
    Renders all interactive admin elements into the given container using simplified 
    st.button and eliminating complex, nested form structures.
    """
    
    # --- MANUAL WORD ENTRY (Synchronous) ---
    container.subheader("Manual Word & All Content Entry")
    
    # Use st.form only for the input to ensure clean state submission if needed.
    with container.form(key="manual_word_input_form", clear_on_submit=True):
        manual_word = st.text_input("Enter SAT-Level Word to Add:", key="manual_word_input_active").strip()
        manual_submit = st.form_submit_button("Generate ALL Content (Synchronous & Slow)")
        if manual_submit: 
            handle_manual_word_entry(manual_word)
            return

    container.markdown("---")
    
    # --- BULK AND REFRESH TOOLS (Using simplified st.button calls) ---
    container.subheader("Audio Integrity & Bulk Fix (Legacy Word Processing)")
    
    if st.session_state.vocab_data:
        missing_audio_count = len([d for d in st.session_state.vocab_data if d.get('audio_base64') is None])
        missing_briefing_count = len([d for d in st.session_state.vocab_data if not d.get('briefing_audio_base64')])
    else: missing_audio_count = 0; missing_briefing_count = 0
            
    container.markdown(f"**Corrupted Entries (Pronunciation):** {missing_audio_count} words.")
    container.markdown(f"**Missing Briefings (2-Min Drill - Legacy):** {missing_briefing_count} words.") 
    
    col_audio_fix, col_briefing_gen = container.columns(2)
    
    with col_audio_fix:
        container.button("Attempt Bulk Audio Fix", key="btn_bulk_audio_fix_simple", type="primary", on_click=handle_bulk_audio_fix)
    with col_briefing_gen:
        container.button(f"Force Generate {MANUAL_BRIEFING_BATCH} Missing Briefings (Background Task)", key="btn_force_briefing_simple", type="secondary", on_click=lambda: auto_generate_briefings_manual(MANUAL_BRIEFING_BATCH))

    container.markdown("---")
    container.subheader("Vocabulary Extraction (Bulk - Background Task)")
    container.markdown(f"**Total Words in Database:** `{st.session_state.total_word_count}` (Target: {REQUIRED_WORD_COUNT}).")
    
    container.button(f"Force Extract {MANUAL_EXTRACT_BATCH} New Words (Background Task)", key="btn_force_extract_simple", type="secondary", on_click=lambda: handle_admin_extraction_button(MANUAL_EXTRACT_BATCH, auto_fetch=False))

    container.markdown("---")
    
    container.subheader("Manual Data Refresh (Cache Bust)")
    # This is the final button that has been crashing. Now completely isolated.
    container.button("Force Reload Data from DB", key="btn_force_reload_final", type="danger", on_click=manual_refresh_callback)

def render_admin_status(container: st.delta_generator.DeltaGenerator):
    """Renders the disabled status message into the given container."""
    container.info("🛑 **A Background Task is Running!** Data manipulation buttons are currently inactive. Check the **Application Status Board** for progress.")
    container.markdown("---")
    container.subheader("Manual Word & All Content Entry")
    container.text_input("Enter SAT-Level Word to Add:", key="manual_word_input_disp", disabled=True)
    container.button("Generate ALL Content (Synchronous & Slow)", disabled=True)
    container.markdown("---")
    container.subheader("Bulk Operations Inactive")
    container.info("Wait for background process to finish before using Bulk Tools.")
    container.markdown("---")
    container.subheader("Manual Data Refresh (Cache Bust)")
    container.button("Force Reload Data from DB", key="btn_force_reload_disp", type="danger", disabled=True)


def admin_extraction_ui():
    """Renders the Admin Extraction and User Management feature."""
    st.header("💡 Data Tools", divider="orange") 
    
    if not st.session_state.is_admin: 
        st.warning("You must be logged in as the Admin to use this tool.")
        return
    
    is_task_running = st.session_state.autotask_running
    
    # Use a simple, non-cached container for the primary rendering block.
    primary_container = st.container() 
    
    if is_task_running:
        render_admin_status(primary_container)
    else:
        render_admin_tools(primary_container)


# ======================================================================
# 7. STREAMLIT APPLICATION STRUCTURE (MAIN)
# ======================================================================

def main():
    """The main Streamlit application function."""
    st.set_page_config(page_title="AI Vocabulary Builder", layout="wide")
    st.title("🧠 AI-Powered Vocabulary Builder")
    
    initialize_session_state()
    
    # Check if a background task completed and needs a UI update
    if st.session_state.is_admin:
        task_controller.check_task_status()
    
    with st.sidebar:
        st.header("User Login")
        if not st.session_state.is_auth:
            st.markdown("##### New User Registration / Existing User Login")
            user_email = st.text_input("📧 Email", key="user_email_input", value=st.session_state.current_user_email or "")
            password = st.text_input("🔑 Password", type="password", key="password_input")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Login", key="login_btn", type="primary"): handle_auth(user_email, password)
            with col2:
                if st.button("Register", key="register_btn"): handle_auth(user_email, password)
            st.markdown("---")
            st.markdown(f"**Admin Login:** `{ADMIN_EMAIL}` / `Jamshaid,1981`")
        else:
            display_name = "Admin" if st.session_state.is_admin else st.session_state.current_user_email
            st.success(f"Logged in as: **{display_name}**")
            if st.button("Log Out", on_click=handle_logout): pass
                
    # --- Data Loading and Auto-Fetch Logic ---
    if st.session_state.is_auth:
        
        # 1. Load initial data if it hasn't been loaded yet (vocab_data is None)
        # This occurs on the first run after login.
        if st.session_state.vocab_data is None:
            with st.spinner("Downloading initial vocabulary records from DB... Please wait."):
                load_and_update_vocabulary_data()
            
            # CRITICAL: If this is the *first* run after login, flag it and RERUN immediately.
            # This ensures the UI draws once cleanly before auto-tasks fire.
            if not st.session_state.initial_auth_rerun_done:
                 st.session_state.initial_auth_rerun_done = True
                 st.rerun()
            
        # 2. AUTOMATIC DATA FETCHING/FIXING LOGIC (Admin Only, runs on the SECOND run after load)
        if st.session_state.is_admin and st.session_state.initial_load_done and st.session_state.initial_auth_rerun_done and not st.session_state.autotask_running and not st.session_state.auto_fetch_triggered:
            
            # If database is nearly empty, automatically start bulk extraction of 25 words
            if st.session_state.total_word_count < AUTO_FETCH_THRESHOLD:
                # Set flag to prevent double-triggering
                st.session_state.auto_fetch_triggered = True 
                handle_admin_extraction_button(AUTO_FETCH_BATCH, auto_fetch=True)
            
            # If database is populated, check for and generate missing briefings
            else:
                auto_generate_briefings() 

        # --- Display Core UI (Always visible after data has been attempted to load) ---
        
        # Show a placeholder message if the database is confirmed empty and auto-fetch is starting
        if st.session_state.total_word_count == 0 and st.session_state.is_admin and st.session_state.autotask_running:
            status_msg = "The automatic generation task has been initiated in the background. Please wait a few moments and check the **Application Status Board**."
            st.info(status_msg)

        data_board_ui()

        tab_display, tab_quiz, tab_drill, tab_admin = st.tabs([
            "📚 Vocabulary List", 
            "📝 Quiz Section", 
            "⏱️ 2-Minute Drill",
            "🛠️ Data Tools"
        ])
        
        with tab_display: display_vocabulary_ui()
        with tab_quiz: generate_quiz_ui()
        with tab_drill: two_minute_drill_ui()
        with tab_admin: admin_extraction_ui()

    else:
        st.info("Please log in or register using the sidebar to access the Vocabulary Builder.")

if __name__ == "__main__":
    main()
