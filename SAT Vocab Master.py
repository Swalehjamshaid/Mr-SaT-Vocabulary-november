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
    from firebase_admin import credentials, initialize_app, firestore
    import firebase_admin
except ImportError:
    st.error("FIREBASE ERROR: The required library 'firebase-admin' is likely missing. Please install it.")
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
# 1. CONFIGURATION & MODELS (UNCHANGED)
# ======================================================================

# --- App State and Constants ---
REQUIRED_WORD_COUNT = 2000 
LOAD_BATCH_SIZE = 10         # Used for UI display and database fetching
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
    """Pydantic model for a vocabulary word, defining required structure."""
    word: str = Field(description="The SAT-level word.")
    pronunciation: str = Field(description="Simple, hyphenated phonetic pronunciation (e.g., eh-FEM-er-al).")
    definition: str = Field(description="The concise dictionary definition.")
    tip: str = Field(description="A short, catchy mnemonic memory tip.")
    usage: str = Field(description="A professional sample usage sentence.")
    sat_level: str = Field(default="High", description="Should always be 'High'.")
    audio_base64: Optional[str] = Field(default=None, description="Base64 encoded audio data for word pronunciation.")
    created_at: float = Field(default_factory=time.time)
    
    # PERMANENTLY STORED BRIEFING FIELDS
    briefing_text: Optional[str] = Field(default=None, description="The extended AI-generated briefing text.")
    briefing_audio_base64: Optional[str] = Field(default=None, description="Base64 encoded audio data for the briefing.")

# ======================================================================
# 2. SETUP & INITIALIZATION (Database and AI Clients - UNCHANGED)
# ======================================================================

@st.cache_resource
def initialize_firestore():
    # ... (Rest of Firebase initialization logic - unchanged) ...
    import firebase_admin
    from firebase_admin import credentials, firestore
    import tempfile
    import json
    
    temp_file_path = None
    try:
        firebase_secret_key = next((key for key in st.secrets.keys() if key.upper() == "FIREBASE"), None)
        if not firebase_secret_key:
            st.error("🔴 FIREBASE secret table not found.")
            raise KeyError("FIREBASE")
            
        service_account_info = dict(st.secrets[firebase_secret_key])
        if 'private_key' not in service_account_info:
             st.error("🔴 'private_key' field missing from [FIREBASE] secrets.")
             raise KeyError("private_key")
        
        private_key_content = service_account_info['private_key']
        service_account_info["private_key"] = private_key_content.replace('\\n', '\n').replace('\r', '')

        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json', encoding='utf-8') as f:
            json.dump(service_account_info, f)
            temp_file_path = f.name 

        if not firebase_admin._apps:
            cred = credentials.Certificate(temp_file_path)
            firebase_admin.initialize_app(cred)

        db = firestore.client() 
        return db.collection("sat_vocabulary")
        
    except Exception as e:
        st.error(f"🔴 FIREBASE INITIALIZATION FAILED. Root Cause: {e}.")
        st.stop()
        
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass 

try:
    VOCAB_COLLECTION = initialize_firestore()
    db = VOCAB_COLLECTION.firestore 
except Exception:
    st.stop()

if "GEMINI_API_KEY" not in st.secrets:
    st.error("🔴 GEMINI_API_KEY is missing! Please set it in your Streamlit Secrets.")
    st.stop()

try:
    gemini_client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
except Exception as e:
    st.error(f"🔴 Failed to initialize Gemini Client: {e}")
    st.stop()


# ======================================================================
# 3. CORE UTILITIES (TTS, Data, State)
# ======================================================================

def initialize_session_state():
    """Sets up default session state variables."""
    # ... (Unchanged state initialization) ...
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

    # NEW: LAZY LOADING STATE MANAGEMENT
    if 'last_word_timestamp' not in st.session_state: st.session_state.last_word_timestamp = None
    if 'has_more_data' not in st.session_state: st.session_state.has_more_data = True
    if 'total_word_count' not in st.session_state: st.session_state.total_word_count = 0


# --- NEW LAZY LOADING IMPLEMENTATION ---

def get_word_count_from_firestore() -> int:
    """Fetches the total document count (Expensive, but necessary for metrics)."""
    try:
        # Use an aggregate query for efficient counting
        agg_query = VOCAB_COLLECTION.aggregate.count().get()
        return agg_query[0][0].value
    except Exception as e:
        print(f"🔴 Firestore Count Failed: {e}")
        return len(st.session_state.vocab_data) if st.session_state.vocab_data else 0

@st.cache_data(show_spinner=False)
def fetch_vocabulary_batch(start_timestamp: Optional[float] = None) -> List[Dict]:
    """
    Fetches the next batch of words using the 'created_at' timestamp as a cursor.
    The data refresh key is implicitly handled by the calling functions rerunning.
    """
    print(f"--- FETCHING BATCH: Starting after timestamp {start_timestamp} ---")
    query = VOCAB_COLLECTION.order_by('created_at').limit(LOAD_BATCH_SIZE)
    
    # Use start_after with the timestamp (the field we are ordering by)
    if start_timestamp is not None:
        query = query.start_after({'created_at': start_timestamp})

    try:
        docs = query.stream()
        vocab_list = [doc.to_dict() for doc in docs]
        return vocab_list
    except Exception as e:
        print(f"🔴 Firestore Batch Load Failed: {e}")
        return []

def load_and_update_vocabulary_data():
    """
    Loads the INITIAL batch of data and calculates the total count.
    This is called only once on login.
    """
    if not st.session_state.is_auth or st.session_state.vocab_data is not None: return

    # 1. Fetch Total Count (Synchronous, but needed for the Status Board)
    st.session_state.total_word_count = get_word_count_from_firestore()
    
    # 2. Fetch Initial Batch (Fast)
    vocab_list = fetch_vocabulary_batch(start_timestamp=None)
    
    # 3. Update State
    st.session_state.vocab_data = vocab_list
    st.session_state.initial_load_done = True
    
    # Update cursor and 'has_more' flag
    if vocab_list:
        st.session_state.last_word_timestamp = vocab_list[-1]['created_at']
        st.session_state.has_more_data = len(vocab_list) == LOAD_BATCH_SIZE
    else:
        st.session_state.has_more_data = False
    
    word_count = len(st.session_state.vocab_data)
    
    if word_count > 0:
        st.info(f"✅ Loaded initial {word_count} words from shared database (Total: {st.session_state.total_word_count}).")
    elif st.session_state.is_auth:
        st.info(f"Database is empty. Total count: {st.session_state.total_word_count}.")


def fetch_and_append_next_batch():
    """Fetches the next batch and appends it to session state."""
    if not st.session_state.has_more_data:
        st.warning("No more data to load from the database.")
        return

    next_batch = fetch_vocabulary_batch(st.session_state.last_word_timestamp)
    
    if next_batch:
        # Append new data
        st.session_state.vocab_data.extend(next_batch)
        
        # Update cursor
        st.session_state.last_word_timestamp = next_batch[-1]['created_at']
        
        # Check if this was the last possible batch
        st.session_state.has_more_data = len(next_batch) == LOAD_BATCH_SIZE
        
        st.success(f"Loaded {len(next_batch)} more words.")
    else:
        st.session_state.has_more_data = False
        st.info("Reached the end of the vocabulary list.")
        
    # Total count may have increased due to admin tasks, refetch for accuracy
    st.session_state.total_word_count = get_word_count_from_firestore()
    st.rerun()


# --- Pagination Logic Update ---

def go_to_next_page():
    # If the current page is the last loaded page, fetch the next batch first.
    total_loaded = len(st.session_state.vocab_data)
    max_index = (total_loaded // LOAD_BATCH_SIZE) - 1
    current_index = st.session_state.current_page_index
    
    if current_index == max_index and st.session_state.has_more_data:
        # We are at the end of the currently loaded data, but there is more in the DB.
        fetch_and_append_next_batch() 
        # Rerun handles the index increment implicitly in the main loop
    
    st.session_state.current_page_index += 1
    st.rerun()

def go_to_prev_page():
    # Only need to decrement index and rerun
    st.session_state.current_page_index -= 1
    st.rerun()


# --- Unchanged Helper Functions (TTS, Gemini, Firestore Writes) ---
def generate_tts_audio(text: str) -> Optional[str]:
    # ... (Unchanged) ...
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
    # ... (Unchanged) ...
    word = word_data.get('word', 'a high-level word')
    definition = word_data.get('definition', 'a complex meaning')
    
    prompt = f"""
    You are a vocabulary tutor. Write a **short, memorable, and concise briefing (5-6 sentences maximum, about 60-80 words)** on the word '{word}'. 
    
    The briefing must seamlessly include:
    1. The core definition: {definition}.
    2. A brief note on its origin or etymology (1 sentence).
    3. One compelling example sentence demonstrating high-level usage.
    4. A final, memorable takeaway.
    
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
        
        if not audio_data:
            print(f"🔴 Failed to generate audio for briefing text: '{briefing_text[:20]}...'")
            return None 

        return {
            "briefing_text": briefing_text,
            "briefing_audio_base64": audio_data 
        }
        
    except Exception as e:
        print(f"🔴 Gemini/Briefing Generation Failed for '{word}': {e}")
        return None

def save_word_to_firestore(word_data: Dict) -> bool:
    # ... (Unchanged) ...
    try:
        doc_ref = VOCAB_COLLECTION.document(word_data['word'].lower())
        doc_ref.set(word_data, merge=False)
        return True
    except Exception as e:
        print(f"🔴 Firestore Save Failed for {word_data['word']}: {e}")
        return False
        
def update_word_in_firestore(word_data: Dict, fields_to_update: Dict) -> bool:
    # ... (Unchanged) ...
    try:
        doc_ref = VOCAB_COLLECTION.document(word_data['word'].lower())
        doc_ref.update(fields_to_update)
        return True
    except Exception as e:
        print(f"🔴 Firestore Update Failed for {word_data['word']}: {e}")
        return False


# ======================================================================
# 4. ASYNCHRONOUS TASK CONTROLLER (Long-Running LLM/TTS/Database Ops)
#    (MINIMAL CHANGE: removed `increment_data_refresh_key` call,
#     relying on total count refetch and direct state update)
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
        if st.session_state.autotask_running:
            return False 
        
        st.session_state.task_thread = threading.Thread(
            target=target_function, 
            args=args, 
            kwargs=kwargs,
            daemon=True
        )
        self._update_session_state('Running', 'Task initiated...', True)
        st.session_state.task_thread.start()
        return True

    def check_task_status(self):
        if st.session_state.autotask_running and st.session_state.task_thread and not st.session_state.task_thread.is_alive():
            # Task finished, update state and trigger rerun
            self._update_session_state('Complete', st.session_state.autotask_message or 'Task complete. Reloading data.', False)
            # CRITICAL: Since data was added/updated, update the total count.
            st.session_state.total_word_count = get_word_count_from_firestore()
            st.rerun() 
        elif st.session_state.autotask_running:
             st.rerun()

    # --- THREAD TARGET FUNCTIONS (Non-UI, Blocking I/O) ---
    def _extract_and_save_batch(self, num_words: int, existing_words: List[str], auto_fetch: bool):
        # ... (Unchanged _extract_and_save_batch, relies on save_word_to_firestore) ...
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
            
            # Use ThreadPoolExecutor for concurrent TTS/Briefing generation
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_word = {
                    executor.submit(self._enrich_word, word_data): word_data 
                    for word_data in validated_words
                }
                
                enriched_words = []
                for future in future_to_word:
                    enriched_words.append(future.result())

            st.session_state.autotask_message = f"Saving {len(enriched_words)} words to Firestore..."
            
            for word_data in enriched_words:
                if save_word_to_firestore(word_data):
                    successful_saves += 1
            
            st.session_state.autotask_message = f"✅ Extracted and saved {successful_saves} words."
            
        except Exception as e:
            st.session_state.autotask_status = 'Error'
            st.session_state.autotask_message = f"🔴 Extraction Failed: {e}"
        finally:
            st.session_state.autotask_running = False
            
    def _enrich_word(self, word_data: Dict) -> Dict:
        # ... (Unchanged helper) ...
        # Pronunciation Audio
        pronunciation_audio = generate_tts_audio(word_data['word'])
        word_data['audio_base64'] = pronunciation_audio if pronunciation_audio else None
        
        # 2-Minute Briefing
        briefing_content = generate_full_briefing_content(word_data)
        if briefing_content:
            word_data.update(briefing_content)
        return word_data
        
    def _generate_briefing_batch(self, batch_indices: List[int], batch_size: int):
        # ... (Unchanged _generate_briefing_batch, relies on update_word_in_firestore) ...
        try:
            generated_count = 0
            words_to_process = [st.session_state.vocab_data[i] for i in batch_indices]

            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_word = {
                    executor.submit(self._enrich_briefing, word_data): word_data 
                    for word_data in words_to_process
                }

                for future in future_to_word:
                    result = future.result()
                    if result:
                        word_data = future_to_word[future]
                        if update_word_in_firestore(word_data, result):
                             # Update session state *after* successful DB save
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
        # ... (Unchanged helper) ...
        briefing_content = generate_full_briefing_content(word_data)
        if briefing_content:
            return briefing_content
        return None

task_controller = LongRunningTaskController()


# ======================================================================
# 5. HANDLERS (Adjusted for Lazy Loading)
# ======================================================================

def handle_admin_extraction_button(num_words: int, auto_fetch: bool = False):
    """Triggers the bulk word extraction in a background thread."""
    if st.session_state.autotask_running:
        st.warning("A background task is already running. Please wait.")
        return

    # Pass only the words currently loaded in state to prevent LLM duplicates
    existing_words = [d['word'] for d in st.session_state.vocab_data if st.session_state.vocab_data]
    
    if task_controller.run_task_in_thread(
        task_controller._extract_and_save_batch, 
        num_words=num_words, 
        existing_words=existing_words, 
        auto_fetch=auto_fetch
    ):
        st.session_state.autotask_message = f"Initiated extraction of {num_words} words..."

def handle_manual_word_entry(word: str):
    """Generates all content for a single word and saves it to Firestore."""
    if not word or st.session_state.autotask_running:
        st.error("Please enter a word or wait for the current task to finish.")
        return
        
    st.info(f"Generating content for '{word}'...")
    
    # This must be synchronous for the immediate result feedback
    try:
        # 1. Get Base Word Data via LLM
        prompt = f"Generate the pronunciation, definition, mnemonic tip, and a usage sentence for the high-level SAT word: {word}. Return only the JSON object."
        list_schema = {"type": "array", "items": SatWord.model_json_schema()}
        config = types.GenerateContentConfig(response_mime_type="application/json", response_json_schema=list_schema)
        
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt, config=config
        )
        
        data_list = json.loads(response.text)
        new_word_data = SatWord(**data_list[0]).model_dump()
        
        # 2. Enrich: Pronunciation & Briefing (Synchronous)
        new_word_data = task_controller._enrich_word(new_word_data)
        
    except Exception as e:
        st.error(f"🔴 Failed to generate content for '{word}'. Error: {e}")
        return

    if save_word_to_firestore(new_word_data):
        # The key is reset here to force the UI to fetch the new first page 
        # (containing the new word if sorted by time)
        st.session_state.data_refresh_key += 1
        st.session_state.vocab_data = None # Force complete reload of initial batch
        st.session_state.total_word_count = 0
        st.success(f"✅ Successfully added '{new_word_data['word']}' with ALL content to Firebase! Reloading data...")
        st.rerun()
    else:
        st.error("🔴 Failed to save to Firebase.")


def handle_auth(email: str, password: str):
    """Handles Mock user registration and login."""
    if not email or not password:
        st.error("Please enter both Email and Password.")
        return
        
    is_admin = (email == ADMIN_EMAIL and password == ADMIN_PASSWORD)
    is_valid_user = is_admin or (len(password) >= 6 and '@' in email and '.' in email)
    
    if not is_valid_user:
        st.error("Invalid credentials. Registration/Login requires a valid email and 6+ character password.")
        return

    st.session_state.current_user_email = email
    st.session_state.is_auth = True
    st.session_state.is_admin = is_admin
    # Reset view controls on login
    st.session_state.current_page_index = 0
    st.session_state.quiz_start_index = 0
    st.session_state.drill_word_index = 0 
    st.session_state.autotask_message = "Logged in successfully. Starting data check..."
    
    # NEW: Reset data markers
    st.session_state.vocab_data = None
    st.session_state.last_word_timestamp = None
    st.session_state.has_more_data = True

    # SYNCHRONOUS LOAD WITH VISUAL SPINNER (Only fetches initial batch)
    with st.spinner("Downloading initial vocabulary records from Firestore... Please wait."):
        load_and_update_vocabulary_data() 
        
    st.rerun()
    
def handle_logout():
    # ... (Unchanged) ...
    st.session_state.is_auth = False
    st.session_state.current_user_email = None
    st.session_state.quiz_active = False
    st.session_state.is_admin = False
    st.session_state.data_refresh_key = 0
    st.session_state.vocab_data = None
    # Reset lazy load state
    st.session_state.last_word_timestamp = None
    st.session_state.has_more_data = True
    st.session_state.total_word_count = 0
    st.rerun()

# --- Rest of the existing helper functions (Quiz, Audio Fix) go here, mostly unchanged ---
def auto_generate_briefings():
    # ... (Unchanged) ...
    if not st.session_state.is_admin or st.session_state.autotask_running:
        return

    words_to_brief_indices = [
        i for i, d in enumerate(st.session_state.vocab_data) 
        if not d.get('briefing_audio_base64') 
    ]
    
    if not words_to_brief_indices:
        return

    batch_indices = words_to_brief_indices[:BRIEFING_BATCH_SIZE]
    
    if task_controller.run_task_in_thread(
        task_controller._generate_briefing_batch, 
        batch_indices=batch_indices, 
        batch_size=BRIEFING_BATCH_SIZE
    ):
        st.session_state.autotask_message = f"Admin Auto-Task: Generating {len(batch_indices)} LEGACY missing Briefings..."
        
def auto_generate_briefings_manual(batch_size: int):
    # ... (Unchanged) ...
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

def handle_bulk_audio_fix():
    # ... (Unchanged, ensures total_word_count is updated) ...
    words_to_fix_indices = [i for i, d in enumerate(st.session_state.vocab_data) if d.get('audio_base64') is None]
    
    if not words_to_fix_indices:
        st.success("All loaded words already have pronunciation audio!")
        return

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
                if update_word_in_firestore(word_data, fields_to_update):
                    st.session_state.vocab_data[index].update(fields_to_update)
                    fixed_count += 1
                else:
                    st.warning(f"Audio fixed for {word}, but save to Firebase failed.")
    
    if fixed_count > 0:
        st.session_state.total_word_count = get_word_count_from_firestore() # Refresh count
        st.success(f"✅ Bulk fix complete! Successfully repaired audio for {fixed_count} of {total_count} words.")
    else:
        st.error(f"🔴 Bulk fix attempted, but audio generation failed for all {total_count} words or failed to save to Firebase. Check server logs/quotas.")
        
    status_placeholder.empty()
    st.rerun()

# ... (Other UI/Quiz helpers are unchanged) ...


# ======================================================================
# 6. UI COMPONENTS (Adjusted for Lazy Loading)
# ======================================================================

def data_board_ui():
    """Displays key metrics and the status of background tasks."""
    
    if not st.session_state.is_auth or st.session_state.vocab_data is None:
        return
    
    word_count = st.session_state.total_word_count
    # These integrity checks only apply to the currently *loaded* words in vocab_data
    missing_audio_count = len([d for d in st.session_state.vocab_data if d.get('audio_base64') is None])
    missing_briefing_count = len([d for d in st.session_state.vocab_data if not d.get('briefing_audio_base64')])
    
    st.header("📊 Application Status Board")
    
    cols = st.columns(4)
    
    with cols[0]:
        st.metric(label="Total Words (DB)", value=word_count, delta=f"Target: {REQUIRED_WORD_COUNT}")
    with cols[1]:
        st.metric(label="Words Loaded (RAM)", value=len(st.session_state.vocab_data)) # Show loaded count
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
        st.info("No vocabulary loaded yet. Please check the Data Tools tab to generate the first batch.")
        return

    total_loaded_words = len(st.session_state.vocab_data)
    total_db_words = st.session_state.total_word_count

    # Calculate page bounds based ONLY on the data currently loaded in memory
    start_index = st.session_state.current_page_index * LOAD_BATCH_SIZE
    end_index = min(start_index + LOAD_BATCH_SIZE, total_loaded_words)
    
    # Check if the page index exceeds loaded data bounds
    if start_index >= total_loaded_words and total_loaded_words > 0:
        # User somehow navigated past the loaded data, reset to the last loaded page
        st.session_state.current_page_index = max(0, (total_loaded_words // LOAD_BATCH_SIZE) - 1)
        st.rerun()
        return

    
    st.markdown(f"**Showing Words {start_index + 1} - {end_index} of {total_db_words} High-Level SAT Words** (Loaded: {total_loaded_words})")
    
    
    # --- WORD DISPLAY CONTAINER ---
    with st.container(border=True): 
        
        for i, data in enumerate(st.session_state.vocab_data[start_index:end_index]):
            word_number = start_index + i + 1 
            word = data.get('word', 'N/A').upper()
            pronunciation = data.get('pronunciation', 'N/A')
            
            expander_title = f"**{word_number}. {word}** — {pronunciation}" 
            
            with st.expander(expander_title):
                # ... (Rest of expander content - unchanged) ...
                definition = data.get('definition', 'N/A')
                tip = data.get('tip', 'N/A')
                usage = data.get('usage', 'N/A')
                audio_base64 = data.get('audio_base64') 

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
                    if st.session_state.is_admin:
                        st.button(
                            f"Fix Audio for #{word_number}", 
                            key=f"fix_audio_{start_index + i}", 
                            on_click=handle_fix_single_audio, 
                            args=(start_index + i,),
                            type="primary",
                            disabled=st.session_state.autotask_running
                        )

                st.markdown(f"**📖 Definition:** {definition.capitalize()}") 
                st.markdown(f"**💡 Memory Tip:** *{tip}*") 
                st.markdown(f"**🗣️ Usage:** *'{usage}'*") 
            
            if i < LOAD_BATCH_SIZE - 1 and (start_index + i + 1) < total_loaded_words:
                 st.markdown("---") # Visually separate words

    # --- PAGINATION CONTROLS ---
    col_prev, col_status, col_next = st.columns([1, 2, 1])
    
    with col_prev:
        if st.session_state.current_page_index > 0:
            st.button("⬅️ Previous 10 Words", on_click=go_to_prev_page, disabled=st.session_state.autotask_running)
    
    with col_status:
        current_page = st.session_state.current_page_index + 1
        # Max loaded pages
        max_loaded_pages = (total_loaded_words + LOAD_BATCH_SIZE - 1) // LOAD_BATCH_SIZE
        st.markdown(f"<div style='text-align: center; padding-top: 10px;'>Page {current_page} of ~{max_loaded_pages} (loaded)</div>", unsafe_allow_html=True)

    with col_next:
        # Check if we can go to the next page, EITHER because it's already loaded OR because there is more data in DB
        can_go_next = (end_index < total_loaded_words) or st.session_state.has_more_data
        
        button_label = "Next 10 Words ➡️"
        if end_index == total_loaded_words and st.session_state.has_more_data:
            button_label = "Fetch Next Batch ➡️"
            
        if can_go_next:
            st.button(button_label, on_click=go_to_next_page, type="secondary", disabled=st.session_state.autotask_running)


# ======================================================================
# 7. STREAMLIT APPLICATION STRUCTURE (MAIN)
# ======================================================================

def main():
    """The main Streamlit application function."""
    st.set_page_config(page_title="AI Vocabulary Builder", layout="wide")
    st.title("🧠 AI-Powered Vocabulary Builder")
    
    initialize_session_state()
    
    # Check for completed background task at the very beginning of the rerun
    if st.session_state.is_admin:
        task_controller.check_task_status()
    
    # --- Sidebar for Auth Status ---
    with st.sidebar:
        st.header("User Login")
        
        if not st.session_state.is_auth:
            # ... (Login UI - unchanged) ...
            st.markdown("##### New User Registration / Existing User Login")
            
            user_email = st.text_input("📧 Email", key="user_email_input", value=st.session_state.current_user_email or "")
            password = st.text_input("🔑 Password", type="password", key="password_input")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Login", key="login_btn", type="primary"):
                    handle_auth(user_email, password)
            with col2:
                if st.button("Register", key="register_btn"):
                    handle_auth(user_email, password)
            
            st.markdown("---")
            st.markdown(f"""
            **Admin Login:** `{ADMIN_EMAIL}` / `Jamshaid,1981`
            
            **Note:** Use any email/6+ char password for general access.
            """)
            
        else:
            display_name = "Admin" if st.session_state.is_admin else st.session_state.current_user_email
            st.success(f"Logged in as: **{display_name}**")
            
            if st.button("Log Out", on_click=handle_logout):
                pass
                
    # --- Main Content ---
    
    # 🛑 Load data if logged in but data is not in session state (Only loads initial batch now)
    if st.session_state.is_auth and st.session_state.vocab_data is None:
        load_and_update_vocabulary_data() 
        st.rerun() 
    
    if not st.session_state.is_auth:
        st.info("Please log in or register using the sidebar to access the Vocabulary Builder.")
    else:
        # 2. RUN AUTO TASKS (Triggers non-blocking background process for Admin)
        if st.session_state.is_admin and st.session_state.initial_load_done:
            # Auto-fetch if data is low
            if (st.session_state.vocab_data is None or len(st.session_state.vocab_data) < AUTO_FETCH_THRESHOLD) and not st.session_state.autotask_running:
                 handle_admin_extraction_button(AUTO_FETCH_BATCH, auto_fetch=True)
            # Auto-briefing for legacy words
            auto_generate_briefings() 

        # 3. DISPLAY DATA BOARD
        data_board_ui()

        # 4. DISPLAY TABS
        tab_display, tab_quiz, tab_drill, tab_admin = st.tabs([
            "📚 Vocabulary List", 
            "📝 Quiz Section", 
            "⏱️ 2-Minute Drill",
            "🛠️ Data Tools"
        ])
        
        with tab_display:
            display_vocabulary_ui()
            
        with tab_quiz:
            generate_quiz_ui()
            
        with tab_drill:
            two_minute_drill_ui()

        with tab_admin:
            admin_extraction_ui()

if __name__ == "__main__":
    main()
