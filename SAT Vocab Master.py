import json
import time
import random
import os
import base64
from typing import List, Dict, Optional
import streamlit as st
from pydantic import BaseModel, Field
from pydantic import json_schema

# --- EXTERNAL API IMPORTS ---
try:
    from firebase_admin import credentials, initialize_app, firestore
    import firebase_admin
except ImportError:
    st.error("FIREBASE ERROR: The required library 'firebase-admin' is likely missing.")
    st.stop()

try:
    from gtts import gTTS
    import io
except ImportError:
    st.error("ERROR: The 'gtts' library is required for open-source TTS.")
    st.stop()
    
try:
    from google import genai
    from google.genai import types
except ImportError:
    st.error("ERROR: The 'google-genai' and 'pydantic' libraries are required.")
    st.stop()


# ======================================================================
# 1. SETUP & INITIALIZATION
# ======================================================================

# Check for API Key (Gemini)
if "GEMINI_API_KEY" not in os.environ:
    st.error("🔴 GEMINI_API_KEY is missing! Please set it in your Streamlit Secrets.")
    st.stop()

# Initialize Gemini Client
try:
    # API key is automatically picked up from the environment
    gemini_client = genai.Client()
except Exception as e:
    st.error(f"🔴 Failed to initialize Gemini Client: {e}")
    st.stop()

# Aggressive secret cleaning and Firebase initialization
try:
    secret_value = os.environ["FIREBASE_SERVICE_ACCOUNT"]
    cleaned_value = secret_value.strip().strip("'").strip('"')
    
    # Check and clean potential triple quotes from Streamlit Secrets formatting
    if cleaned_value.startswith('"""') and cleaned_value.endswith('"""'):
        cleaned_value = cleaned_value[3:-3].strip()

    service_account_info = json.loads(cleaned_value)
    
    if 'private_key' in service_account_info:
        raw_key = service_account_info['private_key']
        # Normalize private key format (common issue with environment variables)
        cleaned_key = raw_key.strip()
        cleaned_key = cleaned_key.replace("-----BEGINPRIVATEKEY-----", "-----BEGIN PRIVATE KEY-----")
        cleaned_key = cleaned_key.replace("-----ENDPRIVATEKEY-----", "-----END PRIVATE KEY-----")
        service_account_info['private_key'] = cleaned_key

    if not firebase_admin._apps:
        cred = credentials.Certificate(service_account_info)
        initialize_app(cred, name="vocab_builder_app") 

    db = firestore.client() 
    VOCAB_COLLECTION = db.collection("sat_vocabulary")
    
except KeyError:
    st.error("🔴 FIREBASE SETUP FAILED: 'FIREBASE_SERVICE_ACCOUNT' secret not found. Data cannot be saved permanently.")
    st.stop()
except Exception as e:
    st.error(f"🔴 FIREBASE INITIALIZATION FAILED: {e}. Check service account key format.")
    st.stop()


# --- App State and Constants ---
REQUIRED_WORD_COUNT = 2000 
LOAD_BATCH_SIZE = 10 
QUIZ_SIZE = 5 
AUTO_FETCH_THRESHOLD = 50 
AUTO_FETCH_BATCH = 25 
BRIEFING_BATCH_SIZE = 10 
MANUAL_BRIEFING_BATCH = 50 

# Admin Configuration (Mock Login)
ADMIN_EMAIL = "roy.jamshaid@gmail.com" 
ADMIN_PASSWORD = "Jamshaid,1981" 
MANUAL_EXTRACT_BATCH = 50 

# Pydantic Schema for Vocabulary Word
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
    
    # PERMANENTLY STORED BRIEFING FIELDS (2-Minute Overview)
    briefing_text: Optional[str] = Field(default=None, description="The extended AI-generated briefing text.")
    briefing_audio_base64: Optional[str] = Field(default=None, description="Base64 encoded audio data for the briefing.")


# ======================================================================
# 2. DATA PERSISTENCE & STATE MANAGEMENT (FIREBASE FIRESTORE)
# ======================================================================

# Initialize Session State variables
if 'current_user_email' not in st.session_state: st.session_state.current_user_email = None
if 'is_auth' not in st.session_state: st.session_state.is_auth = False
if 'vocab_data' not in st.session_state: st.session_state.vocab_data = None 
if 'quiz_active' not in st.session_state: st.session_state.quiz_active = False
if 'current_page_index' not in st.session_state: st.session_state.current_page_index = 0
if 'quiz_start_index' not in st.session_state: st.session_state.quiz_start_index = 0
if 'is_admin' not in st.session_state: st.session_state.is_admin = False
if 'drill_word_index' not in st.session_state: st.session_state.drill_word_index = 0
if 'auto_briefing_done' not in st.session_state: st.session_state.auto_briefing_done = False
if 'initial_load_done' not in st.session_state: st.session_state.initial_load_done = False
if 'is_processing_autotask' not in st.session_state: st.session_state.is_processing_autotask = False
if 'autotask_message' not in st.session_state: st.session_state.autotask_message = None
if 'data_refresh_key' not in st.session_state: st.session_state.data_refresh_key = 0


@st.cache_data(show_spinner=False)
def get_all_vocabulary(cache_key: int) -> List[Dict]:
    """Fetches all vocabulary data from Firestore, optimized by Streamlit caching."""
    # The function runs only if the cache_key changes.
    print(f"--- FETCHING DATA: Cache Key {cache_key} changed/not found. Running Firestore query. ---")
    try:
        docs = VOCAB_COLLECTION.order_by('created_at').stream()
        vocab_list = [doc.to_dict() for doc in docs]
        return vocab_list
    except Exception as e:
        st.error(f"🔴 Firestore Load Failed: {e}")
        return []

def load_vocabulary_from_firestore() -> List[Dict]:
    """Wrapper for the cached function, passing the session state key."""
    return get_all_vocabulary(st.session_state.data_refresh_key)

def increment_data_refresh_key():
    """Forces a cache bust on the next data load."""
    st.session_state.data_refresh_key += 1

def save_word_to_firestore(word_data: Dict) -> bool:
    """Adds a single word document to the Firestore collection."""
    try:
        doc_ref = VOCAB_COLLECTION.document(word_data['word'].lower())
        doc_ref.set(word_data, merge=False)
        increment_data_refresh_key()
        return True
    except Exception as e:
        print(f"🔴 Firestore Save Failed for {word_data['word']}: {e}")
        return False
        
def update_word_in_firestore(word_data: Dict, fields_to_update: Dict) -> bool:
    """Updates specific fields of a word document in the Firestore collection."""
    try:
        doc_ref = VOCAB_COLLECTION.document(word_data['word'].lower())
        doc_ref.update(fields_to_update)
        increment_data_refresh_key()
        return True
    except Exception as e:
        print(f"🔴 Firestore Update Failed for {word_data['word']}: {e}")
        return False


# ======================================================================
# 3. AI EXTRACTION & AUDIO FUNCTIONS (FIXED CONTINUOUS FETCH)
# ======================================================================

def generate_tts_audio(text: str) -> Optional[str]:
    """Generates audio via gTTS and returns Base64 encoded MP3 data."""
    if not text:
        return None
        
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

def generate_full_briefing(word_data: Dict) -> Optional[Dict]:
    """
    Generates the detailed briefing text and its corresponding Base64 audio.
    Returns a dictionary of briefing fields or None on failure.
    """
    word = word_data.get('word', 'a high-level word')
    definition = word_data.get('definition', 'a complex meaning')
    
    # Prompt is designed for a concise 60-80 word briefing
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
        # 1. Generate Briefing Text (Gemini API)
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.7) 
        )
        briefing_text = response.text.strip()
            
        # 2. Generate Briefing Audio (gTTS)
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

def real_llm_vocabulary_extraction(num_words: int, existing_words: List[str]) -> List[Dict]:
    """
    Calls Gemini to generate base structured vocabulary and then synchronously
    generates word pronunciation and the FULL 2-Minute briefing and audio.
    This ensures all required data is fetched before saving.
    """
    
    prompt = f"Generate {num_words} unique, extremely high-level SAT vocabulary words. The words must NOT be any of the following: {', '.join(existing_words) if existing_words else 'none'}."

    list_schema = {"type": "array", "items": SatWord.model_json_schema()}
    config = types.GenerateContentConfig(response_mime_type="application/json", response_json_schema=list_schema)
    
    try:
        # 1. Get Base Word Data (Word, Definition, Tip, Usage)
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt, config=config
        )
        new_data_list = json.loads(response.text)
        validated_words = [SatWord(**item).model_dump() for item in new_data_list if 'word' in item]
    except Exception as e:
        print(f"Gemini Extraction Failed: {e}")
        return []
        
    final_words = []
    
    # 2. Add Pronunciation & 2-Minute Briefing (CRITICAL STEP)
    with st.spinner(f"Generating Pronunciation, Briefings, and saving {len(validated_words)} words... This is a slow, multi-step process."):
        for word_data in validated_words:
            
            # Part A: Pronunciation Audio (Word only)
            pronunciation_audio = generate_tts_audio(word_data['word'])
            word_data['audio_base64'] = pronunciation_audio if pronunciation_audio else None
            
            # Part B: 2-Minute Briefing (Long text + audio)
            briefing_content = generate_full_briefing(word_data)
            
            if briefing_content:
                word_data.update(briefing_content)
            else:
                # Still save the base word, but briefing is missing
                print(f"Warning: Briefing generation failed for {word_data['word']}. It will be tagged for legacy fix.")
            
            final_words.append(word_data)
        
    return final_words

def handle_manual_word_entry(word: str):
    """Generates all content for a single word and saves it to Firestore."""
    
    if not word:
        st.error("Please enter a word.")
        return

    st.info(f"Generating all content (Pronunciation, Definition, Briefing) for '{word}'...")
    
    # --- Part 1: Get Base Word Data via LLM ---
    prompt = f"Generate the pronunciation, definition, mnemonic tip, and a usage sentence for the high-level SAT word: {word}. Return only the JSON object."
    
    try:
        list_schema = {"type": "array", "items": SatWord.model_json_schema()}
        config = types.GenerateContentConfig(response_mime_type="application/json", response_json_schema=list_schema)
        
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt, config=config
        )
        
        data_list = json.loads(response.text)
        if not data_list or not isinstance(data_list, list) or 'word' not in data_list[0]:
            raise ValueError("LLM returned invalid data structure.")
            
        new_word_data = SatWord(**data_list[0]).model_dump()
        
    except Exception as e:
        st.error(f"🔴 Failed to generate base content for '{word}'. Error: {e}")
        return

    # --- Part 2: Get Pronunciation & Briefing (Synchronous) ---
    with st.spinner(f"Generating audio and briefing for '{word}'..."):
        # Pronunciation Audio
        pronunciation_audio = generate_tts_audio(new_word_data['word'])
        new_word_data['audio_base64'] = pronunciation_audio if pronunciation_audio else None
        
        # 2-Minute Briefing (Long text + audio)
        briefing_content = generate_full_briefing(new_word_data)
        if briefing_content:
            new_word_data.update(briefing_content)


    if new_word_data['word'].lower() != word.lower():
        st.warning(f"Note: LLM corrected the word to '{new_word_data['word']}'. Using LLM's version.")

    if save_word_to_firestore(new_word_data):
        st.success(f"✅ Successfully added '{new_word_data['word']}' with ALL content to Firebase!")
        st.rerun()
    else:
        st.error("🔴 Failed to save to Firebase.")

def handle_fix_single_audio(word_index: int):
    """Generates missing pronunciation audio for a single word and updates the Firestore document."""
    
    if word_index < 0 or word_index >= len(st.session_state.vocab_data):
        st.error("Invalid word index.")
        return
        
    word_data = st.session_state.vocab_data[word_index]
    word = word_data['word']
    
    st.info(f"Attempting to fix pronunciation for '{word}'...")
    
    audio_data = generate_tts_audio(word)
    
    if audio_data:
        fields_to_update = {'audio_base64': audio_data}
        if update_word_in_firestore(word_data, fields_to_update):
            st.session_state.vocab_data[word_index].update(fields_to_update)
            st.success(f"✅ Successfully fixed audio for '{word}' and saved to Firebase.")
        else:
            st.error(f"🔴 Audio generated, but failed to save update to Firebase for '{word}'.")
    else:
        st.error(f"🔴 Failed to fix audio for '{word}'. TTS generation may still be failing.")
    
    st.rerun()

def handle_bulk_audio_fix():
    """Attempts to generate and save missing pronunciation audio for all corrupted words."""
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
        increment_data_refresh_key()
        st.success(f"✅ Bulk fix complete! Successfully repaired audio for {fixed_count} of {total_count} words.")
    else:
        st.error(f"🔴 Bulk fix attempted, but audio generation failed for all {total_count} words or failed to save to Firebase. Check server logs/quotas.")
        
    status_placeholder.empty()
    st.rerun()

def handle_admin_extraction_button(num_words: int, auto_fetch: bool = False):
    """Handles the bulk word extraction (manual or auto-triggered)."""
    
    status_message = f"Automatically extracting {num_words} new words (Admin Only)..." if auto_fetch else f"Manually extracting {num_words} new words..."

    st.info(status_message)
    
    existing_words = [d['word'] for d in st.session_state.vocab_data]
    
    # 🛑 SLOW STEP: AI extraction and ALL content generation runs here
    new_batch = real_llm_vocabulary_extraction(num_words, existing_words) 
    
    if new_batch:
        successful_saves = 0
        for word_data in new_batch:
            if save_word_to_firestore(word_data):
                st.session_state.vocab_data.append(word_data)
                successful_saves += 1
                
        if not auto_fetch:
            st.success(f"✅ Added {successful_saves} words. Current total: {len(st.session_state.vocab_data)}.")
        st.session_state.auto_fetch_done = True
        st.rerun() 
    else:
        if not auto_fetch:
            st.error("🔴 Failed to generate new words. Check API key and logs.")

def auto_generate_briefings():
    """
    AUTO-FETCH: Admin background task to process LEGACY words missing the 2-minute briefing.
    """
    if not st.session_state.is_admin or st.session_state.auto_briefing_done or st.session_state.is_processing_autotask:
        return

    # Check for words missing the permanent briefing audio
    words_to_brief_indices = [
        i for i, d in enumerate(st.session_state.vocab_data) 
        if not d.get('briefing_audio_base64') 
    ]
    
    if not words_to_brief_indices:
        st.session_state.auto_briefing_done = True
        return

    st.session_state.is_processing_autotask = True
    batch_indices = words_to_brief_indices[:BRIEFING_BATCH_SIZE]
    
    st.session_state.autotask_message = f"Admin Auto-Task: Generating {len(batch_indices)} LEGACY missing Briefings..."
    
    generated_count = 0
    
    # Process the batch
    for index in batch_indices:
        word_data = st.session_state.vocab_data[index]
        
        # Use the combined briefing function
        briefing_content = generate_full_briefing(word_data)

        if briefing_content:
            # Update Firestore and Session State
            if update_word_in_firestore(word_data, briefing_content):
                st.session_state.vocab_data[index].update(briefing_content)
                generated_count += 1
            
    remaining_words_count = len(words_to_brief_indices) - generated_count
        
    st.session_state.is_processing_autotask = False
    
    if remaining_words_count > 0:
        st.session_state.autotask_message = f"✅ Auto-Briefing completed a batch of {generated_count}. Processing next LEGACY batch..."
        st.rerun() 
    else:
        st.session_state.auto_briefing_done = True
        st.session_state.autotask_message = f"✅ Auto-Briefing complete: All {len(st.session_state.vocab_data)} words now have briefings."
        st.rerun()

def auto_generate_briefings_manual(batch_size: int):
    """
    Manually triggers a large batch generation of missing briefing content 
    and forces a rerun to update the word counts in the Admin UI.
    """
    
    words_to_brief_indices = [
        i for i, d in enumerate(st.session_state.vocab_data)  
        if not d.get('briefing_audio_base64') 
    ]
    
    if not words_to_brief_indices:
        st.session_state.autotask_message = "All words already have 2-Minute Briefing content!"
        st.rerun()
        return

    # Select the first 'batch_size' words to process
    batch_indices = words_to_brief_indices[:batch_size]
    
    st.session_state.autotask_message = f"Status: Manually starting bulk generation for {len(batch_indices)} missing briefings (Batch Size {batch_size})..."
    
    generated_count = 0
    
    # Process the batch
    with st.spinner(f"Generating briefing text and audio for {len(batch_indices)} words..."):
        for index in batch_indices:
            word_data = st.session_state.vocab_data[index]
            
            # Use the combined briefing function
            briefing_content = generate_full_briefing(word_data)
            
            if briefing_content:
                # Update Firestore and Session State
                if update_word_in_firestore(word_data, briefing_content):
                    st.session_state.vocab_data[index].update(briefing_content)
                    generated_count += 1
            
    st.session_state.autotask_message = f"Manual Bulk Briefing complete: Generated {generated_count} briefings. Please wait for the page to refresh to see the updated count."
    
    # Force a rerun to reload state/data and update the displayed counts
    st.rerun()


def fill_missing_audio(vocab_data: List[Dict]) -> bool:
    """Checks for missing audio to display the correct warning/fix options."""
    words_to_fix = [d for d in vocab_data if d.get('audio_base64') is None]
    if not words_to_fix:
        return False

    st.warning(f"Audio Integrity Check: Found {len(words_to_fix)} words missing pronunciation. Use the 'Fix Audio' button next to each word or the 'Bulk Fix' tool.")
    
    return False 

def load_and_update_vocabulary_data():
    """
    Loads data into session state using the cached function.
    This runs on every relevant rerun, but only queries Firestore if the cache key is incremented.
    """
    if not st.session_state.is_auth: return
    
    # 🛑 Load data using the cached function (will query Firestore if data_refresh_key changed)
    vocab_list = load_vocabulary_from_firestore()
    st.session_state.vocab_data = vocab_list
    st.session_state.initial_load_done = True
    
    # Check for missing audio (non-blocking status message)
    fill_missing_audio(st.session_state.vocab_data)
        
    word_count = len(st.session_state.vocab_data)
    
    if word_count > 0:
        st.info(f"✅ Loaded {word_count} words from shared database (Firestore).")
    elif st.session_state.is_auth:
        st.info("Database is empty. Please use the 'Data Tools' tab to extract the first batch of words.")

    # 3. AUTO-FETCH LOGIC FOR ADMIN (Vocabulary Extraction)
    if st.session_state.is_admin and word_count < AUTO_FETCH_THRESHOLD and 'auto_fetch_done' not in st.session_state:
        handle_admin_extraction_button(AUTO_FETCH_BATCH, auto_fetch=True)
        return 


def handle_auth(action: str, email: str, password: str):
    """Handles Mock user registration and login."""
    if not email or not password:
        st.error("Please enter both Email and Password.")
        return
        
    if email == ADMIN_EMAIL and password == ADMIN_PASSWORD:
        is_admin = True
    elif len(password) >= 6 and '@' in email and '.' in email:
        is_admin = False
    else:
        st.error("Invalid credentials. Registration/Login requires a valid email and 6+ character password.")
        return

    st.session_state.current_user_email = email
    st.session_state.is_auth = True
    st.session_state.is_admin = is_admin
    st.session_state.current_page_index = 0
    st.session_state.quiz_start_index = 0
    st.session_state.drill_word_index = 0 
    st.session_state.auto_fetch_done = False 
    st.session_state.auto_briefing_done = False 
    st.session_state.autotask_message = "Logged in successfully. Starting data check..."
    
    # CRITICAL FIX: Increment key on login to guarantee a fresh data fetch
    increment_data_refresh_key()

    # 🛑 SYNCHRONOUS LOAD WITH VISUAL SPINNER 
    with st.spinner("Downloading all vocabulary records from Firestore... Please wait."):
        load_and_update_vocabulary_data() 
        
    st.rerun()
            
def handle_logout():
    """Handles session state reset."""
    st.session_state.is_auth = False
    st.session_state.current_user_email = None
    st.session_state.quiz_active = False
    st.session_state.is_admin = False
    st.session_state.current_page_index = 0
    st.session_state.quiz_start_index = 0
    st.session_state.drill_word_index = 0 
    st.session_state.auto_fetch_done = False
    st.session_state.auto_briefing_done = False 
    st.session_state.autotask_message = None
    st.session_state.data_refresh_key = 0
    st.rerun()

# ======================================================================
# 4. UI COMPONENTS: VOCABULARY, QUIZ, ADMIN
# ======================================================================

def go_to_next_page():
    """Advances the displayed word page index."""
    st.session_state.current_page_index += 1
    st.rerun()

def go_to_prev_page():
    """Decrements the displayed word page index."""
    st.session_state.current_page_index -= 1
    st.rerun()
    
def next_drill_word():
    """Advances the drill word index."""
    if st.session_state.drill_word_index < len(st.session_state.vocab_data):
        st.session_state.drill_word_index += 1
        st.rerun()

def prev_drill_word():
    """Decrements the displayed word page index."""
    if st.session_state.drill_word_index > 0:
        st.session_state.drill_word_index -= 1
        st.rerun()


def data_board_ui():
    """Displays key metrics and the status of background tasks."""
    
    if not st.session_state.is_auth or st.session_state.vocab_data is None:
        return
    
    word_count = len(st.session_state.vocab_data)
    missing_audio_count = len([d for d in st.session_state.vocab_data if d.get('audio_base64') is None])
    missing_briefing_count = len([d for d in st.session_state.vocab_data if not d.get('briefing_audio_base64')])
    
    st.header("📊 Application Status Board")
    
    cols = st.columns(4)
    
    with cols[0]:
        st.metric(label="Total Words", value=word_count, delta=f"Target: {REQUIRED_WORD_COUNT}")
    with cols[1]:
        st.metric(label="Words Missing Audio", value=missing_audio_count)
    with cols[2]:
        st.metric(label="Words Missing Briefing", value=missing_briefing_count)
    with cols[3]:
        status_message = st.session_state.get('autotask_message', "System Idle/Complete.")
        
        if "processing next batch" in status_message or "Generating" in status_message:
             st.info(f"Status: {status_message}")
        elif "complete" in status_message or "Idle" in status_message:
             st.success(f"Status: {status_message}")
        else:
             st.markdown(f"**Status:** {status_message}")
            
    st.markdown("---")


def display_vocabulary_ui():
    """Renders the Vocabulary Display feature with Paging functionality and improved styling."""
    st.header("📚 Vocabulary Display", divider="blue")
    
    if st.session_state.vocab_data is None or not st.session_state.vocab_data:
        st.info("No vocabulary loaded yet. Please check the Data Tools tab to generate the first batch.")
        return

    total_words = len(st.session_state.vocab_data)
    
    start_index = st.session_state.current_page_index * LOAD_BATCH_SIZE
    end_index = min(start_index + LOAD_BATCH_SIZE, total_words)
    
    words_to_show = end_index - start_index
    
    st.markdown(f"**Showing Words {start_index + 1} - {end_index} of {total_words} High-Level SAT Words**")
    
    
    # --- WORD DISPLAY CONTAINER ---
    with st.container(border=True): 
        
        # Display the words for the current page only
        for i, data in enumerate(st.session_state.vocab_data[start_index:end_index]):
            word_number = start_index + i + 1 
            word = data.get('word', 'N/A').upper()
            pronunciation = data.get('pronunciation', 'N/A')
            tip = data.get('tip', 'N/A')
            usage = data.get('usage', 'N/A')
            audio_base64 = data.get('audio_base64') 
            definition = data.get('definition', 'N/A')
            
            expander_title = f"**{word_number}. {word}** — {pronunciation}" 
            
            with st.expander(expander_title):
                
                # --- AUDIO PLAYBACK ---
                if audio_base64:
                    audio_data_url = f"data:audio/mp3;base64,{audio_base64}"
                    audio_html = f"""
                        <audio controls style="width: 100%;" src="{audio_data_url}">
                            Your browser does not support the audio element.
                        </audio>
                    """
                    st.markdown(audio_html, unsafe_allow_html=True)
                else:
                    st.warning("Audio not available for this word. TTS generation may have failed.")
                    
                    if st.session_state.is_admin:
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

    # --- PAGINATION CONTROLS ---
    col_prev, col_status, col_next = st.columns([1, 2, 1])
    
    with col_prev:
        if st.session_state.current_page_index > 0:
            st.button("⬅️ Previous 10 Words", on_click=go_to_prev_page)
    
    with col_status:
        current_page = st.session_state.current_page_index + 1
        total_pages = (total_words + LOAD_BATCH_SIZE - 1) // LOAD_BATCH_SIZE
        st.markdown(f"<div style='text-align: center; padding-top: 10px;'>Page {current_page} of {total_pages}</div>", unsafe_allow_html=True)

    with col_next:
        if end_index < total_words:
            st.button("Next 10 Words ➡️", on_click=go_to_next_page, type="secondary")


def start_new_quiz():
    """Initializes the quiz based only on the currently displayed words in sequential order."""
    start = st.session_state.quiz_start_index
    end = start + QUIZ_SIZE
    
    words_pool = st.session_state.vocab_data[start:end]
    
    if len(words_pool) < QUIZ_SIZE:
        st.error(f"Cannot start quiz. Need {QUIZ_SIZE} words starting from position {start + 1}.")
        return

    quiz_details = []
    all_definitions = [d['definition'].capitalize() for d in st.session_state.vocab_data]
    
    for question_data in words_pool:
        correct_answer = question_data['definition'].capitalize()
        
        # Select 3 unique decoys that aren't the correct answer
        decoys = random.sample([
            d for d in all_definitions if d != correct_answer
        ], min(3, len([d for d in all_definitions if d != correct_answer])))
        
        options = [correct_answer] + decoys
        random.shuffle(options)
        
        original_word_index = st.session_state.vocab_data.index(question_data) + 1
        
        quiz_details.append({
            "word": question_data['word'],
            "correct_answer": correct_answer,
            "tip": question_data['tip'],
            "usage": question_data['usage'],
            "options": options,
            "index": original_word_index
        })
        
    st.session_state.quiz_details = quiz_details
    st.session_state.quiz_active = True
    st.session_state.quiz_results = None 

def advance_quiz_index():
    st.session_state.quiz_start_index += QUIZ_SIZE
    st.session_state.quiz_active = False 

def generate_quiz_ui():
    """Renders the Quiz Section feature."""
    st.header("📝 Vocabulary Quiz", divider="green")
    
    total_words = len(st.session_state.vocab_data)
    
    if total_words < QUIZ_SIZE:
        st.info(f"A minimum of {QUIZ_SIZE} words is required to start a quiz. Current total: {total_words}")
        return

    start_word_num = st.session_state.quiz_start_index + 1
    end_word_num = min(st.session_state.quiz_start_index + QUIZ_SIZE, total_words)
    
    
    if not st.session_state.quiz_active:
        
        if start_word_num > total_words:
            st.info("You have completed all available quiz blocks! Resetting to start.")
            st.session_state.quiz_start_index = 0
            start_word_num = 1
            end_word_num = min(QUIZ_SIZE, total_words)
            
        st.markdown(f"**Current Quiz Block:** Words {start_word_num} through {end_word_num}.")
        
        st.button(
            f"Start Quiz on Words #{start_word_num} - #{end_word_num}", 
            on_click=start_new_quiz, 
            type="primary"
        )
        return
    
    # --- Results Display ---
    if st.session_state.quiz_results is not None:
        score = st.session_state.quiz_results['score']
        total = st.session_state.quiz_results['total']
        accuracy = st.session_state.quiz_results['accuracy']
        
        if score == total:
            st.balloons()
            st.success(f"🎉 Quiz Complete! Perfect Score! {score} out of {total} (Accuracy: {accuracy}%)")
        else:
            st.warning(f"Quiz Complete! Final Score: **{score}** out of **{total}** (Accuracy: {accuracy}%)")
            
        st.subheader("Review Your Answers")
        for i, result in enumerate(st.session_state.quiz_results['feedback']):
            st.markdown(f"#### **Word #{result['index']}: {result['word']}**") 
            st.markdown(f"**Your Answer:** {result['user_choice']}")
            st.markdown(f"**Correct Answer:** {result['correct_answer']}")
            
            if not result['is_correct']:
                st.markdown(f"**Memory Tip:** *{result['tip']}*")
                st.markdown(f"**Usage:** *'{result['usage']}'*")
            
            st.markdown("---")
            
        st.session_state.quiz_results = None 
        
        next_start_index = st.session_state.quiz_start_index + QUIZ_SIZE
        if next_start_index < total_words:
            st.button(f"Start Next Quiz Block (Words #{next_start_index + 1} - #{min(next_start_index + QUIZ_SIZE, total_words)})", on_click=advance_quiz_index, type="secondary")
        else:
            st.info("You have completed all available words in the database!")
            st.session_state.quiz_start_index = 0
            st.button("Restart Quiz from Word #1", on_click=advance_quiz_index, type="secondary")
            
        return
    
    # --- Active Quiz Form ---
    quiz_details = st.session_state.quiz_details
    
    with st.form(key="full_quiz_form"):
        st.subheader(f"Answer the following {QUIZ_SIZE} questions:")
        st.caption(f"Testing words **{start_word_num}** to **{end_word_num}**.")
        
        st.session_state.user_responses = [] 
        
        for i, q in enumerate(quiz_details):
            st.markdown(f"#### **Word #{q['index']}. Define: {q['word'].upper()}**") 
            
            user_choice = st.radio(
                "Select the correct definition:", 
                q['options'], 
                key=f"quiz_q_{i}", 
                index=None,
                label_visibility="collapsed"
            )
            st.session_state.user_responses.append(user_choice)

        submitted = st.form_submit_button("Submit All Answers")

        if submitted:
            final_score = 0
            feedback_list = []
            
            if any(response is None for response in st.session_state.user_responses):
                st.error("Please answer ALL questions before submitting.")
                return

            for i, response in enumerate(st.session_state.user_responses):
                q = quiz_details[i]
                is_correct = (response == q['correct_answer'])
                
                if is_correct:
                    final_score += 1
                
                feedback_list.append({
                    "word": q['word'],
                    "user_choice": response,
                    "correct_answer": q['correct_answer'],
                    "is_correct": is_correct,
                    "tip": q['tip'],
                    "usage": q['usage'],
                    "index": q['index']
                })
            
            st.session_state.quiz_results = {
                "score": final_score,
                "total": QUIZ_SIZE,
                "accuracy": round((final_score / QUIZ_SIZE) * 100, 1),
                "feedback": feedback_list
            }
            del st.session_state.user_responses
            st.rerun()

def two_minute_drill_ui():
    """Renders the UI for the 2-Minute Word Briefing feature using sequential navigation."""
    st.header("⏱️ 2-Minute Drill", divider="red")

    if st.session_state.vocab_data is None or not st.session_state.vocab_data:
        st.info("No vocabulary loaded yet. Please generate some words via the Data Tools tab.")
        return

    total_words = len(st.session_state.vocab_data)
    
    current_index = st.session_state.drill_word_index
    if current_index >= total_words:
        st.session_state.drill_word_index = 0
        current_index = 0
        st.rerun()
        
    selected_word_data = st.session_state.vocab_data[current_index]
    selected_word_str = selected_word_data['word'].upper()
    
    st.markdown(f"**Current Word:** **{current_index + 1}** of **{total_words}**")

    # Fetch permanent audio from the database
    briefing_text = selected_word_data.get('briefing_text')
    briefing_audio_base64 = selected_word_data.get('briefing_audio_base64')
    
    briefing_exists_in_db = bool(briefing_audio_base64)

    briefing = None
    
    if briefing_exists_in_db:
        briefing = {
            "text": briefing_text,
            "audio_base64": briefing_audio_base64 
        }
        st.success("Briefing content loaded from database.")
    
    # If content is not in DB, allow manual generation
    if not briefing_exists_in_db:
        st.warning(f"Briefing content missing for {selected_word_str}. Generate it now!")
        if st.button(f"Generate and Save Briefing for {selected_word_str}", type="primary", key="manual_drill_gen"):
            # Call the manual batch function with a size of 1 to process this single word immediately.
            auto_generate_briefings_manual(1) 
            st.rerun() 
    
    # --- Display Briefing Content ---
    if briefing:
        st.subheader(f"Deep Dive: {selected_word_str}")
        
        # Audio Player
        if briefing['audio_base64']:
            audio_data_url = f"data:audio/mp3;base64,{briefing['audio_base64']}"
            audio_html = f"""
                <audio controls style="width: 100%;" src="{audio_data_url}">
                    
                        Your browser does not support the audio element.
                </audio>
            """
            st.markdown(audio_html, unsafe_allow_html=True)
            st.markdown("---")
            
        # Text Content 
        st.markdown("##### 🔊 Full Briefing Transcript")
        st.markdown(briefing['text'])
        
        st.markdown("---")
        st.info(f"The briefing is about {len(briefing['text'].split())} words long, providing a rich, 2-minute study session.")
    
    # --- Navigation Buttons ---
    col_prev, col_next = st.columns([1, 1])
    
    with col_prev:
        if current_index > 0:
            st.button("⬅️ Previous Word", on_click=prev_drill_word)
    
    with col_next:
        if current_index < total_words - 1:
            st.button("Next Word ➡️", on_click=next_drill_word, type="secondary")
        elif total_words > 0:
            st.info("End of current word list reached.")


def admin_extraction_ui():
    """Renders the Admin Extraction and User Management feature."""
    st.header("💡 Data Tools", divider="orange") 
    
    if not st.session_state.is_admin:
        st.warning("You must be logged in as the Admin to use this tool.")
        return

    # --- Manual Single Word Entry ---
    st.subheader("Manual Word & All Content Entry (Vocabulary, Pronunciation, Briefing)")
    with st.form(key="manual_word_form"):
        manual_word = st.text_input("Enter SAT-Level Word to Add:", key="manual_word_input").strip()
        manual_submit = st.form_submit_button("Generate ALL Content (Slow)")
        
        if manual_submit:
            handle_manual_word_entry(manual_word)

    st.markdown("---")
    
    # --- BULK AUDIO & BRIEFING FIX SECTION ---
    st.subheader("Audio Integrity & Bulk Fix (Legacy Word Processing)")
    
    if st.session_state.vocab_data:
        missing_audio_count = len([d for d in st.session_state.vocab_data if d.get('audio_base64') is None])
        missing_briefing_count = len([d for d in st.session_state.vocab_data if not d.get('briefing_audio_base64')])
    else:
        missing_audio_count = 0
        missing_briefing_count = 0
        
    st.markdown(f"**Corrupted Entries (Pronunciation):** {missing_audio_count} words.")
    st.markdown(f"**Missing Briefings (2-Min Drill - Legacy):** {missing_briefing_count} words.") 

    col_audio_fix, col_briefing_gen = st.columns(2)
    
    with col_audio_fix:
        st.button(
            "Attempt Bulk Audio Fix (Fix All Missing Pronunciations)", 
            on_click=handle_bulk_audio_fix, 
            type="primary"
        )
    
    with col_briefing_gen:
        st.button(
            f"Force Generate {MANUAL_BRIEFING_BATCH} Missing Briefings", 
            on_click=lambda: auto_generate_briefings_manual(MANUAL_BRIEFING_BATCH), 
            type="secondary"
        )


    st.markdown("---")

    # --- Extraction Control (Bulk Word Generation) ---
    st.subheader("Vocabulary Extraction (Bulk - Generates ALL Content Simultaneously)")
    
    word_count = len(st.session_state.vocab_data) if st.session_state.vocab_data else 0
    st.markdown(f"**Total Words in Database:** `{word_count}` (Target: {REQUIRED_WORD_COUNT}).")
    st.warning("Note: This button now generates Pronunciation and 2-Minute Briefings in the same step. It will be slow.")

    if st.button(f"Force Extract {MANUAL_EXTRACT_BATCH} New Words", type="secondary"): 
        handle_admin_extraction_button(MANUAL_EXTRACT_BATCH, auto_fetch=False)

    st.markdown("---")
    
    # --- Manual Cache Control ---
    st.subheader("Manual Data Refresh (Cache Bust)")
    st.info(f"Current Cache Key: `{st.session_state.data_refresh_key}`. Increment to force a full data reload.")
    
    if st.button("Force Clear Cache and Reload Data from Firestore", type="danger", help="Use this if external changes aren't showing or data looks stale."):
        increment_data_refresh_key()
        st.session_state.vocab_data = None 
        st.rerun()


# ======================================================================
# 5. STREAMLIT APPLICATION STRUCTURE
# ======================================================================

def main():
    """The main Streamlit application function."""
    st.set_page_config(page_title="AI Vocabulary Builder", layout="wide")
    st.title("🧠 AI-Powered Vocabulary Builder")
    
    # --- Sidebar for Auth Status ---
    with st.sidebar:
        st.header("User Login")
        
        if not st.session_state.is_auth:
            
            st.markdown("##### New User Registration / Existing User Login")
            
            user_email = st.text_input("📧 Email", key="user_email_input", value=st.session_state.current_user_email or "")
            password = st.text_input("🔑 Password", type="password", key="password_input")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Login", key="login_btn", type="primary"):
                    handle_auth("login", user_email, password)
            with col2:
                if st.button("Register", key="register_btn"):
                    handle_auth("register", user_email, password)
            
            st.markdown("---")
            st.markdown(f"""
            **Admin Login:** `{ADMIN_EMAIL}` / `Jamshaid,1981`
            
            **Note:** Use any email/6+ char password to simulate general user access.
            """)
            
        else:
            display_name = "Admin" if st.session_state.is_admin else st.session_state.current_user_email
            st.success(f"Logged in as: **{display_name}**")
            
            if st.button("Log Out", on_click=handle_logout):
                pass
                
    # --- Main Content ---
    
    # 🛑 CHECK 1: Load data if logged in but data is not in session state
    if st.session_state.is_auth and st.session_state.vocab_data is None:
        load_and_update_vocabulary_data() 
        st.rerun()
    
    if not st.session_state.is_auth:
        st.info("Please log in or register using the sidebar to access the Vocabulary Builder.")
    else:
        # 2. RUN AUTO TASKS (Triggers non-blocking background process for Admin)
        if st.session_state.is_admin and st.session_state.initial_load_done:
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
