import json
import time
import random
import sys
import os
import base64
import urllib.parse 
import re 
from typing import List, Dict, Optional
import streamlit as st
from pydantic import BaseModel, Field, ValidationError
from pydantic import json_schema 

# 🟢 FINAL FIREBASE IMPORTS 
try:
    from firebase_admin import credentials, initialize_app, firestore 
    import firebase_admin 
except ImportError:
    st.error("FIREBASE ERROR: The required library 'firebase-admin' is likely missing in requirements.txt.")
    st.stop()


# 🟢 Import gTTS and io 
try:
    from gtts import gTTS
    import io
except ImportError:
    st.error("ERROR: The 'gtts' library is required for open-source TTS.")
    st.stop()
    
# --- GEMS API ---
try:
    from google import genai
    from google.genai import types
except ImportError:
    st.error("ERROR: The 'google-genai' and 'pydantic' libraries are required.")
    st.stop()


# ======================================================================
# *** FIREBASE SETUP & INITIALIZATION ***
# ======================================================================

# Check for API Key (Gemini)
if "GEMINI_API_KEY" in os.environ:
    api_key = os.environ["GEMINI_API_KEY"]
else:
    st.error("🔴 GEMINI_API_KEY is missing! Please set it in your Streamlit Secrets.")
    st.stop()

# Initialize Gemini Client
try:
    gemini_client = genai.Client()
except Exception as e:
    st.error(f"🔴 Failed to initialize Gemini Client: {e}")
    st.stop()

# 🟢 CRITICAL FIX: AGGRESSIVE SECRET CLEANING
try:
    # 1. Get the raw secret value (which may include unwanted characters from Streamlit's editor)
    secret_value = os.environ["FIREBASE_SERVICE_ACCOUNT"]

    # 2. AGGRESSIVE CLEANING: Strip all surrounding quotes, newlines, and tabs.
    cleaned_value = secret_value.strip()
    
    # Remove surrounding triple quotes
    if cleaned_value.startswith('"""') and cleaned_value.endswith('"""'):
        cleaned_value = cleaned_value[3:-3].strip()

    # Final strip of any remaining single/double quotes or whitespace on the ends
    cleaned_value = cleaned_value.strip().strip("'").strip('"')

    # 3. Attempt to load the cleaned string as JSON
    service_account_info = json.loads(cleaned_value)
    
    # 🛑 SAFE FIX FOR "Invalid private key" ERROR
    if 'private_key' in service_account_info:
        raw_key = service_account_info['private_key']
        
        # Only strip surrounding whitespace and fix header spacing
        cleaned_key = raw_key.strip()
        
        # Ensure the header and footer are correctly formatted (with spaces)
        cleaned_key = cleaned_key.replace("-----BEGINPRIVATEKEY-----", "-----BEGIN PRIVATE KEY-----")
        cleaned_key = cleaned_key.replace("-----ENDPRIVATEKEY-----", "-----END PRIVATE KEY-----")
        
        service_account_info['private_key'] = cleaned_key

    # 4. Initialize Firebase Admin SDK
    if not firebase_admin._apps:
        cred = credentials.Certificate(service_account_info)
        initialize_app(cred)

    db = firestore.client() 
    VOCAB_COLLECTION = db.collection("sat_vocabulary")
    
except KeyError:
    st.error("🔴 FIREBASE SETUP FAILED: 'FIREBASE_SERVICE_ACCOUNT' secret not found. Data cannot be saved permanently.")
    st.stop()
except Exception as e:
    st.error(f"🔴 FIREBASE INITIALIZATION FAILED: {e}. Check service account key format.")
    st.stop()


# --- App State and Constants ---
# 🟢 TARGET SIZE RESTORED TO 2000
REQUIRED_WORD_COUNT = 2000 
LOAD_BATCH_SIZE = 10 
AUTO_EXTRACT_TARGET_SIZE = REQUIRED_WORD_COUNT 
QUIZ_SIZE = 5 
# 🟢 Auto-fetching threshold (e.g., if less than 50 words, fetch 25)
AUTO_FETCH_THRESHOLD = 50 
AUTO_FETCH_BATCH = 25 
# 🟢 NEW: Batch size for auto-generating 2-Minute Briefings
BRIEFING_BATCH_SIZE = 5

# Admin Configuration (Mock Login)
ADMIN_EMAIL = "roy.jamshaid@gmail.com" 
ADMIN_PASSWORD = "Jamshaid,1981" 
MANUAL_EXTRACT_BATCH = 50 

# Pydantic Schema for Vocabulary Word
class SatWord(BaseModel):
    word: str = Field(description="The SAT-level word.")
    pronunciation: str = Field(description="Simple, hyphenated phonetic pronunciation (e.g., eh-FEM-er-al).")
    definition: str = Field(description="The concise dictionary definition.")
    tip: str = Field(description="A short, catchy mnemonic memory tip.")
    usage: str = Field(description="A professional sample usage sentence.")
    sat_level: str = Field(default="High", description="Should always be 'High'.")
    audio_base64: Optional[str] = Field(default=None, description="Base64 encoded audio data for pronunciation.")
    created_at: float = Field(default_factory=time.time)
    # 🟢 Fields for the 2-Minute Drill content (briefing_audio_base64 is NOW REMOVED from the schema)
    briefing_text: Optional[str] = Field(default=None, description="The extended AI-generated briefing text.")

# ----------------------------------------------------------------------
# 2. DATA PERSISTENCE & STATE MANAGEMENT (FIREBASE FIRESTORE)
# ----------------------------------------------------------------------

if 'current_user_email' not in st.session_state: st.session_state.current_user_email = None
if 'is_auth' not in st.session_state: st.session_state.is_auth = False
if 'vocab_data' not in st.session_state: st.session_state.vocab_data = []
if 'quiz_active' not in st.session_state: st.session_state.quiz_active = False
if 'current_page_index' not in st.session_state: st.session_state.current_page_index = 0
if 'quiz_start_index' not in st.session_state: st.session_state.quiz_start_index = 0
if 'is_admin' not in st.session_state: st.session_state.is_admin = False
# 🟢 State for the 2-Minute Drill feature 
if 'drill_word_index' not in st.session_state: st.session_state.drill_word_index = 0
# 🟢 NEW: Flag to track auto-briefing status
if 'auto_briefing_done' not in st.session_state: st.session_state.auto_briefing_done = False
# 🟢 NEW: Store generated briefing content temporarily with audio
if 'briefing_content_cache' not in st.session_state: st.session_state.briefing_content_cache = {}


def load_vocabulary_from_firestore():
    """Loads all vocabulary data from Firestore."""
    try:
        docs = VOCAB_COLLECTION.order_by('created_at').stream()
        vocab_list = [doc.to_dict() for doc in docs]
        return vocab_list
    except Exception as e:
        st.error(f"🔴 Firestore Load Failed: {e}")
        return []

def save_word_to_firestore(word_data: Dict):
    """Adds a single word document to the Firestore collection."""
    try:
        doc_ref = VOCAB_COLLECTION.document(word_data['word'].lower())
        # Use a subset of the fields for initial saving, exclude large briefing content
        data_to_save = {k: v for k, v in word_data.items() if k != 'briefing_audio_base64'}
        doc_ref.set(data_to_save, merge=False)
        return True
    except Exception as e:
        st.error(f"🔴 Firestore Save Failed for {word_data['word']}: {e}")
        return False
        
def update_word_in_firestore(word_data: Dict, fields_to_update: Optional[Dict] = None):
    """Updates a single word document in the Firestore collection."""
    try:
        doc_ref = VOCAB_COLLECTION.document(word_data['word'].lower())
        
        if fields_to_update:
            # 🛑 CRITICAL CHECK: Remove the large audio field if present before updating Firestore
            if 'briefing_audio_base64' in fields_to_update:
                del fields_to_update['briefing_audio_base64']
                
            doc_ref.update(fields_to_update)
        else:
            # Default update (used for audio fix in older version)
            doc_ref.update({
                'audio_base64': word_data['audio_base64']
            })
        return True
    except Exception as e:
        # Log the specific error to help with debugging
        st.error(f"🔴 Firestore Update Failed for {word_data['word']}: {e}")
        return False


# ----------------------------------------------------------------------
# 3. AI EXTRACTION & AUDIO FUNCTIONS
# ----------------------------------------------------------------------

def generate_tts_audio(text: str) -> Optional[str]:
    """Generates audio via gTTS and returns Base64 encoded MP3 data."""
    try:
        # gTTS has a max character limit, typically 5000. 
        tts = gTTS(text=text, lang='en', slow=False)
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        base64_data = base64.b64encode(mp3_fp.read()).decode('utf-8')
        return base64_data

    except Exception as e:
        print(f"gTTS Generation failed for text segment. Error: {e}")
        return None

def real_llm_vocabulary_extraction(num_words: int, existing_words: List[str]) -> List[Dict]:
    """
    Calls the Gemini API to generate structured vocabulary data and gTTS for audio.
    """
    
    prompt = f"Generate {num_words} unique, extremely high-level SAT vocabulary words. The words must NOT be any of the following: {', '.join(existing_words) if existing_words else 'none'}."

    list_schema = {"type": "array", "items": SatWord.model_json_schema()}
    config = types.GenerateContentConfig(response_mime_type="application/json", response_json_schema=list_schema)
    
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt, config=config
        )
        new_data_list = json.loads(response.text)
        validated_words = [SatWord(**item).model_dump() for item in new_data_list if 'word' in item]
    except Exception as e:
        print(f"Gemini Extraction Failed: {e}")
        return []
        
    words_with_audio = []
    
    # Use st.spinner for a single operation outside the extraction loop for simplicity
    with st.spinner(f"Generating audio and saving {len(validated_words)} words..."):
        for word_data in validated_words:
            word = word_data['word']
            audio_data = generate_tts_audio(word)
            word_data['audio_base64'] = audio_data if audio_data else None
            words_with_audio.append(word_data)
        
    return words_with_audio

# 🟢 CORE LLM FUNCTION: Generates detailed briefing (Used by both manual/auto)
def generate_word_briefing(word_data: Dict, word_index: int):
    """Generates a detailed 2-minute text briefing and its audio counterpart, saving text to Firestore."""
    word = word_data['word']
    definition = word_data['definition']
    
    # 🟢 MODIFIED PROMPT: Ensure the text is concise (approx 150 words/8-10 sentences) to guarantee it fits in Firestore.
    prompt = f"""
    You are a vocabulary tutor. Write a **concise, engaging, and persuasive briefing (8-10 sentences maximum)** on the word '{word}'. The text must be perfect for a short audio presentation.
    
    The briefing must seamlessly include:
    1. The core definition: {definition}.
    2. A brief note on its origin or etymology.
    3. Two complex example sentences demonstrating high-level usage.
    4. A final, memorable takeaway.
    
    Ensure the entire text is conversational and suitable for speech synthesis. Do not use bullet points or lists; write it as a continuous, flowing speech.
    """
    
    try:
        with st.spinner(f"Generating concise briefing text for '{word}'..."):
            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash", contents=prompt
            )
            briefing_text = response.text.strip()
            
        # --- Generate Audio for the entire briefing (DYNAMICALLY, NOT STORED) ---
        with st.spinner("Generating full audio for the briefing..."):
            # This generates the audio needed for immediate playback
            audio_data = generate_tts_audio(briefing_text)
            
        if not audio_data:
            raise Exception("Failed to generate TTS audio.")

        # 🛑 ONLY STORE TEXT IN FIREBASE (briefing_text)
        briefing_content_to_save = {
            "briefing_text": briefing_text
        }
        
        # 🟢 CRITICAL STEP: Save only the text content back to Firestore
        if update_word_in_firestore(word_data, briefing_content_to_save):
            # Update the session state data list with the new briefing content
            st.session_state.vocab_data[word_index].update(briefing_content_to_save)
            
            # Return full content (text + dynamic audio) for immediate display
            return {
                "text": briefing_text,
                "audio_base64": audio_data
            }
        else:
            st.error("Could not save briefing text to Firestore.")
            return None
        
    except Exception as e:
        st.error(f"🔴 Briefing Generation Failed for '{word}': {e}")
        return None

def handle_manual_word_entry(word: str):
    """Generates pronunciation and LLM content for a single word and saves it to Firestore."""
    
    if not word:
        st.error("Please enter a word.")
        return

    st.info(f"Generating data for '{word}'...")
    
    audio_data = generate_tts_audio(word)
    if not audio_data:
        st.error(f"🔴 Failed to generate pronunciation audio for '{word}'. Please check API key and retry.")
        return

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
        st.error(f"🔴 Failed to generate content for '{word}'. Error: {e}")
        return

    new_word_data['audio_base64'] = audio_data
    
    if new_word_data['word'].lower() != word.lower():
        st.warning(f"Note: LLM corrected the word to '{new_word_data['word']}'. Using LLM's version.")

    if save_word_to_firestore(new_word_data):
        st.success(f"✅ Manually added '{new_word_data['word']}' with working pronunciation to Firebase!")
        st.rerun()
    else:
        st.error("🔴 Failed to save to Firebase.")

def handle_fix_single_audio(word_index: int):
    """Generates missing audio for a single word and updates the Firestore document."""
    
    if word_index < 0 or word_index >= len(st.session_state.vocab_data):
        st.error("Invalid word index.")
        return
        
    word_data = st.session_state.vocab_data[word_index]
    word = word_data['word']
    
    st.info(f"Attempting to fix pronunciation for '{word}'...")
    
    audio_data = generate_tts_audio(word)
    
    if audio_data:
        # Pass update dictionary explicitly
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
    """
    Scans all loaded vocabulary data and attempts to generate and save missing audio
    for every word that is currently corrupted (audio_base64 is None).
    """
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
            time.sleep(0.1) # Prevents hitting API limits too fast
    
    
    if fixed_count > 0:
        st.success(f"✅ Bulk fix complete! Successfully repaired audio for {fixed_count} of {total_count} words.")
    else:
        st.error(f"🔴 Bulk fix attempted, but audio generation failed for all {total_count} words or failed to save to Firebase. Check server logs/quotas.")
        
    status_placeholder.empty()
    st.rerun()

def handle_admin_extraction_button(num_words: int, auto_fetch: bool = False):
    """Handles the bulk word extraction (manual or auto-triggered)."""
    
    if auto_fetch:
        status_message = f"Automatically extracting {num_words} new words (Admin Only)..."
    else:
        status_message = f"Manually extracting {num_words} new words..."

    st.info(status_message)
    
    existing_words = [d['word'] for d in st.session_state.vocab_data]
    
    # 🛑 SLOW STEP: AI extraction and TTS generation runs here
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
    Scans for words missing briefing content and auto-generates for a batch.
    This is designed to be called by the Admin user on app load.
    """
    if not st.session_state.is_admin or st.session_state.auto_briefing_done:
        return

    # Check only for words missing the *text*, as the audio is dynamic
    words_to_brief_indices = [
        i for i, d in enumerate(st.session_state.vocab_data) 
        if not d.get('briefing_text')
    ]
    
    if not words_to_brief_indices:
        st.session_state.auto_briefing_done = True
        return

    # Select the first BRIEFING_BATCH_SIZE words to process
    batch_indices = words_to_brief_indices[:BRIEFING_BATCH_SIZE]
    
    st.warning(f"Admin Auto-Task: Generating {len(batch_indices)} missing 2-Minute Briefings...")
    
    generated_count = 0
    
    # Process the batch
    for index in batch_indices:
        word_data = st.session_state.vocab_data[index]
        
        # NOTE: generate_word_briefing handles LLM call, TTS, and saving back to Firestore/session state
        briefing = generate_word_briefing(word_data, index)
        
        if briefing:
            generated_count += 1
        
        # Add a small delay to respect API limits
        time.sleep(1)
        
    st.info(f"Auto-Briefing complete: Generated {generated_count} briefings. Refresh the app to continue the auto-task.")
    
    # Set the flag to true only if the list is empty (or we stop auto-running)
    if generated_count == 0 or len(words_to_brief_indices) <= BRIEFING_BATCH_SIZE:
        st.session_state.auto_briefing_done = True
        
    # Force a rerun to reload state/data and potentially kick off the next batch
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
    Loads data and manages the auto-fetch process for the Admin user.
    """
    if not st.session_state.is_auth: return
    
    st.session_state.vocab_data = load_vocabulary_from_firestore()
    
    fill_missing_audio(st.session_state.vocab_data)
        
    word_count = len(st.session_state.vocab_data)
    
    if word_count > 0:
        st.info(f"✅ Loaded {word_count} words from shared database (Firestore).")
    elif st.session_state.is_auth:
        st.info("Database is empty. Please use the 'Data Tools' tab to extract the first batch of words.")

    # 🟢 AUTO-FETCH LOGIC FOR ADMIN (Vocabulary Extraction)
    if st.session_state.is_admin and word_count < AUTO_FETCH_THRESHOLD and 'auto_fetch_done' not in st.session_state:
        # Prevents re-running, runs only once if under threshold
        handle_admin_extraction_button(AUTO_FETCH_BATCH, auto_fetch=True)
        return # Stop loading here to allow extraction to complete

    # 🟢 AUTO-FETCH LOGIC FOR ADMIN (Briefing Generation)
    if st.session_state.is_admin and 'auto_fetch_done' in st.session_state and not st.session_state.auto_briefing_done:
        # Only start briefing if base vocabulary fetch is done
        auto_generate_briefings()
        # auto_generate_briefings() calls rerun internally if it ran.

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
    st.session_state.current_page_index = 0 # Reset word display page
    st.session_state.quiz_start_index = 0
    st.session_state.drill_word_index = 0 # 🟢 Reset drill word index
    st.session_state.auto_fetch_done = False # Reset auto-fetch flag
    st.session_state.auto_briefing_done = False # 🟢 Reset auto-briefing flag
    
    display_name = "Admin" if is_admin else email
    st.success(f"Logged in as: {display_name}! Access granted (Simulated).")
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
    st.session_state.drill_word_index = 0 # 🟢 Reset drill word index
    st.session_state.auto_fetch_done = False
    st.session_state.auto_briefing_done = False # 🟢 Reset auto-briefing flag
    st.rerun()

# ----------------------------------------------------------------------
# 4. UI COMPONENTS: VOCABULARY, QUIZ, ADMIN
# ----------------------------------------------------------------------

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
    if st.session_state.drill_word_index < len(st.session_state.vocab_data) - 1:
        st.session_state.drill_word_index += 1
        st.session_state.briefing_content_cache = {} # Clear cache
        st.rerun()

def prev_drill_word():
    """Decrements the drill word index."""
    if st.session_state.drill_word_index > 0:
        st.session_state.drill_word_index -= 1
        st.session_state.briefing_content_cache = {} # Clear cache
        st.rerun()

def display_vocabulary_ui():
    """Renders the Vocabulary Display feature with Paging functionality and improved styling."""
    st.header("📚 Vocabulary Display", divider="blue")
    
    if not st.session_state.vocab_data:
        st.info("No vocabulary loaded yet. Please check the Data Tools tab to generate the first batch.")
        return

    total_words = len(st.session_state.vocab_data)
    
    # Calculate start and end indices for the current page
    start_index = st.session_state.current_page_index * LOAD_BATCH_SIZE
    end_index = min(start_index + LOAD_BATCH_SIZE, total_words)
    
    words_to_show = end_index - start_index
    
    st.markdown(f"**Showing Words {start_index + 1} - {end_index} of {total_words} High-Level SAT Words**")
    
    
    # --- WORD DISPLAY CONTAINER (Removed fixed height for better vertical flow) ---
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
            
            expander_title = f"**{word_number}. {word}** — {pronunciation}" # Enhanced visual title
            
            with st.expander(expander_title):
                
                # --- AUDIO PLAYBACK (USES BASE64 MP3 DATA from gTTS) ---
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
    
    # Previous Button
    with col_prev:
        if st.session_state.current_page_index > 0:
            st.button("⬅️ Previous 10 Words", on_click=go_to_prev_page)
    
    # Status
    with col_status:
        # Displaying the current page number
        current_page = st.session_state.current_page_index + 1
        total_pages = (total_words + LOAD_BATCH_SIZE - 1) // LOAD_BATCH_SIZE
        st.markdown(f"<div style='text-align: center; padding-top: 10px;'>Page {current_page} of {total_pages}</div>", unsafe_allow_html=True)

    # Next Button
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

# 🟢 MODIFIED UI: 2-Minute Drill feature now uses index navigation
def two_minute_drill_ui():
    """Renders the UI for the 2-Minute Word Briefing feature using sequential navigation."""
    st.header("⏱️ 2-Minute Drill", divider="red")

    if not st.session_state.vocab_data:
        st.info("No vocabulary loaded yet. Please generate some words via the Data Tools tab.")
        return

    total_words = len(st.session_state.vocab_data)
    
    # Safely get the current word data based on the index
    current_index = st.session_state.drill_word_index
    if current_index >= total_words:
        st.session_state.drill_word_index = 0
        current_index = 0
        st.rerun()
    
    selected_word_data = st.session_state.vocab_data[current_index]
    selected_word_str = selected_word_data['word'].upper()
    
    st.markdown(f"**Current Word:** **{current_index + 1}** of **{total_words}**")

    # Check if briefing already exists in the database record (only check for text)
    briefing_text = selected_word_data.get('briefing_text')
    briefing_exists_in_db = bool(briefing_text)

    # --- Generation/Display Logic ---
    
    briefing = None
    
    if briefing_exists_in_db:
        # Check cache first for dynamic audio
        cache_key = selected_word_str
        if cache_key in st.session_state.briefing_content_cache:
            briefing = st.session_state.briefing_content_cache[cache_key]
            st.success("Briefing loaded from cache.")
        else:
            # 🟢 Generate audio dynamically if text exists in DB but audio is not in cache
            with st.spinner(f"Generating audio for {selected_word_str}..."):
                audio_data = generate_tts_audio(briefing_text)
            
            if audio_data:
                briefing = {
                    "text": briefing_text,
                    "audio_base64": audio_data
                }
                st.session_state.briefing_content_cache[cache_key] = briefing
                st.info("Briefing audio generated dynamically.")
            else:
                st.error("Could not generate audio dynamically. Try again.")

    
    # If content is not in DB AND the user clicks the manual generate button (for missing ones)
    if not briefing_exists_in_db:
        st.warning(f"Briefing content missing for {selected_word_str}. Generate it now!")
        if st.button(f"Generate and Save Briefing for {selected_word_str}", type="primary", key="manual_drill_gen"):
            # Generate and Save content to the database (only text)
            # This function returns the full briefing content (text + dynamic audio)
            briefing = generate_word_briefing(selected_word_data, current_index)
            if briefing:
                st.session_state.briefing_content_cache[selected_word_str] = briefing
                st.toast(f"Briefing generated and saved for {selected_word_str}!")
            else:
                st.error("Could not generate or save content.")
            st.rerun() # Rerun to display the content
    
    # --- Display Briefing Content ---
    if briefing:
        st.subheader(f"Deep Dive: {selected_word_str}")
        
        # Audio Player - FIX APPLIED: REMOVED 'autoplay'
        if briefing['audio_base64']:
            audio_data_url = f"data:audio/mp3;base64,{briefing['audio_base64']}"
            audio_html = f"""
                <audio controls style="width: 100%; margin-bottom: 15px;" src="{audio_data_url}">
                    Your browser does not support the audio element.
                </audio>
            """
            st.markdown(audio_html, unsafe_allow_html=True)
            st.markdown("---")
            
        # Text Content (with improved formatting)
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
    st.subheader("Manual Word & Pronunciation Entry")
    with st.form(key="manual_word_form"):
        manual_word = st.text_input("Enter SAT-Level Word to Add:", key="manual_word_input").strip()
        manual_submit = st.form_submit_button("Generate Pronunciation & Content")
        
        if manual_submit:
            handle_manual_word_entry(manual_word)

    st.markdown("---")
    
    # 🟢 NEW: BULK AUDIO FIX SECTION
    st.subheader("Audio Integrity & Bulk Fix")
    st.markdown(f"**Corrupted Entries (Pronunciation):** {len([d for d in st.session_state.vocab_data if d.get('audio_base64') is None])} words.")
    st.markdown(f"**Missing Briefings (2-Min Drill):** {len([d for d in st.session_state.vocab_data if not d.get('briefing_text')])} words.") 

    st.button(
        "Attempt Bulk Audio Fix (Fix All Missing Pronunciations)", 
        on_click=handle_bulk_audio_fix, 
        type="primary"
    )

    st.markdown("---")

    # --- Extraction Control (Manual Only) ---
    st.subheader("Extraction Control")
    
    # Check if auto-fetch is done or not needed
    if len(st.session_state.vocab_data) >= AUTO_EXTRACT_TARGET_SIZE: # Check against 2000
        st.success(f"Database contains {len(st.session_state.vocab_data)} words. Auto-fetch target ({AUTO_EXTRACT_TARGET_SIZE}) has been reached.")
    else:
        st.info(f"Database contains {len(st.session_state.vocab_data)} words. Automatically fetching new words will trigger if the count is below the threshold ({AUTO_FETCH_THRESHOLD}).")

    st.markdown("---")
    
    # --- Manual Extraction Override (Admin Only) ---
    st.subheader("Vocabulary Extraction (Bulk)")
    st.markdown(f"**Total Words in Database:** `{len(st.session_state.vocab_data)}` (Target: {REQUIRED_WORD_COUNT}).")

    if st.button(f"Force Extract {MANUAL_EXTRACT_BATCH} New Words", type="secondary"): 
        handle_admin_extraction_button(MANUAL_EXTRACT_BATCH, auto_fetch=False)


# ----------------------------------------------------------------------
# 5. STREAMLIT APPLICATION STRUCTURE
# ----------------------------------------------------------------------

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
    
    if not st.session_state.is_auth:
        st.info("Please log in or register using the sidebar to access the Vocabulary Builder.")
    else:
        # Load data on successful login (Fast operation)
        if not st.session_state.vocab_data:
            load_and_update_vocabulary_data() 
        else:
            # 🟢 Run auto-briefing on subsequent loads if the first fetch is done
            load_and_update_vocabulary_data() 


        # Non-blocking status message
        if len(st.session_state.vocab_data) < AUTO_EXTRACT_TARGET_SIZE:
             st.info(f"The vocabulary list currently has {len(st.session_state.vocab_data)} words. The target is {AUTO_EXTRACT_TARGET_SIZE}. Use the Admin 'Data Tools' tab to extract more.")

        # 🟢 UPDATED TABS: Added '2-Minute Drill' tab
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
