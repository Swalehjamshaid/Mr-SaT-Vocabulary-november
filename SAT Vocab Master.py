import json
import time
import random
import sys
import os
import base64
import urllib.parse 
from typing import List, Dict, Optional
import streamlit as st
from pydantic import BaseModel, Field, ValidationError
from pydantic import json_schema 

# --- GEMS API ---
try:
    from google import genai
    from google.genai import types
except ImportError:
    st.error("ERROR: The 'google-genai' and 'pydantic' libraries are required.")
    st.error("Please run: pip install google-genai pydantic")
    st.stop()


# ======================================================================
# *** LOCAL EXECUTION SETUP & FILE PATHS ***
# ======================================================================

# Check for API Key (Gemini)
if "GEMINI_API_KEY" in os.environ:
    # Use the API key provided in the secrets file
    api_key = os.environ["GEMINI_API_KEY"]
else:
    st.error("🔴 GEMINI_API_KEY is missing! Please set it in your secrets.")
    st.stop()

# Initialize Gemini Client
try:
    gemini_client = genai.Client()
except Exception as e:
    st.error(f"🔴 Failed to initialize Gemini Client: {e}")
    st.stop()

# --- App State and Constants ---
# Use a local JSON file for persistent storage
JSON_FILE_PATH = "vocab_data.json" 
REQUIRED_WORD_COUNT = 2000
LOAD_BATCH_SIZE = 10 
AUTO_EXTRACT_TARGET_SIZE = 200 # Target for automatic extraction attempts
QUIZ_SIZE = 5 

# Admin Configuration (Mock Login)
ADMIN_EMAIL = "roy.jamshaid@gmail.com" 
ADMIN_PASSWORD = "Jamshaid,1981" 
# Manual extraction batch size reduced to 5 for stable TTS generation
MANUAL_EXTRACT_BATCH = 5 


# Pydantic Schema for Vocabulary Word
class SatWord(BaseModel):
    word: str = Field(description="The SAT-level word.")
    pronunciation: str = Field(description="Simple, hyphenated phonetic pronunciation (e.g., eh-FEM-er-al).")
    definition: str = Field(description="The concise dictionary definition.")
    tip: str = Field(description="A short, catchy mnemonic memory tip.")
    usage: str = Field(description="A professional sample usage sentence.")
    sat_level: str = Field(default="High", description="Should always be 'High'.")
    # The audio field stores the Base64-encoded audio data (PCM WAV format)
    audio_base64: Optional[str] = Field(default=None, description="Base64 encoded audio data for pronunciation.")

# ----------------------------------------------------------------------
# 2. DATA PERSISTENCE & STATE MANAGEMENT (LOCAL FILE)
# ----------------------------------------------------------------------

if 'current_user_email' not in st.session_state: st.session_state.current_user_email = None
if 'is_auth' not in st.session_state: st.session_state.is_auth = False
if 'vocab_data' not in st.session_state: st.session_state.vocab_data = []
if 'quiz_active' not in st.session_state: st.session_state.quiz_active = False
if 'words_displayed' not in st.session_state: st.session_state.words_displayed = LOAD_BATCH_SIZE
if 'quiz_start_index' not in st.session_state: st.session_state.quiz_start_index = 0
if 'is_admin' not in st.session_state: st.session_state.is_admin = False

def load_vocabulary_from_file():
    """Loads vocabulary data from the local JSON file."""
    if os.path.exists(JSON_FILE_PATH):
        with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                return data if isinstance(data, list) else []
            except json.JSONDecodeError:
                return []
    return []

def save_vocabulary_to_file(data: List[Dict]):
    """Saves the current vocabulary data to the local JSON file."""
    # NOTE: This saving is unique to the server and not shared among users
    try:
        with open(JSON_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
    except IOError as e:
        st.error(f"Error saving data to {JSON_FILE_PATH}: {e}")

# ----------------------------------------------------------------------
# 3. AI EXTRACTION & AUDIO FUNCTIONS
# ----------------------------------------------------------------------

def pcm_to_wav(pcm_data: bytes, sample_rate: int) -> bytes:
    """Converts raw PCM audio data into a standard WAV file format using only built-in libraries."""
    
    # 1. Prepare header components
    channels = 1
    bits_per_sample = 16
    bytes_per_sample = bits_per_sample // 8
    byte_rate = sample_rate * channels * bytes_per_sample
    
    # Total size of the PCM data
    data_size = len(pcm_data)
    
    # 2. Construct the header
    header = b'RIFF'                           # ChunkID
    header += (36 + data_size).to_bytes(4, 'little') # ChunkSize
    header += b'WAVE'                           # Format
    header += b'fmt '                           # Subchunk1ID
    header += (16).to_bytes(4, 'little')        # Subchunk1Size (16 for PCM)
    header += (1).to_bytes(2, 'little')         # AudioFormat (1 for PCM)
    header += channels.to_bytes(2, 'little')    # NumChannels
    header += sample_rate.to_bytes(4, 'little') # SampleRate
    header += byte_rate.to_bytes(4, 'little')   # ByteRate
    header += bytes_per_sample.to_bytes(2, 'little') # BlockAlign
    header += bits_per_sample.to_bytes(2, 'little')  # BitsPerSample
    header += b'data'                           # Subchunk2ID
    header += data_size.to_bytes(4, 'little')   # Subchunk2Size

    return header + pcm_data

def generate_tts_audio(text: str) -> Optional[str]:
    """Generates audio via Gemini TTS and returns Base64 encoded WAV data."""
    try:
        # Use a reliable voice (Kore is a clear voice)
        tts_config = types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config={
                "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Kore"}}
            }
        )
        
        # NOTE: Use gemini-2.5-flash-preview-tts for TTS
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash-preview-tts", 
            contents=[{"parts": [{"text": text}]}], 
            config=tts_config
        )

        # Extract base64 PCM data and mime type
        audio_part = response.candidates[0].content.parts[0].inlineData
        pcm_base64 = audio_part.data
        mime_type = audio_part.mimeType # Should be audio/L16;rate=24000
        
        if not pcm_base64 or 'audio/L16' not in mime_type:
            # If API returns an error or empty data
            return None
            
        # Extract sample rate from mime type (default to 24000 if extraction fails)
        try:
            # Safely extract rate from the mime type string
            rate_match = [part for part in mime_type.split(';') if 'rate=' in part]
            sample_rate = int(rate_match[0].split('=')[1]) if rate_match else 24000
        except:
            sample_rate = 24000

        # Decode base64 PCM data
        pcm_bytes = base64.b64decode(pcm_base64)
        
        # Convert raw PCM bytes to WAV format
        wav_bytes = pcm_to_wav(pcm_bytes, sample_rate)
        
        # Encode the final WAV bytes back to base64 for embedding in the HTML audio tag
        return base64.b64encode(wav_bytes).decode('utf-8')

    except Exception as e:
        # Log the detailed error but return None
        print(f"TTS Generation failed for word: {text}. Error: {e}")
        return None

def real_llm_vocabulary_extraction(num_words: int, existing_words: List[str]) -> List[Dict]:
    """
    1. Calls the Gemini API to generate structured vocabulary data.
    2. Calls the Gemini TTS model for each word to generate and encode the audio.
    """
    
    # --- Step 1: Generate Text Data (Remains the same using gemini-2.5-flash) ---
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
        st.error(f"🔴 Gemini Text Extraction Failed: {e}")
        return []
        
    # --- Step 2: Generate and Attach Audio Data (TTS) ---
    words_with_audio = []
    
    # We use a progress bar for the slow step
    progress_bar = st.progress(0, text=f"Generating TTS audio for 0 of {len(validated_words)} words...")
    
    for i, word_data in enumerate(validated_words):
        word = word_data['word']
        
        # Call the TTS function
        audio_data = generate_tts_audio(word)
        
        if audio_data:
            word_data['audio_base64'] = audio_data
        else:
            word_data['audio_base64'] = None # Explicitly set to None if generation fails
            
        words_with_audio.append(word_data)
        
        # Update progress bar
        progress = (i + 1) / len(validated_words)
        progress_bar.progress(progress, text=f"Generating TTS audio for {i + 1} of {len(validated_words)} words...")
        
    progress_bar.empty() # Clear progress bar on completion
    
    return words_with_audio

def load_and_update_vocabulary_data():
    """
    Loads existing data from local file (FAST) and implements aggressive goal-seeking extraction.
    This function will try to fill the vocabulary up to AUTO_EXTRACT_TARGET_SIZE.
    """
    if not st.session_state.is_auth: return
    
    st.session_state.vocab_data = load_vocabulary_from_file()
    word_count = len(st.session_state.vocab_data)
    
    if word_count > 0:
        st.info(f"✅ Loaded {word_count} words from local file.")
    
    # --- Aggressive Auto-Extraction Logic ---
    if word_count < AUTO_EXTRACT_TARGET_SIZE:
        words_needed = AUTO_EXTRACT_TARGET_SIZE - word_count
        # Calculate how many full batches we need (use LOAD_BATCH_SIZE as the extraction unit)
        batches_to_extract = words_needed // LOAD_BATCH_SIZE
        
        if batches_to_extract > 0:
            st.warning(f"Need {words_needed} more words. Triggering {LOAD_BATCH_SIZE} word extraction now...")
            
            # Use LOAD_BATCH_SIZE (10) as the extraction amount, even if less is needed, to be efficient.
            num_to_extract = LOAD_BATCH_SIZE 

            # Blocking extraction 
            existing_words = [d['word'] for d in st.session_state.vocab_data]
            new_words = real_llm_vocabulary_extraction(num_to_extract, existing_words)
            
            if new_words:
                st.session_state.vocab_data.extend(new_words)
                save_vocabulary_to_file(st.session_state.vocab_data)
                st.success(f"✅ Added {len(new_words)} words. Current total: {len(st.session_state.vocab_data)}.")
                st.rerun() 
            else:
                st.error("🔴 Failed to generate new words. Check API key and logs.")
    
    # Check if we need initial words for display (LOAD_BATCH_SIZE = 10)
    elif word_count < LOAD_BATCH_SIZE:
        st.warning(f"Need {LOAD_BATCH_SIZE} words for initial display. Triggering extraction...")
        # Blocking extraction for initial display (this handles the very first load)
        existing_words = [d['word'] for d in st.session_state.vocab_data]
        new_words = real_llm_vocabulary_extraction(LOAD_BATCH_SIZE, existing_words)
        
        if new_words:
            st.session_state.vocab_data.extend(new_words)
            save_vocabulary_to_file(st.session_state.vocab_data)
            st.success(f"✅ Initial {len(new_words)} words generated and saved.")
            st.rerun()


# --- Mock Authentication Handlers (Based on previous correct implementation) ---

def handle_auth(action: str, email: str, password: str):
    """
    Handles Mock user registration and login.
    """
    if not email or not password:
        st.error("Please enter both Email and Password.")
        return
        
    # 1. Admin Login Check
    if email == ADMIN_EMAIL and password == ADMIN_PASSWORD:
        is_admin = True
        
    # 2. General User Login Check (Simple Format Validation)
    elif len(password) >= 6 and '@' in email and '.' in email:
        is_admin = False
    
    else:
        st.error("Invalid credentials. Registration/Login requires a valid email and 6+ character password.")
        return

    # Login/Register Success
    st.session_state.current_user_email = email
    st.session_state.is_auth = True
    st.session_state.is_admin = is_admin
    st.session_state.words_displayed = LOAD_BATCH_SIZE
    st.session_state.quiz_start_index = 0
    
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
    st.session_state.words_displayed = LOAD_BATCH_SIZE
    st.rerun()

# ----------------------------------------------------------------------
# 4. UI COMPONENTS: VOCABULARY, QUIZ, ADMIN
# ----------------------------------------------------------------------

def load_more_words():
    """Increments the displayed word count."""
    st.session_state.words_displayed += LOAD_BATCH_SIZE
    st.rerun()

def display_vocabulary_ui():
    """Renders the Vocabulary Display feature with Load More functionality."""
    st.header("📚 Vocabulary Display", divider="blue")
    
    if not st.session_state.vocab_data:
        st.info("No vocabulary loaded yet. Please check the data status.")
        return

    total_words = len(st.session_state.vocab_data)
    words_to_show = min(total_words, st.session_state.words_displayed)
    
    st.markdown(f"**Showing {words_to_show} of {total_words} High-Level SAT Words**")
    
    with st.container(height=500, border=True):
        for i, data in enumerate(st.session_state.vocab_data[:words_to_show]):
            word_number = i + 1 
            word = data.get('word', 'N/A').upper()
            pronunciation = data.get('pronunciation', 'N/A')
            tip = data.get('tip', 'N/A')
            usage = data.get('usage', 'N/A')
            # 🟢 CHANGE: Fetch base64 audio data
            audio_base64 = data.get('audio_base64') 
            definition = data.get('definition', 'N/A')
            
            expander_title = f"**{word_number}.** {word} - {pronunciation}"
            
            with st.expander(expander_title):
                
                # --- AUDIO PLAYBACK (USES BASE64 WAV DATA) ---
                if audio_base64:
                    # Construct the data URL for the WAV format audio
                    audio_data_url = f"data:audio/wav;base64,{audio_base64}"
                    audio_html = f"""
                        <audio controls style="width: 100%;" src="{audio_data_url}">
                            Your browser does not support the audio element.
                        </audio>
                    """
                    st.markdown(audio_html, unsafe_allow_html=True)
                else:
                    st.warning("Audio not available for this word. TTS generation may have failed.")

                st.markdown(f"**Definition:** {definition.capitalize()}")
                st.markdown(f"**Memory Tip:** *{tip}*") 
                st.markdown(f"**Usage:** *'{usage}'*")

    if words_to_show < total_words:
        if st.button(f"Load {LOAD_BATCH_SIZE} More Words", on_click=load_more_words, type="secondary"):
            pass

def start_new_quiz():
    """
    Initializes the quiz based only on the currently displayed words in sequential order.
    """
    start = st.session_state.quiz_start_index
    end = start + QUIZ_SIZE
    
    # Ensure quiz words are picked sequentially from the current vocabulary data
    words_pool = st.session_state.vocab_data[start:end]
    
    if len(words_pool) < QUIZ_SIZE:
        st.error(f"Cannot start quiz. Need {QUIZ_SIZE} words starting from position {start + 1}.")
        return

    quiz_details = []
    all_definitions = [d['definition'].capitalize() for d in st.session_state.vocab_data]
    
    for question_data in words_pool:
        correct_answer = question_data['definition'].capitalize()
        
        # Select 3 unique decoy definitions from the full list
        decoys = random.sample([
            d for d in all_definitions if d != correct_answer
        ], min(3, len([d for d in all_definitions if d != correct_answer])))
        
        options = [correct_answer] + decoys
        random.shuffle(options)
        
        # Calculate the word's original index
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
        
        # NOTE: Quiz results are NOT saved to database in this stable version.
        
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


def admin_extraction_ui():
    """Renders the Admin Extraction and User Management feature."""
    st.header("💡 Data Tools", divider="orange") 
    
    if not st.session_state.is_admin:
        st.warning("You must be logged in as the Admin to use this tool.")
        return

    # --- User Management & Progress Tracking (MOCK) ---
    st.subheader("User Progress Overview")
    st.warning("⚠️ User progress tracking is disabled because a reliable shared database (Firebase) could not be installed.")
    st.markdown(f"""
    **Current Admin Email:** `{ADMIN_EMAIL}`
    
    To implement this section, the app needs a persistent, shared backend database to track multiple users. This feature is mocked.
    """)
    st.dataframe([
        {'Email': ADMIN_EMAIL, 'Status': 'Admin/Active', 'Quizzes Done': 'N/A'},
        {'Email': 'user@example.com', 'Status': 'User/Mock', 'Quizzes Done': 'N/A'}
    ], use_container_width=True)
        
    st.markdown("---")
    
    # --- Manual Extraction Override (Admin Only) ---
    st.subheader("Vocabulary Extraction")
    st.markdown(f"**Total Words in Database:** `{len(st.session_state.vocab_data)}` (Target: {REQUIRED_WORD_COUNT}).")

    if st.button(f"Force Extract {MANUAL_EXTRACT_BATCH} New Words", type="secondary"): 
        st.info(f"Manually extracting {MANUAL_EXTRACT_BATCH} new words...")
        
        existing_words = [d['word'] for d in st.session_state.vocab_data]
        new_batch = real_llm_vocabulary_extraction(MANUAL_EXTRACT_BATCH, existing_words) 
        
        if new_batch:
            st.session_state.vocab_data.extend(new_batch)
            save_vocabulary_to_file(st.session_state.vocab_data)
            st.success(f"✅ Added {len(new_batch)} words to the database.")
            st.rerun()
        else:
            st.error("Failed to generate new words. Check API key and logs.")

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
        # Load data on successful login 
        if not st.session_state.vocab_data:
            load_and_update_vocabulary_data() 

        # Auto-extraction logic (non-blocking status message)
        if len(st.session_state.vocab_data) < AUTO_EXTRACT_TARGET_SIZE:
             st.info("The vocabulary list is currently building to the target size...")

        # Use tabs for the main features
        tab_display, tab_quiz, tab_admin = st.tabs(["📚 Vocabulary List", "📝 Quiz Section", "🛠️ Data Tools"])
        
        with tab_display:
            display_vocabulary_ui()
            
        with tab_quiz:
            generate_quiz_ui()

        with tab_admin:
            admin_extraction_ui()

if __name__ == "__main__":
    main()
