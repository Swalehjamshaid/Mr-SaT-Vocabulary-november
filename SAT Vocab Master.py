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

# 🟢 FINAL CORRECTED FIREBASE IMPORTS (Requires 'firebase-admin' in requirements.txt)
try:
    from firebase_admin import credentials, initialize_app, firestore 
    import firebase_admin 
except ImportError:
    st.error("FIREBASE ERROR: The required library 'firebase-admin' is likely missing in requirements.txt.")
    st.stop()


# 🟢 NEW: Import gTTS and io for open-source TTS solution (Requires 'gTTS' in requirements.txt)
try:
    from gtts import gTTS
    import io
except ImportError:
    st.error("ERROR: The 'gtts' library is required for open-source TTS.")
    st.stop()
    
# --- GEMS API (Requires 'google-genai' and 'pydantic' in requirements.txt) ---
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

# 🟢 FINAL FIRESTORE INITIALIZATION FIX APPLIED HERE
try:
    # CRITICAL FIX: Ensure the secret value is treated as a string before loading the JSON.
    secret_value = os.environ["FIREBASE_SERVICE_ACCOUNT"].strip().strip('"').strip("'")
    service_account_info = json.loads(secret_value)
    
    # Initialize Firebase Admin SDK (Only once)
    if not firebase_admin._apps:
        cred = credentials.Certificate(service_account_info)
        initialize_app(cred)

    # CORRECTED LINE: This now correctly calls client() on the firebase_admin.firestore module
    db = firestore.client() 
    # Define the main collection path (shared public data)
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
AUTO_EXTRACT_TARGET_SIZE = REQUIRED_WORD_COUNT 
QUIZ_SIZE = 5 

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
    # 🟢 NEW: Firestore field to ensure proper sorting
    created_at: float = Field(default_factory=time.time)

# ----------------------------------------------------------------------
# 2. DATA PERSISTENCE & STATE MANAGEMENT (FIREBASE FIRESTORE)
# ----------------------------------------------------------------------

if 'current_user_email' not in st.session_state: st.session_state.current_user_email = None
if 'is_auth' not in st.session_state: st.session_state.is_auth = False
if 'vocab_data' not in st.session_state: st.session_state.vocab_data = []
if 'quiz_active' not in st.session_state: st.session_state.quiz_active = False
if 'words_displayed' not in st.session_state: st.session_state.words_displayed = LOAD_BATCH_SIZE
if 'quiz_start_index' not in st.session_state: st.session_state.quiz_start_index = 0
if 'is_admin' not in st.session_state: st.session_state.is_admin = False
# 🟢 NEW FLAG FOR BACKGROUND EXTRACTION
if 'is_extracting_background' not in st.session_state: st.session_state.is_extracting_background = False


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
        doc_ref.set(word_data, merge=False)
        return True
    except Exception as e:
        st.error(f"🔴 Firestore Save Failed for {word_data['word']}: {e}")
        return False
        
def update_word_in_firestore(word_data: Dict):
    """Updates a single word document in the Firestore collection."""
    try:
        doc_ref = VOCAB_COLLECTION.document(word_data['word'].lower())
        doc_ref.update({
            'audio_base64': word_data['audio_base64']
        })
        return True
    except Exception as e:
        st.error(f"🔴 Firestore Update Failed for {word_data['word']}: {e}")
        return False


# ----------------------------------------------------------------------
# 3. AI EXTRACTION & AUDIO FUNCTIONS
# ----------------------------------------------------------------------

def generate_tts_audio(text: str) -> Optional[str]:
    """Generates audio via gTTS and returns Base64 encoded MP3 data."""
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        base64_data = base64.b64encode(mp3_fp.read()).decode('utf-8')
        return base64_data

    except Exception as e:
        print(f"gTTS Generation failed for word: {text}. Error: {e}")
        # Note: Do NOT use st.error here, as this function is run in the background.
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
    
    # Run TTS without Streamlit progress bars (as it's background)
    for word_data in validated_words:
        word = word_data['word']
        audio_data = generate_tts_audio(word)
        word_data['audio_base64'] = audio_data if audio_data else None
        words_with_audio.append(word_data)
        
    return words_with_audio

def run_background_extraction():
    """
    CRITICAL FIX: This runs the slow AI task in the background (triggered by a flag)
    and uses st.rerun() only once when the slow task is complete, minimizing delay.
    """
    if not st.session_state.is_auth or not st.session_state.is_extracting_background:
        return
        
    # Check if we still need words
    word_count = len(st.session_state.vocab_data)
    if word_count >= AUTO_EXTRACT_TARGET_SIZE:
        st.session_state.is_extracting_background = False
        st.toast(f"✅ Extraction target ({AUTO_EXTRACT_TARGET_SIZE}) reached!")
        return

    # Extract only 1 word (for minimum delay)
    words_to_extract = 1
    
    existing_words = [d['word'] for d in st.session_state.vocab_data]
    
    # 🛑 SLOW STEP: AI extraction and TTS generation runs here
    new_words = real_llm_vocabulary_extraction(words_to_extract, existing_words) 
    
    if new_words:
        word_data = new_words[0]
        if save_word_to_firestore(word_data):
            st.session_state.vocab_data.append(word_data)
            
            # Use st.toast for passive feedback
            st.toast(f"✅ Added 1 word in background. Total: {word_count + 1}")
            
            # Only rerun if extraction was successful to check for the next word
            time.sleep(0.1)
            st.rerun()
        else:
            print("Failed to save word to Firestore. Stopping background extraction.")
            st.session_state.is_extracting_background = False
    else:
        print("AI extraction failed. Stopping background extraction.")
        st.session_state.is_extracting_background = False

def start_background_extraction_manual():
    """Admin button handler to start the background process."""
    if not st.session_state.is_admin:
        st.error("Only the Admin can start background extraction.")
        return
    
    st.session_state.is_extracting_background = True
    st.rerun() # Forces a rerun to hit the passive extraction function

def stop_background_extraction_manual():
    """Admin button handler to stop the background process."""
    st.session_state.is_extracting_background = False
    st.info("Background extraction paused.")

def handle_admin_extraction_button(num_words: int):
    """Handles the bulk 50 word manual extraction."""
    st.session_state.is_extracting_background = False
    st.info(f"Manually extracting {num_words} new words...")
    
    existing_words = [d['word'] for d in st.session_state.vocab_data]
    new_batch = real_llm_vocabulary_extraction(num_words, existing_words) 
    
    if new_batch:
        successful_saves = 0
        for word_data in new_batch:
            if save_word_to_firestore(word_data):
                st.session_state.vocab_data.append(word_data)
                successful_saves += 1
                
        st.success(f"✅ Added {successful_saves} words. Current total: {len(st.session_state.vocab_data)}.")
        st.rerun() 
    else:
        st.error("🔴 Failed to generate new words. Check API key and logs.")

def load_and_update_vocabulary_data():
    """
    Loads data passively upon login/app start. 
    It is no longer responsible for *starting* extraction.
    """
    if not st.session_state.is_auth: return
    
    st.session_state.vocab_data = load_vocabulary_from_firestore()
    
    fill_missing_audio(st.session_state.vocab_data)
        
    word_count = len(st.session_state.vocab_data)
    
    if word_count > 0:
        st.info(f"✅ Loaded {word_count} words from shared database (Firestore).")
    elif st.session_state.is_auth:
        st.info("Database is empty. Please log in as Admin and use the 'Data Tools' tab to extract the first batch of words.")

    # 🛑 CRITICAL: Check and run the background task here after loading data
    if st.session_state.is_extracting_background and st.session_state.is_admin:
        run_background_extraction() 

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
        st.info("No vocabulary loaded yet. Please check the Data Tools tab to generate the first batch.")
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
            audio_base64 = data.get('audio_base64') 
            definition = data.get('definition', 'N/A')
            
            expander_title = f"**{word_number}.** {word} - {pronunciation}"
            
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
                            key=f"fix_audio_{i}", 
                            on_click=handle_fix_single_audio, 
                            args=(i,),
                            type="primary"
                        )

                st.markdown(f"**Definition:** {definition.capitalize()}")
                st.markdown(f"**Memory Tip:** *{tip}*") 
                st.markdown(f"**Usage:** *'{usage}'*")

    if words_to_show < total_words:
        if st.button(f"Load {LOAD_BATCH_SIZE} More Words", on_click=load_more_words, type="secondary"):
            pass

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
        st.caption(f"Testing words **start_word_num** to **end_word_num**.")
        
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
    st.markdown(f"**Corrupted Entries:** {len([d for d in st.session_state.vocab_data if d.get('audio_base64') is None])} words currently missing audio.")

    st.button(
        "Attempt Bulk Audio Fix (Fix All Missing Pronunciations)", 
        on_click=handle_bulk_audio_fix, 
        type="primary"
    )

    st.markdown("---")

    # --- Background Extraction Control ---
    st.subheader("Background Extraction Control (1 Word Patches)")
    
    if st.session_state.is_extracting_background:
        st.warning("Background extraction is **ACTIVE**. New words are being generated one at a time.")
        st.button("Stop Continuous Extraction", on_click=stop_background_extraction_manual, type="secondary")
    else:
        st.info("Background extraction is **PAUSED**. Start to continuously build the vocabulary database.")
        st.button("Start Continuous Extraction (1 Word Patch)", on_click=start_background_extraction_manual, type="primary")

    st.markdown("---")
    
    # --- Manual Extraction Override (Admin Only) ---
    st.subheader("Vocabulary Extraction (Bulk)")
    st.markdown(f"**Total Words in Database:** `{len(st.session_state.vocab_data)}` (Target: {REQUIRED_WORD_COUNT}).")

    if st.button(f"Force Extract {MANUAL_EXTRACT_BATCH} New Words", type="secondary"): 
        handle_admin_extraction_button(MANUAL_EXTRACT_BATCH)


# ----------------------------------------------------------------------
# 5. STREAMLIT APPLICATION STRUCTURE
# ----------------------------------------------------------------------

def main():
    """The main Streamlit application function."""
    # 🟢 CRITICAL: This function must be defined before it is called below.
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
        if st.session_state.is_admin and len(st.session_state.vocab_data) < AUTO_EXTRACT_TARGET_SIZE:
             st.info(f"The vocabulary list currently has {len(st.session_state.vocab_data)} words. Start the background extraction in 'Data Tools' to build the list.")

        # Use tabs for the main features
        tab_display, tab_quiz, tab_admin = st.tabs(["📚 Vocabulary List", "📝 Quiz Section", "🛠️ Data Tools"])
        
        with tab_display:
            display_vocabulary_ui()
            
        with tab_quiz:
            generate_quiz_ui()

        with tab_admin:
            admin_extraction_ui()

# 🟢 CRITICAL FIX FOR NameError: The script execution starts here.
if __name__ == "__main__":
    main()
