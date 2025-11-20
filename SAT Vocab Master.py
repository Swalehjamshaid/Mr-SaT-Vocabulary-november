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

# --- FIREBASE IMPORTS ---
try:
    from firebase import initialize_app, get_auth, sign_up, sign_in, sign_out, is_user_logged_in, get_user_id
    from firebase import get_firestore, doc, set_doc, get_doc, collection, query, where, get_docs
except ImportError:
    st.error("🔴 FIREBASE ERROR: Please update requirements.txt with the necessary Firebase libraries and run in a compatible environment.")
    st.stop()

# --- GEMS API ---
try:
    from google import genai
    from google.genai import types
except ImportError:
    st.error("ERROR: The 'google-genai' and 'pydantic' libraries are required.")
    st.error("Please run: pip install google-genai pydantic")
    st.stop()


# ======================================================================
# *** LOCAL EXECUTION SETUP & FIREBASE CONFIGURATION ***
# ======================================================================

# Check for API Key (Gemini)
if "GEMINI_API_KEY" not in os.environ:
    st.error("🔴 GEMINI_API_KEY is missing! Please set it in your secrets.")
    st.stop()

# Initialize Gemini Client
try:
    gemini_client = genai.Client()
except Exception as e:
    st.error(f"🔴 Failed to initialize Gemini Client: {e}")
    st.stop()

# --- App State and Constants ---
REQUIRED_WORD_COUNT = 2000
LOAD_BATCH_SIZE = 10 
AUTO_EXTRACT_TARGET_SIZE = 200 
QUIZ_SIZE = 5 

# Admin Configuration
ADMIN_EMAIL = "rot.jamshaid@gmail.com"

# Firestore Initialization (Uses environment provided config)
if '__firebase_config__' in globals():
    try:
        firebase_config = json.loads(os.environ.get('__firebase_config__', '{}'))
        app = initialize_app(firebase_config)
        db = get_firestore(app)
        auth = get_auth(app)
        # Unique app ID for Firestore security rules
        APP_ID = os.environ.get('__app_id__', 'sat_vocab_master') 
    except Exception as e:
        st.error(f"🔴 Firestore Initialization Failed: {e}")
        db = None
else:
    st.error("🔴 Firestore environment configuration missing.")
    db = None

# --- Firestore Paths ---
# Public collection for vocabulary data shared among all users
VOCAB_COLLECTION_PATH = f"artifacts/{APP_ID}/public/data/vocabulary"
# Private collection for user records/progress
USERS_COLLECTION_PATH = f"artifacts/{APP_ID}/users/user_data/profiles"


# Pydantic Schema for Vocabulary Word
class SatWord(BaseModel):
    word: str = Field(description="The SAT-level word.")
    pronunciation: str = Field(description="Simple, hyphenated phonetic pronunciation (e.g., eh-FEM-er-al).")
    definition: str = Field(description="The concise dictionary definition.")
    tip: str = Field(description="A short, catchy mnemonic memory tip.")
    usage: str = Field(description="A professional sample usage sentence.")
    sat_level: str = Field(default="High", description="Should always be 'High'.")
    audio_url: Optional[str] = Field(default=None, description="Public URL for MP3 audio pronunciation.")

# ----------------------------------------------------------------------
# 2. DATA PERSISTENCE & STATE MANAGEMENT
# ----------------------------------------------------------------------

if 'current_user_email' not in st.session_state: st.session_state.current_user_email = None
if 'is_auth' not in st.session_state: st.session_state.is_auth = False
if 'vocab_data' not in st.session_state: st.session_state.vocab_data = []
if 'quiz_active' not in st.session_state: st.session_state.quiz_active = False
if 'words_displayed' not in st.session_state: st.session_state.words_displayed = LOAD_BATCH_SIZE
if 'quiz_start_index' not in st.session_state: st.session_state.quiz_start_index = 0
if 'is_admin' not in st.session_state: st.session_state.is_admin = False
if 'quiz_scores' not in st.session_state: st.session_state.quiz_scores = [] # Stores user scores per session

# --- Firestore Data Management (Replacing local JSON file) ---

@st.cache_data(ttl=3600, show_spinner=False)
def get_all_vocabulary_from_firestore():
    """Fetches all vocabulary words from the shared public Firestore collection."""
    if not db: return []
    try:
        vocab_ref = collection(db, VOCAB_COLLECTION_PATH)
        docs = get_docs(vocab_ref)
        # Ensure data integrity by validating against the Pydantic model
        data = [SatWord(**d.to_dict()).model_dump() for d in docs]
        return data
    except Exception as e:
        st.error(f"Firestore Read Error (Vocabulary): {e}")
        return []

def save_new_vocabulary_to_firestore(new_words: List[Dict]):
    """Saves new words to the shared public Firestore collection."""
    if not db: return
    try:
        vocab_ref = collection(db, VOCAB_COLLECTION_PATH)
        for word_data in new_words:
            # Generate a unique ID based on the word for idempotent writes
            word_id = word_data['word'].lower() 
            set_doc(doc(vocab_ref, word_id), word_data)
        st.toast(f"✅ Saved {len(new_words)} words to Firestore.")
        get_all_vocabulary_from_firestore.clear() # Clear cache to fetch new data
    except Exception as e:
        st.error(f"Firestore Write Error (Vocabulary): {e}")

def update_user_progress_in_firestore(quiz_score_data: Dict):
    """Saves the latest quiz score data to the user's document."""
    if not db or not st.session_state.current_user_email: return
    try:
        user_id = get_user_id()
        user_ref = doc(db, USERS_COLLECTION_PATH, user_id)
        
        # Structure the score entry
        new_score = {
            "timestamp": time.time(),
            "score": quiz_score_data['score'],
            "total": quiz_score_data['total'],
            "accuracy": quiz_score_data['accuracy']
        }
        st.session_state.quiz_scores.append(new_score)
        
        # Read the current document, append the new score, and save back
        user_doc = get_doc(user_ref)
        if user_doc.exists:
            current_data = user_doc.to_dict()
            current_scores = current_data.get('quiz_scores', [])
            current_scores.append(new_score)
            
            # Simple max score logic (Admin visibility)
            max_score = max([s['score'] for s in current_scores])
            
            set_doc(user_ref, {
                'email': st.session_state.current_user_email,
                'quiz_scores': current_scores,
                'last_activity': new_score['timestamp'],
                'max_score_5_questions': max_score
            }, merge=True)
            
    except Exception as e:
        st.warning(f"Failed to update user progress: {e}")

# ----------------------------------------------------------------------
# 3. AI EXTRACTION & AUTHENTICATION FUNCTIONS
# ----------------------------------------------------------------------

def construct_tts_url(text: str) -> str:
    """Constructs a reliable TTS URL using Google's public endpoint."""
    encoded_text = urllib.parse.quote(text)
    return f"https://translate.google.com/translate_tts?ie=UTF-8&q={encoded_text}&tl=en&client=tw-ob"


def real_llm_vocabulary_extraction(num_words: int, existing_words: List[str]) -> List[Dict]:
    """
    Calls the Gemini API to generate structured vocabulary data and constructs the audio URL.
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
        st.error(f"🔴 Gemini Text Extraction Failed: {e}")
        return []
        
    words_with_audio = []
    with st.spinner(f"🔗 Constructing audio links for {len(validated_words)} words..."):
        for word_data in validated_words:
            word = word_data['word']
            audio_url = construct_tts_url(word)
            word_data['audio_url'] = audio_url
            words_with_audio.append(word_data)

    return words_with_audio

def load_and_update_vocabulary_data():
    """
    Loads existing data from Firestore (FAST) and checks initial word count.
    """
    if not st.session_state.is_auth: return
    
    st.session_state.vocab_data = get_all_vocabulary_from_firestore()
    word_count = len(st.session_state.vocab_data)
    
    if word_count > 0:
        st.info(f"✅ Loaded {word_count} words from the shared database.")
    
    # Check if the initial required count (10 words for display) is met
    if word_count < LOAD_BATCH_SIZE:
        st.warning(f"Need {LOAD_BATCH_SIZE - word_count} more words for initial display. Triggering extraction now...")
        
        # Blocking extraction for initial display
        existing_words = [d['word'] for d in st.session_state.vocab_data]
        new_words = real_llm_vocabulary_extraction(LOAD_BATCH_SIZE - word_count, existing_words)
        
        if new_words:
            save_new_vocabulary_to_firestore(new_words)
            # Fetch updated data after saving
            st.session_state.vocab_data = get_all_vocabulary_from_firestore()
            st.success(f"✅ Initial {len(new_words)} words generated and saved.")
            st.rerun() 


# --- Authentication Handlers ---

def handle_auth(action: str, email: str, password: str):
    """
    Handles Firebase user registration and login.
    """
    if not email or not password:
        st.error("Please enter both Email and Password.")
        return
    if action == "register" and len(password) < 6:
        st.error("Password must be at least 6 characters long.")
        return
        
    try:
        if action == "register":
            # Firebase handles registration
            user_id = sign_up(email, password)
            st.success("✅ Registration successful. Please Sign In.")
            return

        elif action == "login":
            # Firebase handles login
            user_id = sign_in(email, password)
            st.session_state.current_user_email = email
            st.session_state.is_auth = True
            st.session_state.is_admin = (email == ADMIN_EMAIL)
            st.session_state.words_displayed = LOAD_BATCH_SIZE
            st.session_state.quiz_start_index = 0
            
            display_name = "Admin" if st.session_state.is_admin else email
            st.success(f"Logged in as: {display_name}! Access granted.")
            load_and_update_vocabulary_data() 
            st.rerun()
            

    except Exception as e:
        error_msg = str(e)
        if "EMAIL_EXISTS" in error_msg:
            st.error("This email is already registered. Please login.")
        elif "INVALID_LOGIN_CREDENTIALS" in error_msg or "USER_NOT_FOUND" in error_msg:
            st.error("Invalid email or password.")
        else:
            st.error(f"Authentication failed: {error_msg}")

def handle_logout():
    """Handles session state reset and signs out the user."""
    try:
        sign_out()
    except:
        pass # Ignore errors if already signed out
    st.session_state.is_auth = False
    st.session_state.current_user_email = None
    st.session_state.quiz_active = False
    st.session_state.quiz_scores = []
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
            audio_url = data.get('audio_url')
            definition = data.get('definition', 'N/A')
            
            expander_title = f"**{word_number}.** {word} - {pronunciation}"
            
            with st.expander(expander_title):
                
                if audio_url:
                    audio_html = f"""
                        <audio controls style="width: 100%;" src="{audio_url}">
                            Your browser does not support the audio element.
                        </audio>
                    """
                    st.markdown(audio_html, unsafe_allow_html=True)
                else:
                    st.warning("Audio URL is missing for this word.")

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
        
        # Save score to Firestore before displaying results
        update_user_progress_in_firestore(st.session_state.quiz_results)
        
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

    # --- User Management & Progress Tracking ---
    st.subheader("User Progress Overview")
    
    if not db:
        st.error("Cannot access user data. Firestore is not initialized.")
        return
        
    try:
        users_ref = collection(db, USERS_COLLECTION_PATH)
        user_docs = get_docs(users_ref)
        
        user_list = []
        for doc in user_docs:
            data = doc.to_dict()
            user_list.append({
                'Email': data.get('email', 'N/A'),
                'Max Score': data.get('max_score_5_questions', 0),
                'Last Activity': time.strftime('%Y-%m-%d %H:%M', time.localtime(data.get('last_activity', 0))),
                'Total Quizzes': len(data.get('quiz_scores', []))
            })

        if user_list:
            st.dataframe(user_list, use_container_width=True)
        else:
            st.info("No registered users have completed a quiz yet.")
            
    except Exception as e:
        st.error(f"Error fetching user data: {e}")
        
    st.markdown("---")
    
    # --- Manual Extraction Override (Admin Only) ---
    st.subheader("Vocabulary Extraction")
    st.markdown(f"**Total Words in Database:** `{len(st.session_state.vocab_data)}` (Target: {REQUIRED_WORD_COUNT}).")

    if st.button("Force Extract 5 New Words", type="secondary"):
        st.info("Manually extracting 5 new words...")
        
        existing_words = [d['word'] for d in st.session_state.vocab_data]
        new_batch = real_llm_vocabulary_extraction(5, existing_words)
        
        if new_batch:
            save_new_vocabulary_to_firestore(new_batch)
            st.success(f"✅ Added {len(new_batch)} words to the database.")
            # Reload data and rerun the app to show new words
            get_all_vocabulary_from_firestore.clear() 
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
        
        # Check authentication state (Firebase provides the sign-in status)
        if not st.session_state.is_auth and is_user_logged_in():
            st.session_state.is_auth = True
            user_email = get_user_id()
            st.session_state.current_user_email = user_email
            st.session_state.is_admin = (user_email == ADMIN_EMAIL)
            st.success(f"Re-authenticated as: {user_email}")
            load_and_update_vocabulary_data() 

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
            
            **Note:** Registration requires a unique email and 6+ character password.
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
        # Load data on successful login or re-authentication
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
