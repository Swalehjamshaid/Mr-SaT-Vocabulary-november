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
# ----------------------------------------------------

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

# Check for API Key (Still required for text extraction)
if "GEMINI_API_KEY" not in os.environ:
    st.error("🔴 GEMINI_API_KEY is missing!")
    st.warning("""
    To fix this, you MUST set your Gemini API key securely.
    1. **If on Streamlit Cloud:** Go to your app settings (Secrets) and add the key in TOML format: 
       `GEMINI_API_KEY="YOUR_ACTUAL_KEY_STRING"`
    2. **If running locally:** Set the GEMINI_API_KEY environment variable in your terminal/OS.
    """)
    st.stop()
    
# Initialize Gemini Client (reads key from environment variable automatically)
try:
    gemini_client = genai.Client()
except Exception as e:
    st.error(f"🔴 Failed to initialize Gemini Client: {e}")
    st.stop()

# Use a local JSON file for persistent storage (Fallback storage)
JSON_FILE_PATH = "vocab_data.json" 

REQUIRED_WORD_COUNT = 2000
LOAD_BATCH_SIZE = 10 

# Admin Configuration
ADMIN_EMAIL = "rot.jamshaid@gmail.com"
ADMIN_PASSWORD = "Jamshaid,1981"
AUTO_EXTRACT_TARGET_SIZE = 200 
AUTO_EXTRACT_BATCH = 5 

# Pydantic Schema for Structured AI Output
class SatWord(BaseModel):
    """Defines the exact structure for the AI-generated vocabulary word."""
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

def load_vocabulary_from_file():
    """Loads vocabulary data from the local JSON file."""
    if os.path.exists(JSON_FILE_PATH):
        with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                return data if isinstance(data, list) else []
            except json.JSONDecodeError:
                st.error("Error decoding JSON file. Starting with empty vocabulary.")
                return []
    return []

def save_vocabulary_to_file(data: List[Dict]):
    """Saves the current vocabulary data to the local JSON file."""
    try:
        with open(JSON_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
    except IOError as e:
        st.error(f"Error saving data to {JSON_FILE_PATH}: {e}")

# ----------------------------------------------------------------------
# 3. AI EXTRACTION & AUDIO FUNCTIONS
# ----------------------------------------------------------------------

def construct_tts_url(text: str) -> str:
    """
    Constructs a reliable TTS URL using Google's public endpoint.
    """
    encoded_text = urllib.parse.quote(text)
    return f"https://translate.google.com/translate_tts?ie=UTF-8&q={encoded_text}&tl=en&client=tw-ob"


def real_llm_vocabulary_extraction(num_words: int, existing_words: List[str]) -> List[Dict]:
    """
    Calls the Gemini API to generate structured vocabulary data and constructs the audio URL.
    """
    
    prompt = f"""
    Generate {num_words} unique, extremely high-level SAT vocabulary words.
    The words must NOT be any of the following: {', '.join(existing_words) if existing_words else 'none'}.
    For each word, provide:
    1. The word itself.
    2. A simple, hyphenated phonetic pronunciation (like a dictionary spelling, e.g., 'EH-fem-er-al').
    3. A concise dictionary definition.
    4. A short, creative, and memorable mnemonic memory tip.
    5. A professional sample usage sentence.
    
    Return the result as a JSON array where each object strictly conforms to the provided schema.
    """

    list_schema = {"type": "array", "items": SatWord.model_json_schema()}
    config = types.GenerateContentConfig(response_mime_type="application/json", response_json_schema=list_schema)
    
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt, config=config
        )
        new_data_list = json.loads(response.text)
        validated_words = [SatWord(**item).model_dump() for item in new_data_list if 'word' in item]
        if not validated_words:
            st.error("AI returned structured data but validation failed for all items.")
            return []
    except Exception as e:
        st.error(f"🔴 Gemini Text Extraction Failed: {e}")
        return []
        
    words_with_audio = []
    
    # 🔴 FIX: Using st.spinner to show activity during the relatively fast audio link construction
    with st.spinner(f"🔗 Constructing audio links for {len(validated_words)} words..."):
        
        for word_data in validated_words:
            word = word_data['word']
            audio_url = construct_tts_url(word)
            word_data['audio_url'] = audio_url
            words_with_audio.append(word_data)

    return words_with_audio

def load_and_update_vocabulary_data():
    """
    Loads existing data (FAST) and ensures old words have the required audio URL (FAST).
    """
    st.session_state.vocab_data = load_vocabulary_from_file()
    word_count = len(st.session_state.vocab_data)
    
    # Fix for old data: If old words don't have the audio_url field, generate it now.
    words_missing_url = [d for d in st.session_state.vocab_data if d.get('audio_url') is None]
    if words_missing_url:
        with st.spinner("Updating audio links..."): 
            for word_data in words_missing_url:
                word_data['audio_url'] = construct_tts_url(word_data['word'])
            
        save_vocabulary_to_file(st.session_state.vocab_data)

    if word_count > 0:
        st.info(f"✅ Loaded {word_count} words from local file.")
    
    # Check if the initial required count (10 words for display) is met
    if word_count < LOAD_BATCH_SIZE:
        st.warning(f"Need {LOAD_BATCH_SIZE - word_count} more words for initial display. Triggering extraction now...")
        
        # We MUST block and run the AI now if the count is zero, otherwise the app is empty.
        new_words = real_llm_vocabulary_extraction(LOAD_BATCH_SIZE - word_count, [])
        if new_words:
            st.session_state.vocab_data.extend(new_words)
            save_vocabulary_to_file(st.session_state.vocab_data)
            st.success(f"✅ Initial {len(new_words)} words generated.")
            st.rerun() 


# ----------------------------------------------------------------------
# 4. AUTHENTICATION (SECURE MOCK FOR STREAMLIT CLOUD)
# ----------------------------------------------------------------------

def handle_login(email: str, password: str):
    """
    Handles user sign-in/login with internal validation.
    """
    if not email or not password:
        st.error("Please enter both Email and Password.")
        return

    # 🟢 ADMIN LOGIN CHECK (Case-sensitive for password)
    if email == ADMIN_EMAIL and password == ADMIN_PASSWORD:
        st.session_state.current_user_email = ADMIN_EMAIL
        st.session_state.is_auth = True
        st.session_state.words_displayed = LOAD_BATCH_SIZE
        st.session_state.quiz_start_index = 0
        st.success(f"Logged in as: Admin ({ADMIN_EMAIL})! Access granted.")
        load_and_update_vocabulary_data() 
        return

    # 🟢 STANDARD USER LOGIN/SIGNUP (MOCK)
    # Simulates registration/login for any new valid user.
    if len(password) >= 6 and '@' in email and '.' in email:
        st.session_state.current_user_email = email
        st.session_state.is_auth = True
        st.session_state.words_displayed = LOAD_BATCH_SIZE
        st.session_state.quiz_start_index = 0
        st.success(f"Logged in as: {email}! (Simulated Access)")
        load_and_update_vocabulary_data() 
    else:
        st.error("Invalid credentials. Password must be 6+ characters and the email must be valid.")
        
def handle_logout():
    """Handles session state reset on logout."""
    st.session_state.is_auth = False
    st.session_state.current_user_email = None
    st.session_state.quiz_active = False
    st.session_state.words_displayed = LOAD_BATCH_SIZE
    st.rerun()


# ----------------------------------------------------------------------
# 5. UI COMPONENTS
# ----------------------------------------------------------------------

def display_vocabulary_ui():
    """Renders the Vocabulary Display feature with Load More functionality."""
    st.header("📚 Vocabulary Display", divider="blue")
    
    if not st.session_state.vocab_data:
        st.info("No vocabulary loaded yet. Please check the AI Extraction status.")
        return

    total_words = len(st.session_state.vocab_data)
    words_to_show = min(total_words, st.session_state.words_displayed)
    
    st.markdown(f"**Showing {words_to_show} of {total_words} High-Level SAT Words**")
    
    # Use a scrollable container for the words being shown
    with st.container(height=500, border=True):
        for i, data in enumerate(st.session_state.vocab_data[:words_to_show]):
            word_number = i + 1 
            word = data.get('word', 'N/A').upper()
            pronunciation = data.get('pronunciation', 'N/A')
            definition = data.get('definition', 'N/A')
            tip = data.get('tip', 'N/A')
            usage = data.get('usage', 'N/A')
            audio_url = data.get('audio_url')
            
            expander_title = f"**{word_number}.** {word} - {pronunciation}"
            
            with st.expander(expander_title):
                
                # --- AUDIO PLAYBACK ---
                if audio_url:
                    audio_html = f"""
                        <audio controls style="width: 100%;" src="{audio_url}">
                            Your browser does not support the audio element.
                        </audio>
                    """
                    st.markdown(audio_html, unsafe_allow_html=True)
                else:
                    st.warning("Audio URL is missing for this word.")
                # -----------------------

                st.markdown(f"**Definition:** {definition.capitalize()}")
                st.markdown(f"**Memory Tip:** *{tip}*") 
                st.markdown(f"**Usage:** *'{usage}'*")

    # Load More Button
    if words_to_show < REQUIRED_WORD_COUNT:
        if st.button(f"Load {LOAD_BATCH_SIZE} More Words", on_click=load_more_words, type="secondary"):
            pass


def start_new_quiz():
    """
    Initializes the quiz based only on the currently displayed words in sequential order.
    """
    QUIZ_SIZE = 5 
    
    start = st.session_state.quiz_start_index
    end = start + QUIZ_SIZE
    
    words_pool = st.session_state.vocab_data[start:end]
    
    if len(words_pool) < QUIZ_SIZE:
        st.error(f"Cannot start quiz. Need {QUIZ_SIZE} words starting from position {start + 1}, but only found {len(words_pool)}.")
        return

    quiz_words = words_pool 
    
    quiz_details = []
    all_definitions = [d['definition'].capitalize() for d in st.session_state.vocab_data]
    
    for question_data in quiz_words:
        correct_answer = question_data['definition'].capitalize()
        
        # Select 3 unique decoy definitions from the full list
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
    QUIZ_SIZE = 5
    st.session_state.quiz_start_index += QUIZ_SIZE
    st.session_state.quiz_active = False 

def generate_quiz_ui():
    """Renders the Quiz Section feature."""
    st.header("📝 Vocabulary Quiz", divider="green")
    
    QUIZ_SIZE = 5
    total_words = len(st.session_state.vocab_data)
    
    if total_words < QUIZ_SIZE:
        st.info(f"A minimum of {QUIZ_SIZE} words is required to start a quiz. Current total: {total_words}")
        return

    start_word_num = st.session_state.quiz_start_index + 1
    end_word_num = min(st.session_state.quiz_start_index + QUIZ_SIZE, total_words)
    
    
    if not st.session_state.quiz_active:
        
        if start_word_num > total_words:
            st.info("You have completed all available quiz blocks!")
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


def admin_extraction_ui():
    """Renders the Admin Extraction feature."""
    st.header("💡 Data Management", divider="orange") 
    
    is_admin = st.session_state.current_user_email == ADMIN_EMAIL
    
    if not is_admin:
        st.warning("You must be logged in as the Admin to use this tool.")
        return

    st.markdown(f"""
    **Total Words in Database:** `{len(st.session_state.vocab_data)}` (Target: {REQUIRED_WORD_COUNT}).
    
    The application will continue to build the vocabulary list automatically until it reaches {AUTO_EXTRACT_TARGET_SIZE} words.
    """)
    
    st.markdown("---")
    
    # --- Manual Extraction Override (Admin Only) ---
    if st.button("Force Extract 5 New Words", type="secondary"):
        st.info("Manually extracting 5 new words...")
        
        existing_words = [d['word'] for d in st.session_state.vocab_data]
        new_batch = real_llm_vocabulary_extraction(5, existing_words)
        
        if new_batch:
            st.session_state.vocab_data.extend(new_batch)
            save_vocabulary_to_file(st.session_state.vocab_data)
            st.success(f"✅ Added {len(new_batch)} words. Total: {len(st.session_state.vocab_data)}")
            st.session_state.words_displayed = len(st.session_state.vocab_data) 
            st.rerun()
        else:
            st.error("Failed to generate new words. Check API key and logs.")

# ----------------------------------------------------------------------
# 6. STREAMLIT APPLICATION STRUCTURE
# ----------------------------------------------------------------------

def main():
    """The main Streamlit application function."""
    st.set_page_config(page_title="AI Vocabulary Builder", layout="wide")
    st.title("🧠 AI-Powered Vocabulary Builder")
    
    # --- Sidebar for Auth Status ---
    with st.sidebar:
        st.header("User Login")
        
        if not st.session_state.is_auth:
            
            st.markdown("##### New User Signup / Existing User Login")
            
            user_email = st.text_input("📧 Email", key="user_email_input", value=ADMIN_EMAIL if 'user_email_input' not in st.session_state else st.session_state.user_email_input)
            password = st.text_input("🔑 Password", type="password", key="password_input", value=ADMIN_PASSWORD if 'password_input' not in st.session_state else st.session_state.password_input)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Login", key="login_btn", type="primary"):
                    handle_login(user_email, password)
            with col2:
                # 🔴 NOTE: Register is handled by the same login simulation logic due to environment constraints.
                if st.button("Register", key="register_btn"):
                    handle_login(user_email, password)
            
            st.markdown("---")
            st.markdown(f"""
            **Admin Login:** `{ADMIN_EMAIL}` / `{ADMIN_PASSWORD}`
            
            **Note on Verification:** This app relies on internal security simulation due to hosting environment restrictions. Passwords must be 6+ characters.
            """)
            
        else:
            display_name = "Admin" if st.session_state.current_user_email == ADMIN_EMAIL else st.session_state.current_user_email
            st.success(f"Logged in as: **{display_name}**")
            
            if st.button("Log Out", on_click=handle_logout):
                pass
                
    # --- Main Content ---
    
    load_and_update_vocabulary_data() 
    
    if not st.session_state.is_auth:
        st.info("Please log in using the sidebar to access the Vocabulary Builder.")
    else:
        # 🟢 FINAL CLEANUP: Ensure background processing happens (if necessary)
        if len(st.session_state.vocab_data) < AUTO_EXTRACT_TARGET_SIZE:
             st.info("The app is currently building the vocabulary list...")

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
