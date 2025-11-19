import json
import time
import random
import sys
import os
from typing import List, Dict, Optional
import streamlit as st
from pydantic import BaseModel, Field, ValidationError
from pydantic import json_schema 

# --- AI & STRUCTURED OUTPUT LIBRARIES ---
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

# Check for API Key (Works for local environment variables and Streamlit Secrets)
if "GEMINI_API_KEY" not in os.environ and not ("__initial_auth_token__" in globals() and __initial_auth_token__):
    st.error("🔴 GEMINI_API_KEY environment variable is not set.")
    st.warning("Please set your Gemini API key before running the application.")
    st.stop()
    
# Initialize Gemini Client (reads key from environment variable automatically)
try:
    # Client will automatically read the GEMINI_API_KEY environment variable
    gemini_client = genai.Client()
except Exception as e:
    st.error(f"🔴 Failed to initialize Gemini Client: {e}")
    st.stop()

# Use a local JSON file for persistent storage
JSON_FILE_PATH = "vocab_data.json" 
# Target is 2000 words, but we load in batches of 10 for display
REQUIRED_WORD_COUNT = 2000
LOAD_BATCH_SIZE = 10 

# Pydantic Schema for Structured AI Output - ADDED PRONUNCIATION
class SatWord(BaseModel):
    """Defines the exact structure for the AI-generated vocabulary word."""
    word: str = Field(description="The SAT-level word.")
    pronunciation: str = Field(description="Phonetic transcription (e.g., /ɪˈfɛmərəl/).")
    definition: str = Field(description="The concise dictionary definition.")
    tip: str = Field(description="A short, catchy mnemonic memory tip.")
    usage: str = Field(description="A professional sample usage sentence.")
    sat_level: str = Field(default="High", description="Should always be 'High'.")

# ----------------------------------------------------------------------
# 2. DATA PERSISTENCE & STATE MANAGEMENT
# ----------------------------------------------------------------------

# Initialize Streamlit session state variables
if 'current_user_id' not in st.session_state: st.session_state.current_user_id = None
if 'is_auth' not in st.session_state: st.session_state.is_auth = False
if 'vocab_data' not in st.session_state: st.session_state.vocab_data = []
if 'quiz_active' not in st.session_state: st.session_state.quiz_active = False
if 'words_displayed' not in st.session_state: st.session_state.words_displayed = LOAD_BATCH_SIZE


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
# 3. REAL AI EXTRACTION & LOADING (Gemini API)
# ----------------------------------------------------------------------

def real_llm_vocabulary_extraction(num_words: int, existing_words: List[str]) -> List[Dict]:
    """
    Calls the Gemini API to generate structured vocabulary data not already in the list.
    """
    
    # 1. Define the words to exclude in the prompt
    exclusion_list = ", ".join(existing_words) if existing_words else "none"

    # 2. Define the prompt
    prompt = f"""
    Generate {num_words} unique, extremely high-level SAT vocabulary words.
    The words must NOT be any of the following: {exclusion_list}.
    For each word, provide:
    1. The word itself.
    2. The standard phonetic pronunciation.
    3. A concise dictionary definition.
    4. A short, creative, and memorable mnemonic memory tip.
    5. A professional sample usage sentence.
    
    Return the result as a JSON array where each object strictly conforms to the provided schema.
    """

    # 3. Define the JSON schema for a LIST of SatWord objects
    # This is the guaranteed Pydantic V2 way to get the JSON schema for a list response.
    list_schema = {
        "type": "array",
        "items": SatWord.model_json_schema() # Get the schema for a single item
    }

    # 4. Configure structured output
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_json_schema=list_schema, 
    )

    with st.spinner(f"🤖 Calling Gemini AI to generate {num_words} new SAT words..."):
        try:
            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=config
            )
            
            # 5. Parse the JSON response
            new_data_list = json.loads(response.text)
            
            # 6. Validate the structured data
            validated_words = []
            for item in new_data_list:
                try:
                    # Validate against the Pydantic model
                    word_obj = SatWord(**item)
                    validated_words.append(word_obj.model_dump())
                except ValidationError as e:
                    # This handles network errors, API key errors, or structured output failures
                    st.warning(f"AI generated invalid data for a word (validation error: {e}). Skipping item.")
                    continue
            
            return validated_words

        except Exception as e:
            st.error(f"🔴 Gemini API Extraction Failed: {e}")
            return []

def load_or_extract_initial_vocabulary(required_count: int = LOAD_BATCH_SIZE):
    """
    Loads existing data, extracts new words via AI if needed, and saves the updated list.
    
    Note: This is designed to only check if the *initial* words are present,
          the "Load More" button handles subsequent extraction.
    """
    # 1. Load existing data
    st.session_state.vocab_data = load_vocabulary_from_file()
    word_count = len(st.session_state.vocab_data)
    
    if st.session_state.vocab_data:
        st.info(f"✅ Loaded {word_count} words from local file.")

    # 2. Check if the initial required count (10 words) is met
    if word_count < required_count:
        words_to_extract = required_count - word_count
        st.warning(f"Need {words_to_extract} more words for initial load. Triggering AI extraction...")
        
        existing_words = [d['word'] for d in st.session_state.vocab_data]
        
        # 3. Call the real AI function (limited to the initial missing amount)
        new_words = real_llm_vocabulary_extraction(words_to_extract, existing_words)
        
        if new_words:
            # 4. Add new words to the list and save
            st.session_state.vocab_data.extend(new_words)
            save_vocabulary_to_file(st.session_state.vocab_data)
            st.success(f"✅ Added {len(new_words)} words. Total words available: {len(st.session_state.vocab_data)}")
        else:
            st.error("Could not generate new words. Check API key and internet connection.")
    else:
        st.info(f"✅ Ready to start.")

# --- Login Handler (Defined before main() to prevent NameError) ---
def handle_login(user_id, password):
    """Mocks email/password verification."""
    # Since we cannot use real Firebase Auth, this mocks a successful login
    # if a simple validation is met (e.g., non-empty fields).
    
    if user_id and password:
        # A simple mock success criteria
        st.session_state.current_user_id = user_id
        st.session_state.is_auth = True
        st.session_state.words_displayed = LOAD_BATCH_SIZE # Reset display count on login
        st.success(f"Logged in as: {user_id}! Access granted.")
        # Trigger the initial load/extraction after login
        load_or_extract_initial_vocabulary(required_count=LOAD_BATCH_SIZE)
    else:
        st.error("Please enter a valid Email and Password.")

def load_more_words():
    """Increments the displayed word count and triggers extraction if needed."""
    
    new_display_count = st.session_state.words_displayed + LOAD_BATCH_SIZE
    
    # Check if we need to extract new words to meet the display requirement
    if new_display_count > len(st.session_state.vocab_data):
        
        words_to_extract = new_display_count - len(st.session_state.vocab_data)
        
        # We only extract up to the LOAD_BATCH_SIZE in one go
        extraction_limit = min(words_to_extract, LOAD_BATCH_SIZE)
        
        st.info(f"Automatically extracting {extraction_limit} new words...")
        
        existing_words = [d['word'] for d in st.session_state.vocab_data]
        new_words = real_llm_vocabulary_extraction(extraction_limit, existing_words)
        
        if new_words:
            st.session_state.vocab_data.extend(new_words)
            save_vocabulary_to_file(st.session_state.vocab_data)
            st.success(f"✅ Added {len(new_words)} words. Total: {len(st.session_state.vocab_data)}")
        else:
            # If extraction fails, we still try to increase the display count to show existing words
            pass 

    st.session_state.words_displayed = new_display_count
    st.rerun()

# ----------------------------------------------------------------------
# 4. MAIN FEATURES: UI COMPONENTS
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
            word = data.get('word', 'N/A').upper()
            pronunciation = data.get('pronunciation', 'N/A')
            definition = data.get('definition', 'N/A')
            tip = data.get('tip', 'N/A')
            usage = data.get('usage', 'N/A')
            
            with st.expander(f"**{word}** - {pronunciation}"):
                st.markdown(f"**Definition:** {definition.capitalize()}")
                st.markdown(f"**Memory Tip:** *{tip}*")
                st.markdown(f"**Usage:** *'{usage}'*")

    # Load More Button
    if words_to_show < REQUIRED_WORD_COUNT:
        if st.button(f"Load {LOAD_BATCH_SIZE} More Words", on_click=load_more_words, type="secondary"):
            pass


def start_new_quiz():
    """Initializes the quiz in session state."""
    words = st.session_state.vocab_data
    QUIZ_SIZE = 10 # Quiz size is 10 questions
    
    if len(words) < 4:
        st.error("Not enough words for a meaningful quiz. Need at least 4 unique words.")
        return

    quiz_words = random.sample(words, min(QUIZ_SIZE, len(words)))
    
    st.session_state.quiz_data = quiz_words
    st.session_state.current_question_index = 0
    st.session_state.score = 0
    st.session_state.quiz_active = True
    st.session_state.quiz_feedback = ""

def generate_quiz_ui():
    """Renders the Quiz Section feature."""
    st.header("📝 Vocabulary Quiz", divider="green")

    if not st.session_state.vocab_data or len(st.session_state.vocab_data) < 4:
        st.info("A minimum of 4 words is required to start a quiz.")
        return

    if not st.session_state.quiz_active:
        st.button("Start New Quiz", on_click=start_new_quiz, type="primary")
        return
    
    # --- Active Quiz Logic ---
    q_index = st.session_state.current_question_index
    quiz_data = st.session_state.quiz_data
    total_questions = len(quiz_data)
    
    if q_index >= total_questions:
        st.balloons()
        st.success(f"Quiz Complete! Final Score: **{st.session_state.score}** out of **{total_questions}**")
        st.session_state.quiz_active = False # Reset quiz
        st.session_state.quiz_feedback = ""
        st.button("Try Another Quiz", on_click=start_new_quiz)
        return

    question_data = quiz_data[q_index]
    
    st.markdown(f"**Question {q_index + 1} of {total_questions}**")
    st.subheader(f"Define: **{question_data['word'].upper()}**")
    
    correct_answer = question_data['definition'].capitalize()
    
    # Select 3 incorrect definitions (decoys)
    all_words = st.session_state.vocab_data
    decoys = []
    while len(decoys) < 3:
        decoy_word = random.choice(all_words)
        decoy_definition = decoy_word['definition'].capitalize()
        # Ensure decoy is not the correct answer and is not already a decoy
        if decoy_definition != correct_answer and decoy_definition not in decoys:
            decoys.append(decoy_definition)

    options = [correct_answer] + decoys
    random.shuffle(options)
    
    with st.form(key=f"quiz_form_{q_index}"):
        user_choice = st.radio("Select the correct definition:", options, index=None)
        submitted = st.form_submit_button("Submit Answer")

        if submitted:
            if user_choice is None:
                st.error("Please select an option before submitting.")
            else:
                if user_choice == correct_answer:
                    st.session_state.score += 1
                    st.session_state.quiz_feedback = f"🎉 **Correct!** Score: {st.session_state.score}/{q_index + 1}"
                else:
                    st.session_state.quiz_feedback = f"😔 **Incorrect.** The correct answer was: **{correct_answer}**\n\n**Tip:** {question_data['tip']}\n\n**Usage:** *'{question_data['usage']}'*"
                
                # Move to next question
                st.session_state.current_question_index += 1
                st.rerun()
    
    if st.session_state.quiz_feedback:
        st.markdown(st.session_state.quiz_feedback)

def admin_extraction_ui():
    """Renders the Admin Extraction feature."""
    st.header("💡 AI Extraction & Data Management", divider="orange")
    
    st.markdown(f"""
    **Total Words in Database:** `{len(st.session_state.vocab_data)}` (Target: {REQUIRED_WORD_COUNT}).
    
    The application uses the Gemini AI to generate new vocabulary and saves it to **`{JSON_FILE_PATH}`**.
    """)

    # Admin Action: Manually trigger extraction
    if st.button("Manually Extract 5 New Words (Real AI Call)", type="secondary"):
        
        existing_words = [d['word'] for d in st.session_state.vocab_data]
        new_batch = real_llm_vocabulary_extraction(5, existing_words)
        
        if new_batch:
            st.session_state.vocab_data.extend(new_batch)
            save_vocabulary_to_file(st.session_state.vocab_data)
            st.success(f"✅ Extracted and added {len(new_batch)} words. Total: {len(st.session_state.vocab_data)}")
            # Reset display to current count if extraction occurs
            st.session_state.words_displayed = len(st.session_state.vocab_data) 
            st.rerun()
        else:
            st.error("Failed to generate new words. Check API key and internet connection.")

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
            # Login form using email/password
            user_input = st.text_input("📧 Email", key="user_email_input", value="jamshaid@example.com")
            password_input = st.text_input("🔑 Password", type="password", key="password_input", value="password123")
            
            st.button("Login", on_click=handle_login, args=(user_input, password_input), type="primary")
            st.caption("Mock login: Just enter text in both fields to proceed.")
        else:
            st.success(f"Logged in as: **{st.session_state.current_user_id}**")
            if st.button("Log Out"):
                st.session_state.is_auth = False
                st.session_state.current_user_id = None
                st.session_state.quiz_active = False
                st.session_state.words_displayed = LOAD_BATCH_SIZE # Reset display
                st.rerun()
                
    # --- Main Content ---
    if not st.session_state.is_auth:
        st.info("Please log in using the sidebar to access the Vocabulary Builder.")
    else:
        # Initial check to load 10 words if none are loaded
        if not st.session_state.vocab_data:
            load_or_extract_initial_vocabulary(required_count=LOAD_BATCH_SIZE)
            
        # Use tabs for the main features
        tab_display, tab_quiz, tab_admin = st.tabs(["📚 Vocabulary List", "📝 Quiz Section", "🛠️ AI Tools"])
        
        with tab_display:
            display_vocabulary_ui()
            
        with tab_quiz:
            generate_quiz_ui()

        with tab_admin:
            admin_extraction_ui()

if __name__ == "__main__":
    main()
