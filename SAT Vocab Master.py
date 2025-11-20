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

# Check for API Key (Still required for text extraction)
if "GEMINI_API_KEY" not in os.environ and not ("__initial_auth_token__" in globals() and __initial_auth_token__):
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
    # Client will automatically read the GEMINI_API_KEY environment variable
    gemini_client = genai.Client()
except Exception as e:
    st.error(f"🔴 Failed to initialize Gemini Client: {e}")
    st.stop()

# Use a local JSON file for persistent storage
JSON_FILE_PATH = "vocab_data.json" 
# Target is the ultimate goal, but we use a smaller target for AUTO-EXTRACTION
REQUIRED_WORD_COUNT = 2000
LOAD_BATCH_SIZE = 10 
# 🟢 CHANGE: New target size for automatic extraction upon startup
AUTO_EXTRACT_TARGET_SIZE = 200 
AUTO_EXTRACT_BATCH = 5 # How many words to extract automatically if under target

# Pydantic Schema for Structured AI Output - UPDATED to store audio URL
class SatWord(BaseModel):
    """Defines the exact structure for the AI-generated vocabulary word."""
    word: str = Field(description="The SAT-level word.")
    pronunciation: str = Field(description="Simple, hyphenated phonetic pronunciation (e.g., eh-FEM-er-al).")
    definition: str = Field(description="The concise dictionary definition.")
    tip: str = Field(description="A short, catchy mnemonic memory tip.")
    usage: str = Field(description="A professional sample usage sentence.")
    sat_level: str = Field(default="High", description="Should always be 'High'.")
    # 🟢 FINAL AUDIO FIELD: Store the public URL instead of base64
    audio_url: Optional[str] = Field(default=None, description="Public URL for MP3 audio pronunciation.")

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
# 3. AI EXTRACTION & AUDIO FUNCTIONS
# ----------------------------------------------------------------------

def construct_tts_url(text: str) -> str:
    """
    Constructs a reliable TTS URL using Google's public endpoint.
    This avoids the server-side network block.
    """
    # URL is generated based on the text, permanently storing the link.
    encoded_text = urllib.parse.quote(text)
    return f"https://translate.google.com/translate_tts?ie=UTF-8&q={encoded_text}&tl=en&client=tw-ob"


def real_llm_vocabulary_extraction(num_words: int, existing_words: List[str]) -> List[Dict]:
    """
    1. Calls the Gemini API to generate structured vocabulary data.
    2. Constructs the audio URL for each new word and stores it.
    """
    
    # --- Step 1: Generate Text Data (Remains the same using Gemini) ---
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

    with st.spinner(f"🤖 Calling Gemini AI for text generation of {num_words} words..."):
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


    # --- Step 2: Generate and Attach Audio URL ---
    
    words_with_audio = []
    
    with st.spinner(f"🔗 Constructing audio links for {len(validated_words)} words..."):
        
        for word_data in validated_words:
            word = word_data['word']
            
            # 🟢 FINAL FIX: Store the direct audio URL instead of relying on server-side generation
            audio_url = construct_tts_url(word)
            
            # Attach the URL directly to the word's data structure
            word_data['audio_url'] = audio_url
            words_with_audio.append(word_data)

    return words_with_audio

def load_or_extract_initial_vocabulary(required_count: int = LOAD_BATCH_SIZE):
    """
    Loads existing data, extracts new words via AI if needed, and saves the updated list.
    Also ensures old words are updated with the new audio URL scheme.
    """
    # 1. Load existing data
    st.session_state.vocab_data = load_vocabulary_from_file()
    word_count = len(st.session_state.vocab_data)
    
    # 🔴 Fix for old data: If old words don't have the audio_url field, generate it now.
    words_missing_url = [d for d in st.session_state.vocab_data if d.get('audio_url') is None]
    if words_missing_url:
        st.info(f"Updating {len(words_missing_url)} existing words with reliable audio links.")
        
        with st.spinner("Updating audio links..."):
            for word_data in words_missing_url:
                word_data['audio_url'] = construct_tts_url(word_data['word'])
            
        # Save the updated list (with newly generated audio URL)
        save_vocabulary_to_file(st.session_state.vocab_data)
        st.success(f"Successfully updated {len(words_missing_url)} words with audio links.")


    if st.session_state.vocab_data:
        st.info(f"✅ Loaded {word_count} words from local file.")

    # 🟢 CHANGE: Automatic extraction logic on startup if word count is below target
    if word_count < AUTO_EXTRACT_TARGET_SIZE:
        
        # Calculate how many words we need, capped at our small batch size
        words_to_extract = min(AUTO_EXTRACT_TARGET_SIZE - word_count, AUTO_EXTRACT_BATCH)
        
        if words_to_extract > 0:
            st.info(f"Auto-extracting {words_to_extract} new words in the background...")
            
            existing_words = [d['word'] for d in st.session_state.vocab_data]
            
            # 3. Call the real AI function
            new_words = real_llm_vocabulary_extraction(words_to_extract, existing_words)
            
            if new_words:
                # 4. Add new words to the list and save
                st.session_state.vocab_data.extend(new_words)
                save_vocabulary_to_file(st.session_state.vocab_data)
                st.success(f"✅ Auto-extracted {len(new_words)} words. Total words available: {len(st.session_state.vocab_data)}")
            else:
                st.error("Could not automatically generate new words.")
    
    # 2. Check if the initial required count (10 words for display) is met
    if word_count < required_count:
        st.info("Initial words for display are loaded.")
    else:
        st.info(f"✅ Ready to start. Database size: {len(st.session_state.vocab_data)}")

# --- Login Handler (Defined before main() to prevent NameError) ---
def handle_login(user_id, password):
    """Mocks email/password verification."""
    
    if user_id and password:
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
        
        st.info(f"Manually extracting {extraction_limit} new words...")
        
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
            # 🟢 CHANGE: Calculate the word number based on the index (i)
            word_number = i + 1 
            
            word = data.get('word', 'N/A').upper()
            pronunciation = data.get('pronunciation', 'N/A')
            definition = data.get('definition', 'N/A')
            tip = data.get('tip', 'N/A')
            usage = data.get('usage', 'N/A')
            audio_url = data.get('audio_url') # 🟢 READ AUDIO URL DIRECTLY FROM DATABASE
            
            
            # 🟢 CHANGE: Prepend the word number to the expander title
            expander_title = f"**{word_number}.** {word} - {pronunciation}"
            
            with st.expander(expander_title):
                
                # --- AUDIO PLAYBACK ---
                if audio_url:
                    # Streamlit can play public audio URLs directly
                    audio_html = f"""
                        <audio controls style="width: 100%;" src="{audio_url}">
                            Your browser does not support the audio element.
                        </audio>
                    """
                    st.markdown(audio_html, unsafe_allow_html=True)
                else:
                    # This message should only appear for old words that failed the URL update
                    st.warning("Audio URL is missing for this word.")
                # -----------------------

                st.markdown(f"**Definition:** {definition.capitalize()}")
                # 🟢 MEMORY TIP DISPLAY: This is already present and working.
                st.markdown(f"**Memory Tip:** *{tip}*") 
                st.markdown(f"**Usage:** *'{usage}'*")

    # Load More Button
    if words_to_show < REQUIRED_WORD_COUNT:
        if st.button(f"Load {LOAD_BATCH_SIZE} More Words", on_click=load_more_words, type="secondary"):
            pass


def start_new_quiz():
    """Initializes the quiz in session state."""
    words = st.session_state.vocab_data
    # 🟢 QUIZ CHANGE: Set QUIZ_SIZE to 5 questions
    QUIZ_SIZE = 5 
    
    if len(words) < 4:
        st.error("Not enough words for a meaningful quiz. Need at least 4 unique words.")
        return

    # Select 5 random words for the quiz
    quiz_words = random.sample(words, min(QUIZ_SIZE, len(words)))
    
    # Store quiz details for accurate scoring later
    quiz_details = []
    all_definitions = [d['definition'].capitalize() for d in words]
    
    for question_data in quiz_words:
        correct_answer = question_data['definition'].capitalize()
        
        # Select 3 unique decoy definitions from the full list
        decoys = random.sample([
            d for d in all_definitions if d != correct_answer
        ], min(3, len([d for d in all_definitions if d != correct_answer])))
        
        options = [correct_answer] + decoys
        random.shuffle(options)
        
        quiz_details.append({
            "word": question_data['word'],
            "correct_answer": correct_answer,
            "tip": question_data['tip'],
            "usage": question_data['usage'],
            "options": options
        })
        
    st.session_state.quiz_details = quiz_details
    st.session_state.quiz_active = True
    st.session_state.quiz_results = None # Store results after submission

# 🟢 QUIZ CHANGE: Function now processes all questions at once
def generate_quiz_ui():
    """Renders the Quiz Section feature."""
    st.header("📝 Vocabulary Quiz", divider="green")
    
    # QUIZ SIZE is 5 questions now
    QUIZ_SIZE = 5
    
    if not st.session_state.vocab_data or len(st.session_state.vocab_data) < 4:
        st.info("A minimum of 4 words is required to start a quiz.")
        return

    if not st.session_state.quiz_active:
        st.button(f"Start New {QUIZ_SIZE}-Question Quiz", on_click=start_new_quiz, type="primary")
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
            
        # Display feedback for each question
        st.subheader("Review Your Answers")
        for i, result in enumerate(st.session_state.quiz_results['feedback']):
            # 🟢 CHANGE: Use i+1 for question numbering in results review
            st.markdown(f"#### **{i+1}. {result['word']}**") 
            st.markdown(f"**Your Answer:** {result['user_choice']}")
            st.markdown(f"**Correct Answer:** {result['correct_answer']}")
            
            if not result['is_correct']:
                 st.markdown(f"**Memory Tip:** *{result['tip']}*")
                 st.markdown(f"**Usage:** *'{result['usage']}'*")
            
            st.markdown("---")
            
        st.session_state.quiz_active = False # Reset quiz state
        st.session_state.quiz_results = None # Clear results
        st.button(f"Start Another {QUIZ_SIZE}-Question Quiz", on_click=start_new_quiz)
        return
    
    # --- Active Quiz Form ---
    
    quiz_details = st.session_state.quiz_details
    
    # 🟢 QUIZ CHANGE: Show all questions in one form
    with st.form(key="full_quiz_form"):
        st.subheader(f"Answer the following {QUIZ_SIZE} questions:")
        
        # Collect user responses in a list
        st.session_state.user_responses = [] 
        
        for i, q in enumerate(quiz_details):
            # 🟢 CHANGE: Use i+1 for question numbering in the active quiz form
            st.markdown(f"#### **{i + 1}. Define: {q['word'].upper()}**") 
            
            # Key must be unique per question
            user_choice = st.radio(
                "Select the correct definition:", 
                q['options'], 
                key=f"quiz_q_{i}", 
                index=None,
                label_visibility="collapsed"
            )
            # Store the selected answer (or None) for submission processing
            st.session_state.user_responses.append(user_choice)

        submitted = st.form_submit_button("Submit All Answers")

        if submitted:
            final_score = 0
            feedback_list = []
            
            # Check if all fields were answered (basic validation)
            if any(response is None for response in st.session_state.user_responses):
                st.error("Please answer ALL questions before submitting.")
                # We return here to keep the form active until all are answered
                return

            # Process responses
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
                    "usage": q['usage']
                })
            
            # Store final results
            st.session_state.quiz_results = {
                "score": final_score,
                "total": QUIZ_SIZE,
                "accuracy": round((final_score / QUIZ_SIZE) * 100, 1),
                "feedback": feedback_list
            }
            # Clear intermediate responses
            del st.session_state.user_responses
            st.rerun()


def admin_extraction_ui():
    """Renders the Admin Extraction feature."""
    # 🟢 CHANGE: Removed "AI Extraction & Data Management" title
    st.header("💡 Data Management", divider="orange") 
    
    st.markdown(f"""
    **Total Words in Database:** `{len(st.session_state.vocab_data)}` (Target: {REQUIRED_WORD_COUNT}).
    
    The application automatically extracts new words upon startup until it reaches {AUTO_EXTRACT_TARGET_SIZE} words.
    """)
    
    st.markdown("---")
    
    # 🔴 Removed Manual Extraction Button (enforcing auto-extraction)
    st.info("Word extraction is now automatic upon app startup until the target is met.")

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
            
            # 🐞 FIX: Corrected typo from on_onclick to on_click
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
