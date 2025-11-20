import json
import time
import random
import sys
import os
import base64
from typing import List, Dict, Optional
import streamlit as st
from pydantic import BaseModel, Field, ValidationError
from pydantic import json_schema 
# --- New Imports for gTTS and Audio Processing ---
try:
    from gtts import gTTS
    import io
except ImportError:
    # This block ensures the app stops if gtts is missing, preventing later errors.
    st.error("ERROR: The 'gtts' library is required.")
    st.error("Please ensure it is in requirements.txt and installed.")
    st.stop()
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
# Target is 2000 words, but we load in batches of 10 for display
REQUIRED_WORD_COUNT = 2000
LOAD_BATCH_SIZE = 10 

# Pydantic Schema for Structured AI Output - UPDATED to store audio base64
class SatWord(BaseModel):
    """Defines the exact structure for the AI-generated vocabulary word."""
    word: str = Field(description="The SAT-level word.")
    # Asking for a more TTS-friendly pronunciation format
    pronunciation: str = Field(description="Simple, hyphenated phonetic pronunciation (e.g., eh-FEM-er-al).")
    definition: str = Field(description="The concise dictionary definition.")
    tip: str = Field(description="A short, catchy mnemonic memory tip.")
    usage: str = Field(description="A professional sample usage sentence.")
    sat_level: str = Field(default="High", description="Should always be 'High'.")
    # 🟢 NEW FIELD: To store the audio base64 string permanently
    audio_base64: Optional[str] = Field(default=None, description="Base64 encoded MP3 audio data for pronunciation.")

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
# 3. AI EXTRACTION & gTTS FUNCTIONS
# ----------------------------------------------------------------------

def generate_gtts_audio(text: str) -> Optional[str]:
    """
    Generates audio using gTTS, converts it to base64, and returns the string.
    This relies only on gTTS and standard Python IO, making it Streamlit Cloud friendly.
    """
    try:
        # 1. Create gTTS object and save to an in-memory byte buffer
        tts = gTTS(text=text, lang='en')
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        
        # 2. Read the MP3 audio bytes
        audio_bytes = mp3_fp.read()
        
        # 3. Encode to base64
        base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
        return base64_audio

    except Exception as e:
        # Logging the error locally for debugging, but failing silently in the app UI
        # st.error(f"gTTS Audio Generation Failed for '{text}': {e}")
        return None

def real_llm_vocabulary_extraction(num_words: int, existing_words: List[str]) -> List[Dict]:
    """
    1. Calls the Gemini API to generate structured vocabulary data.
    2. Calls the gTTS library for each new word and stores the audio data.
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
        # ... (Text extraction logic remains the same)
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


    # --- Step 2: Generate and Attach Audio Data using gTTS ---
    
    words_with_audio = []
    
    with st.spinner(f"🔊 Generating gTTS audio for {len(validated_words)} words..."):
        tts_progress = st.progress(0, text="Generating Audio...")
        
        for i, word_data in enumerate(validated_words):
            word = word_data['word']
            
            # 🟢 NEW: Use the local Python library for audio generation
            audio_base64 = generate_gtts_audio(word)
            
            # Attach the base64 string directly to the word's data structure
            word_data['audio_base64'] = audio_base64
            words_with_audio.append(word_data)

            # Update progress bar
            tts_progress.progress((i + 1) / len(validated_words), text=f"Generated audio for {word}...")
            
        tts_progress.empty() # Clear progress bar once finished

    return words_with_audio

def load_or_extract_initial_vocabulary(required_count: int = LOAD_BATCH_SIZE):
    """
    Loads existing data, extracts new words via AI if needed, and saves the updated list.
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
            audio_base64 = data.get('audio_base64') # 🟢 READ AUDIO DIRECTLY FROM DATABASE
            
            
            with st.expander(f"**{word}** - {pronunciation}"):
                
                # --- AUDIO PLAYBACK ---
                if audio_base64:
                    # Streamlit can play base64-encoded audio directly using HTML audio tag
                    # We specify the MIME type as MP3 since gTTS generates MP3
                    audio_html = f"""
                        <audio controls style="width: 100%;">
                            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                            Your browser does not support the audio element.
                        </audio>
                    """
                    st.markdown(audio_html, unsafe_allow_html=True)
                else:
                    st.warning("Audio not available for this word.")
                # -----------------------

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
    
    The application uses the Gemini AI (for text) and **gTTS (for audio)** to generate and save data to **`{JSON_FILE_PATH}`**.
    """)
    
    # --- Manual TTS Test Tool ---
    st.subheader("🔊 Audio Pronunciation Test")
    st.markdown("This uses the robust **gTTS** system. Test here to confirm audio libraries are working.")
    test_word = st.text_input("Enter word to test audio:", value="ephemeral")
    
    if st.button("Test Pronunciation"):
        if test_word:
            with st.spinner(f"Testing audio for '{test_word}'..."):
                test_audio = generate_gtts_audio(test_word)
            
            if test_audio:
                audio_html = f"""
                    <audio controls autoplay style="width: 100%;">
                        <source src="data:audio/mp3;base64,{test_audio}" type="audio/mp3">
                        Your browser does not support the audio element.
                    </audio>
                """
                st.markdown(audio_html, unsafe_allow_html=True)
                st.success(f"Audio stream successfully generated for '{test_word}'.")
            else:
                st.error(f"Failed to generate audio stream for '{test_word}'. Check the terminal for gTTS errors.")
        else:
            st.warning("Please enter a word to test.")

    st.markdown("---")
    
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
