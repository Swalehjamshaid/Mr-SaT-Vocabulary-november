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
if 'audio_data_cache' not in st.session_state: st.session_state.audio_data_cache = {} # Cache for base64 audio data


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
# 3. AI EXTRACTION & TTS FUNCTIONS
# ----------------------------------------------------------------------

def call_tts_api(text: str, max_retries: int = 3) -> Optional[str]:
    """
    Calls the Gemini TTS API to get base64 encoded PCM audio data with retries.
    Caches results to avoid repeated API calls and implements exponential backoff.
    """
    if text in st.session_state.audio_data_cache:
        return st.session_state.audio_data_cache[text]

    # Only show the info message when running locally
    if 'streamlit run' in sys.argv[0]:
        st.info(f"Attempting to generate pronunciation audio for: **{text}**")
    
    for attempt in range(max_retries):
        try:
            # Use a cheerful voice for a positive learning experience (Zephyr)
            payload = {
                "contents": [{
                    "parts": [{ "text": f"Say clearly and slowly: {text}" }]
                }],
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {
                        "voiceConfig": {
                            "prebuiltVoiceConfig": { "voiceName": "Zephyr" }
                        }
                    }
                }
            }
            
            # NOTE: Using a separate client here for the TTS model 
            tts_client = genai.Client()
            response = tts_client.models.generate_content(
                model="gemini-2.5-flash-preview-tts",
                contents=payload["contents"],
                config=payload["generationConfig"]
            )

            part = response.candidates[0].content.parts[0]
            audio_data = part.inlineData.data
            
            st.session_state.audio_data_cache[text] = audio_data
            return audio_data

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff (1s, 2s)
                time.sleep(wait_time)
            else:
                # st.error(f"🔴 TTS API Failed permanently for '{text}' after {max_retries} attempts.")
                return None

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
                    word_obj = SatWord(**item)
                    validated_words.append(word_obj.model_dump())
                except ValidationError as e:
                    st.warning(f"AI generated invalid data for a word (validation error: {e}). Skipping item.")
                    continue
            
            return validated_words

        except Exception as e:
            st.error(f"🔴 Gemini API Extraction Failed: {e}")
            return []

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

# --- JAVASCRIPT FOR PCM TO WAV CONVERSION (Required for TTS) ---
JS_PCM_TO_WAV = """
<script>
    function base64ToArrayBuffer(base64) {
        var binaryString = atob(base64);
        var len = binaryString.length;
        var bytes = new Int8Array(len);
        for (var i = 0; i < len; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }
        return bytes.buffer;
    }

    function pcmToWav(pcm16, sampleRate) {
        var buffer = new ArrayBuffer(44 + pcm16.length * 2);
        var view = new DataView(buffer);
        var pcmLength = pcm16.length * 2;
        var channels = 1;
        var bitsPerSample = 16;
        var byteRate = sampleRate * channels * (bitsPerSample / 8);

        // RIFF chunk descriptor
        writeString(view, 0, 'RIFF');
        view.setUint32(4, 36 + pcmLength, true);
        writeString(view, 8, 'WAVE');

        // FMT chunk
        writeString(view, 12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true); // Linear PCM
        view.setUint16(22, channels, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, byteRate, true);
        view.setUint16(32, channels * (bitsPerSample / 8), true);
        view.setUint16(34, bitsPerSample, true);

        // Data chunk
        writeString(view, 36, 'data');
        view.setUint32(40, pcmLength, true);

        // Write PCM data
        var offset = 44;
        for (var i = 0; i < pcm16.length; i++, offset += 2) {
            view.setInt16(offset, pcm16[i], true);
        }

        return new Blob([view], { type: 'audio/wav' });
    }

    function writeString(view, offset, string) {
        for (var i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    }
    
    // --- MAIN PLAY FUNCTION ---
    window.playAudio = function(base64Data, sampleRate = 24000) {
        try {
            var pcmData = base64ToArrayBuffer(base64Data);
            // The API returns 16-bit signed PCM, which we need to convert to an Int16Array
            var pcm16 = new Int16Array(pcmData.slice(0)); 
            var wavBlob = pcmToWav(pcm16, sampleRate);
            
            var audioUrl = URL.createObjectURL(wavBlob);
            var audio = new Audio(audioUrl);
            audio.play();

        } catch(e) {
            console.error("Error playing audio:", e);
            alert("Could not play audio. Check console for details.");
        }
    }
</script>
"""
st.components.v1.html(JS_PCM_TO_WAV, height=0)


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
                
                # --- TTS PLAY BUTTON ---
                audio_base64 = call_tts_api(word)
                if audio_base64:
                    # Create the JavaScript call string
                    js_call = f'window.playAudio("{audio_base64}");'
                    
                    # Embed the button and call the JS function on click
                    st.markdown(f"""
                        <button 
                            onclick='{js_call}' 
                            style='background-color: #4CAF50; color: white; padding: 5px 10px; border: none; border-radius: 4px; cursor: pointer; margin-bottom: 10px;'>
                            🔊 Listen
                        </button>
                    """, unsafe_allow_html=True)
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
    
    The application uses the Gemini AI to generate new vocabulary and saves it to **`{JSON_FILE_PATH}`**.
    """)
    
    # --- Manual TTS Test Tool ---
    st.subheader("🔊 Audio Pronunciation Test")
    test_word = st.text_input("Enter word to test audio:", value="ephemeral")
    
    if st.button("Test Pronunciation"):
        if test_word:
            with st.spinner(f"Testing audio for '{test_word}'..."):
                test_audio = call_tts_api(test_word, max_retries=3)
            
            if test_audio:
                # Use the embedded JS player for playback
                js_call = f'window.playAudio("{test_audio}");'
                st.markdown(f"""
                    <button 
                        onclick='{js_call}' 
                        style='background-color: #007BFF; color: white; padding: 5px 10px; border: none; border-radius: 4px; cursor: pointer;'>
                        ▶️ Play "{test_word}"
                    </button>
                """, unsafe_allow_html=True)
                st.success(f"Audio stream successfully generated for '{test_word}'. If you don't hear anything, check your browser permissions.")
            else:
                st.error(f"Failed to generate audio stream for '{test_word}'. The TTS server may be overloaded. Try again in a minute.")
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
