
import json
import time
import random
import sys
import os
import base64
import urllib.parse 
from typing import List, Dict, Optional
import streamlit as st
# 🛑 REMOVING google-genai for TTS
from pydantic import BaseModel, Field, ValidationError
from pydantic import json_schema 

# 🟢 NEW: Import gTTS and io for open-source TTS solution
try:
    from gtts import gTTS
    import io
except ImportError:
    st.error("ERROR: The 'gtts' library is required for open-source TTS.")
    st.error("Please ensure 'gtts' is in your requirements.txt file.")
    st.stop()
    
# --- GEMS API (Still required for text extraction) ---
try:
    from google import genai
    from google.genai import types
except ImportError:
    # This block is for redundancy, main error is caught by gTTS import block
    st.error("ERROR: The 'google-genai' and 'pydantic' libraries are required.")
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
    # We still need the Gemini client for text extraction
    gemini_client = genai.Client()
except Exception as e:
    st.error(f"🔴 Failed to initialize Gemini Client: {e}")
    st.stop()

# --- App State and Constants ---
# Use a local JSON file for persistent storage
JSON_FILE_PATH = "vocab_data.json" 
REQUIRED_WORD_COUNT = 2000
LOAD_BATCH_SIZE = 10 
# The target for automatic extraction is now the full 2000 words
AUTO_EXTRACT_TARGET_SIZE = REQUIRED_WORD_COUNT 
QUIZ_SIZE = 5 

# Admin Configuration (Mock Login)
ADMIN_EMAIL = "roy.jamshaid@gmail.com" 
ADMIN_PASSWORD = "Jamshaid,1981" 
# Manual extraction batch size set to 50 for the Admin button
MANUAL_EXTRACT_BATCH = 50 


# Pydantic Schema for Vocabulary Word
class SatWord(BaseModel):
    word: str = Field(description="The SAT-level word.")
    pronunciation: str = Field(description="Simple, hyphenated phonetic pronunciation (e.g., eh-FEM-er-al).")
    definition: str = Field(description="The concise dictionary definition.")
    tip: str = Field(description="A short, catchy mnemonic memory tip.")
    usage: str = Field(description="A professional sample usage sentence.")
    sat_level: str = Field(default="High", description="Should always be 'High'.")
    # The audio field now stores MP3 Base64 data from gTTS
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

# 🛑 DELETING pcm_to_wav because gTTS produces MP3, not raw PCM.

def generate_tts_audio(text: str) -> Optional[str]:
    """Generates audio via gTTS and returns Base64 encoded MP3 data."""
    try:
        # 🟢 Using gTTS to generate the speech
        tts = gTTS(text=text, lang='en', slow=False)
        
        # Save to an in-memory file
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        
        # Encode the MP3 bytes to base64
        base64_data = base64.b64encode(mp3_fp.read()).decode('utf-8')
        return base64_data

    except Exception as e:
        # gTTS network failure will land here, showing a severe error
        print(f"gTTS Generation failed for word: {text}. Error: {e}")
        st.error(f"TTS Audio Error: {e}")
        return None

def real_llm_vocabulary_extraction(num_words: int, existing_words: List[str]) -> List[Dict]:
    """
    1. Calls the Gemini API to generate structured vocabulary data.
    2. Calls the gTTS library for each word to generate and encode the audio.
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
        
    # --- Step 2: Generate and Attach Audio Data (gTTS) ---
    words_with_audio = []
    
    # We use a progress bar for the slow step
    progress_bar = st.progress(0, text=f"Generating TTS audio for 0 of {len(validated_words)} words...")
    
    for i, word_data in enumerate(validated_words):
        word = word_data['word']
        
        # Call the TTS function (now using gTTS)
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

def handle_manual_word_entry(word: str):
    """Generates pronunciation and LLM content for a single word and saves it."""
    
    if not word:
        st.error("Please enter a word.")
        return

    st.info(f"Generating data for '{word}'...")
    
    # 1. Generate Pronunciation Audio (Manual Step 1: gTTS)
    audio_data = generate_tts_audio(word)
    if not audio_data:
        st.error(f"🔴 Failed to generate pronunciation audio for '{word}'. Please check API key and retry.")
        return

    # 2. Generate Definition, Tip, and Usage (Manual Step 2: LLM)
    prompt = f"Generate the pronunciation, definition, mnemonic tip, and a usage sentence for the high-level SAT word: {word}. Return only the JSON object."
    
    try:
        list_schema = {"type": "array", "items": SatWord.model_json_schema()}
        config = types.GenerateContentConfig(response_mime_type="application/json", response_json_schema=list_schema)
        
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt, config=config
        )
        
        # Parse the JSON response
        data_list = json.loads(response.text)
        if not data_list or not isinstance(data_list, list) or 'word' not in data_list[0]:
            raise ValueError("LLM returned invalid data structure.")
            
        new_word_data = SatWord(**data_list[0]).model_dump()
        
    except Exception as e:
        st.error(f"🔴 Failed to generate content for '{word}'. Error: {e}")
        return

    # 3. Combine Data and Save
    new_word_data['audio_base64'] = audio_data
    
    if new_word_data['word'].lower() != word.lower():
        st.warning(f"Note: LLM corrected the word to '{new_word_data['word']}'. Using LLM's version.")

    st.session_state.vocab_data.insert(0, new_word_data) # Insert at top for visibility
    save_vocabulary_to_file(st.session_state.vocab_data)
    st.success(f"✅ Manually added '{new_word_data['word']}' with working pronunciation!")
    st.rerun()

def handle_fix_single_audio(word_index: int):
    """Generates missing audio for a single word and saves the update."""
    
    if word_index < 0 or word_index >= len(st.session_state.vocab_data):
        st.error("Invalid word index.")
        return
        
    word_data = st.session_state.vocab_data[word_index]
    word = word_data['word']
    
    st.info(f"Attempting to fix pronunciation for '{word}'...")
    
    # Generate the missing audio
    audio_data = generate_tts_audio(word)
    
    if audio_data:
        st.session_state.vocab_data[word_index]['audio_base64'] = audio_data
        save_vocabulary_to_file(st.session_state.vocab_data)
        st.success(f"✅ Successfully fixed audio for '{word}'.")
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
    
    # Note: Using st.session_state.vocab_data directly for updates
    for i, index in enumerate(words_to_fix_indices):
        word_data = st.session_state.vocab_data[index]
        word = word_data['word']

        status_placeholder.progress((i + 1) / total_count, 
                                    text=f"Fixing {word}... ({i + 1}/{total_count} processed)")

        # 1. Generate the missing audio (using gTTS)
        audio_data = generate_tts_audio(word)

        if audio_data:
            st.session_state.vocab_data[index]['audio_base64'] = audio_data
            fixed_count += 1
        # If gTTS fails, we skip this word and continue to the next one
        time.sleep(0.5) # Throttle to prevent immediate session timeout/rate limiting

    # 2. Save all successful changes in one go
    if fixed_count > 0:
        save_vocabulary_to_file(st.session_state.vocab_data)
        st.success(f"✅ Bulk fix complete! Successfully repaired audio for {fixed_count} of {total_count} words.")
    else:
        st.error(f"🔴 Bulk fix attempted, but audio generation failed for all {total_count} words. Check server logs/quotas.")
        
    status_placeholder.empty()
    st.rerun()


def fill_missing_audio(vocab_data: List[Dict]) -> bool:
    """
    Iterates through existing vocabulary and generates missing audio for corrupted entries.
    Returns True if any changes were made and saved, forcing a rerun.
    """
    words_to_fix = [d for d in vocab_data if d.get('audio_base64') is None]
    if not words_to_fix:
        return False

    st.warning(f"Audio Integrity Check: Found {len(words_to_fix)} words missing pronunciation. Use the 'Fix Audio' button next to each word or the 'Bulk Fix' tool.")
    
    return False # Do not force rerun for bulk fix anymore, let user use button


def load_and_update_vocabulary_data():
    """
    Loads existing data, checks for missing audio, and implements aggressive goal-seeking extraction.
    """
    if not st.session_state.is_auth: return
    
    st.session_state.vocab_data = load_vocabulary_from_file()
    
    # 1. CONTINUOUS PRONUNCIATION CHECK: Check and display warning, but don't force bulk fix.
    fill_missing_audio(st.session_state.vocab_data)
        
    word_count = len(st.session_state.vocab_data)
    
    if word_count > 0:
        st.info(f"✅ Loaded {word_count} words from local file.")
    
    # 2. AGGRESSIVE WORD EXTRACTION: Fill the vocabulary up to the 2000-word target
    if word_count < AUTO_EXTRACT_TARGET_SIZE:
        words_needed = AUTO_EXTRACT_TARGET_SIZE - word_count
        
        # Determine the number of words to extract in this single session run.
        # We cap it at LOAD_BATCH_SIZE (10) for stability.
        num_to_extract = min(LOAD_BATCH_SIZE, words_needed)
        
        if num_to_extract > 0:
            st.warning(f"Goal: {AUTO_EXTRACT_TARGET_SIZE} words. Extracting next {num_to_extract} words now...")
            
            # Blocking extraction 
            existing_words = [d['word'] for d in st.session_state.vocab_data]
            new_words = real_llm_vocabulary_extraction(num_to_extract, existing_words)
            
            if new_words:
                st.session_state.vocab_data.extend(new_words)
                save_vocabulary_to_file(st.session_state.vocab_data)
                st.success(f"✅ Added {len(new_words)} words. Current total: {len(st.session_state.vocab_data)}.")
                # CRITICAL: Rerun to immediately check if the goal is met or extract the next batch
                st.rerun() 
            else:
                st.error("🔴 Failed to generate new words. Check API key and logs.")
    
    # This block handles the very first load when word_count is 0 and no continuous extraction has run
    elif word_count < LOAD_BATCH_SIZE:
        st.warning(f"Need {LOAD_BATCH_SIZE} words for initial display. Triggering extraction...")
        # Blocking extraction for initial display 
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
                
                # --- AUDIO PLAYBACK (USES BASE64 MP3 DATA from gTTS) ---
                if audio_base64:
                    # MIME type must be changed to audio/mp3 for gTTS output
                    audio_data_url = f"data:audio/mp3;base64,{audio_base64}"
                    audio_html = f"""
                        <audio controls style="width: 100%;" src="{audio_data_url}">
                            Your browser does not support the audio element.
                        </audio>
                    """
                    st.markdown(audio_html, unsafe_allow_html=True)
                else:
                    # Show Fix Audio Button if audio is missing
                    st.warning("Audio not available for this word. TTS generation may have failed.")
                    
                    if st.session_state.is_admin:
                        # Add a button unique to this word's index
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
    st.subheader("Vocabulary Extraction (Bulk)")
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
