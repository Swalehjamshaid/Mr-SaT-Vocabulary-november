import json
import time
import random
import sys
import os
import base64
import urllib.parse
from typing import List, Dict, Optional
import streamlit as st

# --- Pronunciation Libraries (Python-based) ---
try:
    import pronouncing          # Great for American English phonetic spelling
except ImportError:
    st.error("Installing 'pronouncing' is recommended: pip install pronouncing")
    pronouncing = None

try:
    import eng_to_ipa as ipa    # Fallback: converts English to IPA
except ImportError:
    st.error("Installing 'eng-to-ipa' as fallback: pip install eng-to-ipa")
    ipa = None

# --- Gemini API ---
try:
    from google import genai
    from google.genai import types
except ImportError:
    st.error("ERROR: google-generativeai required: pip install google-generativeai")
    st.stop()

from pydantic import BaseModel, Field

# ======================================================================
# *** CONFIG & API SETUP ***
# ======================================================================
if "GEMINI_API_KEY" not in os.environ:
    st.error("GEMINI_API_KEY is missing! Set it in secrets or environment.")
    st.stop()

gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

JSON_FILE_PATH = "vocab_data.json"
REQUIRED_WORD_COUNT = 2000
LOAD_BATCH_SIZE = 10
AUTO_EXTRACT_TARGET_SIZE = REQUIRED_WORD_COUNT
QUIZ_SIZE = 5
MANUAL_EXTRACT_BATCH = 50

ADMIN_EMAIL = "roy.jamshaid@gmail.com"
ADMIN_PASSWORD = "Jamshaid,1981"

# ======================================================================
# Pydantic Model
# ======================================================================
class SatWord(BaseModel):
    word: str
    pronunciation: str = Field(description="Hyphenated phonetic: e.g., uh-BAWN-duh")
    definition: str
    tip: str
    usage: str
    sat_level: str = "High"
    audio_base64: Optional[str] = None

# ======================================================================
# Pronunciation from Python Libraries (Fast & Local)
# ======================================================================
def get_pronunciation(word: str) -> str:
    """
    Returns simple hyphenated phonetic pronunciation using local Python libraries.
    Falls back gracefully.
    """
    word_clean = word.strip().lower()

    # 1. Try 'pronouncing' library (best quality)
    if pronouncing and hasattr(pronouncing, 'phones_for_word'):
        phones = pronouncing.phones_for_word(word_clean)
        if phones:
            # Convert ARPAbet to simple spelling (e.g., AH0 B AW1 N D AH0 → uh-BAWN-duh)
            arpabet = phones[0]
            simple = pronouncing.stresses(arpabet).replace('0', '').replace('1', '').replace('2', '')
            mapping = {
                'AA': 'ah', 'AE': 'a', 'AH': 'uh', 'AO': 'aw', 'AW': 'ow',
                'AY': 'igh', 'B': 'b', 'CH': 'ch', 'D': 'd', 'DH': 'th',
                'EH': 'e', 'ER': 'ur', 'EY': 'ay', 'F': 'f', 'G': 'g',
                'HH': 'h', 'IH': 'i', 'IY': 'ee', 'JH': 'j', 'K': 'k',
                'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'ng', 'OW': 'oh',
                'OY': 'oy', 'P': 'p', 'R': 'r', 'S': 's', 'SH': 'sh',
                'T': 't', 'TH': 'th', 'UH': 'oo', 'UW': 'oo', 'V': 'v',
                'W': 'w', 'Y': 'y', 'Z': 'z', 'ZH': 'zh'
            }
            parts = simple.split()
            converted = []
            for p in parts:
                converted.append(mapping.get(p, p.lower()))
            result = '-'.join(converted).upper()
            if result.count('-') > 0:
                return result

    # 2. Fallback: eng-to-ipa (gives IPA like /ɪˈfɛmərəl/)
    if ipa:
        try:
            ipa_text = ipa.convert(word_clean)
            if ipa_text != word_clean and ipa_text != "*":  # not failed
                # Convert common IPA to simple spelling
                ipa_map = {
                    'ɪ': 'i', 'iː': 'ee', 'ɛ': 'e', 'æ': 'a', 'ɑː': 'ah', 'ɔː': 'aw',
                    'ʊ': 'oo', 'uː': 'oo', 'ʌ': 'uh', 'ə': 'uh', 'ɜː': 'ur',
                    'eɪ': 'ay', 'aɪ': 'igh', 'ɔɪ': 'oy', 'oʊ': 'oh', 'aʊ': 'ow',
                    'tʃ': 'ch', 'dʒ': 'j', 'θ': 'th', 'ð': 'th', 'ʃ': 'sh', 'ʒ': 'zh',
                    'ŋ': 'ng'
                }
                simple = ipa_text
                for k, v in ipa_map.items():
                    simple = simple.replace(k, v)
                # Clean stress marks and slashes
                simple = simple.replace('ˈ', '').replace('ˌ', '').replace('/', '').strip()
                if simple:
                    return simple.upper()
        except:
            pass

    # 3. Final fallback: use Forvo-style simple guess
    return word.upper() + " (pronunciation unavailable)"

# ======================================================================
# Gemini TTS: PCM → WAV Conversion (unchanged)
# ======================================================================
def pcm_to_wav(pcm_data: bytes, sample_rate: int = 24000) -> bytes:
    channels = 1
    bits_per_sample = 16
    bytes_per_sample = bits_per_sample // 8
    byte_rate = sample_rate * channels * bytes_per_sample
    data_size = len(pcm_data)

    header = b'RIFF'
    header += (36 + data_size).to_bytes(4, 'little')
    header += b'WAVEfmt '
    header += (16).to_bytes(4, 'little')
    header += (1).to_bytes(2, 'little')
    header += channels.to_bytes(2, 'little')
    header += sample_rate.to_bytes(4, 'little')
    header += byte_rate.to_bytes(4, 'little')
    header += bytes_per_sample.to_bytes(2, 'little')
    header += bits_per_sample.to_bytes(2, 'little')
    header += b'data'
    header += data_size.to_bytes(4, 'little')
    return header + pcm_data

def generate_tts_audio(word: str) -> Optional[str]:
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash-preview-tts",
            contents=[{"parts": [{"text": word}]}],
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config={"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Kore"}}}
            )
        )
        audio_part = response.candidates[0].content.parts[0].inlineData
        pcm_b64 = audio_part.data
        mime = audio_part.mimeType
        rate = 24000
        if 'rate=' in mime:
            try:
                rate = int(mime.split('rate=')[1].split(';')[0])
            except:
                pass
        pcm = base64.b64decode(pcm_b64)
        wav = pcm_to_wav(pcm, rate)
        return base64.b64encode(wav).decode('utf-8')
    except Exception as e:
        print(f"TTS failed for {word}: {e}")
        return None

# ======================================================================
# Main Extraction: Gemini for everything EXCEPT pronunciation
# ======================================================================
def real_llm_vocabulary_extraction(num_words: int, existing_words: List[str]) -> List[Dict]:
    prompt = f"""
    Generate {num_words} extremely advanced, rare SAT-level vocabulary words.
    DO NOT include any of these words: {', '.join(existing_words[:100]) if existing_words else 'none'}.

    For each word, provide:
    - definition (concise)
    - tip (short mnemonic)
    - usage (one natural sentence)

    Return valid JSON array of objects with keys: word, definition, tip, usage
    """

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        raw_list = json.loads(response.text)
    except Exception as e:
        st.error(f"Gemini extraction failed: {e}")
        return []

    words_with_data = []
    progress = st.progress(0)
    for i, item in enumerate(raw_list):
        word = item.get("word", "").strip()
        if not word or word in existing_words:
            continue

        # Local pronunciation (fast!)
        pron = get_pronunciation(word)

        # Generate TTS audio via Gemini
        audio_b64 = generate_tts_audio(word)

        entry = {
            "word": word,
            "pronunciation": pron,
            "definition": item.get("definition", "No definition.").capitalize(),
            "tip": item.get("tip", "No tip available."),
            "usage": item.get("usage", "No example sentence.").capitalize(),
            "sat_level": "High",
            "audio_base64": audio_b64
        }
        words_with_data.append(entry)
        progress.progress((i + 1) / len(raw_list))

    progress.empty()
    return words_with_data

# ======================================================================
# Rest of your app (unchanged except small fixes)
# ======================================================================
# [All the session state, load/save, auth, UI functions remain exactly as before]
# ... (display_vocabulary_ui, quiz, admin, etc.)

# Just paste the rest of your original functions below this line:
# - load_vocabulary_from_file()
# - save_vocabulary_to_file()
# - load_and_update_vocabulary_data()
# - handle_auth(), handle_logout()
# - display_vocabulary_ui() → only small change: show pronunciation in bold
# - generate_quiz_ui(), admin_extraction_ui(), main()

# Example small tweak in display_vocabulary_ui():
# Change line:
# expander_title = f"**{word_number}.** {word} - {pronunciation}"
# → Keep as-is, now pronunciation is accurate and fast!

# ======================================================================
# MAIN (unchanged structure)
# ======================================================================
def main():
    st.set_page_config(page_title="AI Vocabulary Builder", layout="wide")
    st.title("AI-Powered SAT Vocabulary Builder")

    # Sidebar login (same as yours)
    with st.sidebar:
        # ... your login code ...

    if not st.session_state.get("is_auth", False):
        st.info("Please log in from the sidebar.")
        return

    # Auto-load and extract
    load_and_update_vocabulary_data()

    tab1, tab2, tab3 = st.tabs(["Vocabulary List", "Quiz", "Data Tools"])

    with tab1:
        display_vocabulary_ui()
    with tab2:
        generate_quiz_ui()
    with tab3:
        admin_extraction_ui()

if __name__ == "__main__":
    # Initialize session state
    for key in ['vocab_data', 'is_auth', 'is_admin', 'words_displayed', 'quiz_active', 'quiz_start_index']:
        if key not in st.session_state:
            st.session_state[key] = [] if key == 'vocab_data' else (LOAD_BATCH_SIZE if key == 'words_displayed' else 0 if 'index' in key else False)

    main()
