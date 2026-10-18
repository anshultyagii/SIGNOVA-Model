"""
State-of-the-Art NLP Sentence Generation Engine for Indian Sign Language (ISLR)
Translates sign keyword sequences (SOV) into natural, grammatically correct English sentences.
Includes background non-blocking Text-to-Speech (TTS) voice synthesis.
"""

import threading
from typing import List

# Optional TTS
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

TOKEN_MAP = {
    "10_Ten": "ten",
    "1_One": "one",
    "2_Two": "two",
    "3_Three": "three",
    "4_Four": "four",
    "5_Five": "five",
    "6_Six": "six",
    "7_Seven": "seven",
    "8_Eight": "eight",
    "9_Nine": "nine",
    "Drink": "drink",
    "Eat": "eat",
    "Food": "food",
    "Give": "give",
    "Go": "go",
    "Help": "help",
    "I_Me": "I",
    "Milk": "milk",
    "My": "my",
    "Want_Desire_Wish_For": "want",
    "Water": "water",
    "We": "we",
    "What": "what",
    "Where": "where",
    "You": "you"
}

NUMBERS = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"}
PRONOUNS = {"I", "we", "you", "my"}
VERBS = {"eat", "drink", "give", "go", "help", "want"}
NOUNS = {"food", "water", "milk"}
QUESTIONS = {"what", "where"}

import platform
import subprocess

IS_MAC = platform.system() == "Darwin"

class SignLanguageNLP:
    def __init__(self, enable_voice=True):
        self.enable_voice = enable_voice and (TTS_AVAILABLE or IS_MAC)
        self._tts_engine = None
        self._tts_lock = threading.Lock()
        if self.enable_voice and not IS_MAC:
            try:
                # Initialize in main thread once
                self._tts_engine = pyttsx3.init()
                self._tts_engine.setProperty('rate', 155)
            except Exception as e:
                print(f"TTS init note: {e}")
                self.enable_voice = False

    def speak(self, text: str):
        """Speaks the translated sentence aloud in a background thread."""
        if not self.enable_voice or not text.strip():
            return

        # Native macOS speech synthesis (flawless, non-blocking, zero driver issues)
        if IS_MAC:
            try:
                subprocess.Popen(["say", text])
                return
            except Exception as e:
                print(f"macOS say note: {e}")

        def _worker():
            with self._tts_lock:
                try:
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 155)
                    engine.say(text)
                    engine.runAndWait()
                except Exception:
                    pass

        threading.Thread(target=_worker, daemon=True).start()

    def clean_tokens(self, sign_sequence: List[str]) -> List[str]:
        """Convert raw class labels to clean words and filter consecutive repeats."""
        cleaned = []
        last_tok = None
        for sign in sign_sequence:
            word = TOKEN_MAP.get(sign, sign.lower())
            if word != last_tok:
                cleaned.append(word)
                last_tok = word
        return cleaned

    def generate_sentence(self, sign_sequence: List[str]) -> str:
        """
        Translates raw ISL keyword sequences into natural, fluent English sentences.
        Applies grammar restoration: subject-verb agreement, auxiliary insertion,
        and idiomatic request transformations.
        """
        tokens = self.clean_tokens(sign_sequence)
        if not tokens:
            return ""

        # Check special question patterns
        if "where" in tokens:
            if "go" in tokens:
                target = "you" if "you" in tokens else "we"
                return f"Where are {target} going?"
            elif any(n in tokens for n in NOUNS):
                noun = next(n for n in tokens if n in NOUNS)
                return f"Where is the {noun}?"
            else:
                return "Where is it?"

        if "what" in tokens:
            if "want" in tokens:
                return "What do you want?"
            elif "eat" in tokens:
                return "What are you eating?"
            elif "drink" in tokens:
                return "What are you drinking?"
            else:
                return "What is that?"

        # Polite requests
        if tokens[0] == "help" or (len(tokens) > 1 and tokens[1] == "help"):
            if "you" in tokens:
                return "Can you please help me?"
            return "Please help me!"

        if "give" in tokens:
            # Check for quantity + noun (e.g., "give two water")
            num = next((t for t in tokens if t in NUMBERS), None)
            noun = next((t for t in tokens if t in NOUNS), None)
            if num and noun:
                if num == "one":
                    unit = "glass of " if noun in ["water", "milk"] else "serving of "
                else:
                    unit = "glasses of " if noun in ["water", "milk"] else "servings of "
                return f"Please give me {num} {unit}{noun}."
            elif noun:
                return f"Please give me some {noun}."
            elif num:
                return f"Please give me {num}."
            return "Please give me that."

        # Desires & Needs
        if "want" in tokens or "eat" in tokens or "drink" in tokens:
            subj = "We" if "we" in tokens else "I"
            noun = next((t for t in tokens if t in NOUNS), None)
            num = next((t for t in tokens if t in NUMBERS), None)

            if "eat" in tokens:
                food_text = f" some {noun}" if noun else " something"
                return f"{subj} want to eat{food_text}."
            elif "drink" in tokens:
                drink_text = f" {noun}" if noun else " water"
                return f"{subj} want to drink some{drink_text}."
            elif "want" in tokens:
                if num and noun:
                    return f"{subj} want {num} {noun}."
                elif noun:
                    return f"{subj} want some {noun}."
                return f"{subj} want this."

        # Possession
        if "my" in tokens:
            noun = next((t for t in tokens if t in NOUNS), None)
            if noun:
                return f"This is my {noun}."

        # Movement
        if "go" in tokens:
            if "we" in tokens:
                return "We should go now."
            elif "you" in tokens:
                return "You can go."
            return "I am going."

        # Single token conversational expansions
        if len(tokens) == 1:
            tok = tokens[0]
            if tok in NUMBERS:
                return f"{tok.capitalize()}."
            elif tok == "I":
                return "It is me."
            elif tok == "we":
                return "We are here."
            elif tok == "you":
                return "You."
            elif tok == "my":
                return "This is mine."
            elif tok == "what":
                return "What is that?"
            elif tok == "where":
                return "Where is it?"
            elif tok == "water":
                return "I need water."
            elif tok == "food":
                return "I need food."
            elif tok == "milk":
                return "I need milk."
            elif tok == "eat":
                return "I want to eat."
            elif tok == "drink":
                return "I want to drink."
            elif tok == "help":
                return "Please help me!"
            elif tok == "want":
                return "I want this."
            elif tok == "go":
                return "I am going."
            elif tok == "give":
                return "Please give me that."

        # General Syntactic Reconstruction (Subject + Verb + Object)
        subj = next((t for t in tokens if t in PRONOUNS), "I")
        verb = next((t for t in tokens if t in VERBS), None)
        obj = next((t for t in tokens if t in NOUNS), None)

        if subj and verb and obj:
            v_form = f"{verb}s" if subj not in ["I", "we", "you"] else verb
            return f"{subj.capitalize()} {v_form} the {obj}."
        elif subj and verb:
            return f"{subj.capitalize()} {verb}."

        # Fallback: Capitalized sentence
        raw = " ".join(tokens)
        return raw.capitalize() + "."

if __name__ == "__main__":
    nlp = SignLanguageNLP(enable_voice=False)
    tests = [
        ["Where", "Go", "You"],
        ["What", "You", "Want"],
        ["Give", "2_Two", "Water"],
        ["Help", "I_Me"],
        ["I_Me", "Eat", "Food"],
        ["We", "Go"],
        ["My", "Milk"],
        ["5_Five"]
    ]
    print("Testing Advanced NLP Sentence Engine:")
    print("-" * 50)
    for t in tests:
        print(f"Signs: {t} -> \"{nlp.generate_sentence(t)}\"")
