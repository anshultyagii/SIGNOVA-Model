"""
Generate comprehensive ISL Gloss-to-English translation dataset
Includes clean sequences, conversational variants, and noisy/spurious transit sequences.
"""

import json
import random
from typing import List, Tuple

NUMBERS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
NOUNS = ["food", "water", "milk"]
VERBS = ["eat", "drink", "give", "go", "help", "want"]
PRONOUNS = ["I", "we", "you", "my"]
QUESTIONS = ["what", "where"]

def generate_isl_dataset() -> List[Tuple[str, str]]:
    dataset = []

    # 1. Single sign utterances
    singles = [
        ("I", "It is me."),
        ("we", "We are here."),
        ("you", "You."),
        ("my", "This is mine."),
        ("want", "I want this."),
        ("eat", "I want to eat."),
        ("drink", "I want to drink."),
        ("give", "Please give me that."),
        ("go", "I am going."),
        ("help", "Please help me!"),
        ("food", "I need food."),
        ("water", "I need water."),
        ("milk", "I need milk."),
        ("what", "What is that?"),
        ("where", "Where is it?"),
    ]
    for n in NUMBERS:
        singles.append((n, f"{n.capitalize()}."))
    dataset.extend(singles)

    # 2. Desires & Eating / Drinking
    # I/We want eat/drink noun
    # Explicit Subjects
    for subj, subj_text in [("I", "I"), ("we", "We")]:
        dataset.append((f"{subj} want eat food", f"{subj_text} want to eat food."))
        dataset.append((f"{subj} want eat", f"{subj_text} want to eat."))
        dataset.append((f"{subj} eat food", f"{subj_text} want to eat food."))
        dataset.append((f"{subj} want food", f"{subj_text} want food."))

        # Drink
        dataset.append((f"{subj} want drink water", f"{subj_text} want to drink water."))
        dataset.append((f"{subj} want drink milk", f"{subj_text} want to drink milk."))
        dataset.append((f"{subj} want drink", f"{subj_text} want to drink something."))
        dataset.append((f"{subj} drink water", f"{subj_text} want to drink water."))
        dataset.append((f"{subj} drink milk", f"{subj_text} want to drink milk."))
        dataset.append((f"{subj} want water", f"{subj_text} want water."))
        dataset.append((f"{subj} want milk", f"{subj_text} want milk."))

        # Go
        dataset.append((f"{subj} want go", f"{subj_text} want to go."))
        dataset.append((f"{subj} go", f"{subj_text} am going." if subj == "I" else f"{subj_text} are going."))

    # Subjectless utterances (default to 1st person singular 'I')
    dataset.append(("want eat food", "I want to eat food."))
    dataset.append(("want eat", "I want to eat."))
    dataset.append(("eat food", "I want to eat food."))
    dataset.append(("food eat", "I want to eat food."))
    dataset.append(("want drink water", "I want to drink water."))
    dataset.append(("drink water", "I want to drink water."))
    dataset.append(("water drink", "I want to drink water."))
    dataset.append(("drink milk", "I want to drink milk."))
    dataset.append(("milk drink", "I want to drink milk."))
    dataset.append(("want go", "I want to go."))

    # 3. Giving & Requests
    for n in NUMBERS:
        for noun in ["water", "milk"]:
            unit = "glass of " if n == "one" else "glasses of "
            dataset.append((f"give {n} {noun}", f"Please give me {n} {unit}{noun}."))
            dataset.append((f"I want {n} {noun}", f"I want {n} {unit}{noun}."))
            dataset.append((f"give me {n} {noun}", f"Please give me {n} {unit}{noun}."))
        unit_f = "serving of food" if n == "one" else "servings of food"
        dataset.append((f"give {n} food", f"Please give me {n} {unit_f}."))
        dataset.append((f"give food", f"Please give me some food."))
        dataset.append((f"give water", f"Please give me some water."))
        dataset.append((f"give milk", f"Please give me some milk."))

    # 4. Help & Assistance
    dataset.append(("help", "Please help me!"))
    dataset.append(("help I", "Please help me!"))
    dataset.append(("help me", "Please help me!"))
    dataset.append(("you help I", "Can you please help me?"))
    dataset.append(("you help me", "Can you please help me?"))
    dataset.append(("I help you", "I can help you."))

    # 5. Questions
    dataset.append(("where go you", "Where are you going?"))
    dataset.append(("you go where", "Where are you going?"))
    dataset.append(("where go", "Where are we going?"))
    dataset.append(("where you go", "Where are you going?"))
    dataset.append(("where water", "Where is the water?"))
    dataset.append(("where food", "Where is the food?"))
    dataset.append(("where milk", "Where is the milk?"))
    dataset.append(("what you want", "What do you want?"))
    dataset.append(("you want what", "What do you want?"))
    dataset.append(("what want", "What do you want?"))
    dataset.append(("what you eat", "What are you eating?"))
    dataset.append(("what you drink", "What are you drinking?"))

    # 6. Possession
    dataset.append(("my food", "This is my food."))
    dataset.append(("my water", "This is my water."))
    dataset.append(("my milk", "This is my milk."))
    dataset.append(("my", "This is mine."))

    # 7. CRITICAL: Noisy / Glitchy Transit Datasets (Spurious transitions injected!)
    clean_examples = list(dataset)
    noisy_dataset = []
    glitch_tokens = ["what", "where", "water", "want"]

    for gloss, target in clean_examples:
        tokens = gloss.split()
        if len(tokens) >= 3 and tokens[0] in ["I", "we"] and target.startswith(("I", "We", "Please")):
            # Inject glitch token in the middle
            for glitch in ["what", "where"]:
                # e.g., ["I", "what", "want", "eat", "food"] -> "I want to eat food."
                noisy_tokens_1 = [tokens[0], glitch] + tokens[1:]
                noisy_dataset.append((" ".join(noisy_tokens_1), target))

                # e.g., ["I", "want", "what", "eat", "food"] -> "I want to eat food."
                if len(tokens) >= 4:
                    noisy_tokens_2 = tokens[:2] + [glitch] + tokens[2:]
                    noisy_dataset.append((" ".join(noisy_tokens_2), target))

    dataset.extend(noisy_dataset)
    return dataset

if __name__ == "__main__":
    data = generate_isl_dataset()
    print(f"Generated {len(data)} ISL Gloss-to-English translation samples (Clean + Noisy Transit Denoised).")
    print("Sample pairs:")
    for inp, tgt in random.sample(data, 10):
        print(f"  [{inp}]  -->  \"{tgt}\"")
