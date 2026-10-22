import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.sign_nlp_sentence_generator import SignLanguageNLP

nlp = SignLanguageNLP(enable_voice=False)
tests = [
    ['I_Me', 'Want_Desire_Wish_For', 'Eat', 'Food'],
    ['I_Me', 'What', 'Want_Desire_Wish_For', 'Eat', 'Food'],
    ['I_Me', 'Want_Desire_Wish_For', 'What', 'Eat', 'Food'],
    ['Where', 'Go', 'You'],
    ['What', 'You', 'Want'],
    ['Give', '2_Two', 'Water'],
    ['Give', 'Food'],
    ['Help', 'I_Me'],
    ['You', 'Help', 'I_Me'],
    ['We', 'Go'],
    ['My', 'Milk'],
    ['Eat', 'Food'],
    ['Drink', 'Water']
]
print("\n" + "="*70)
print("             NLP BATTERY VERIFICATION RESULTS")
print("="*70)
for t in tests:
    res = nlp.generate_sentence(t)
    print(f"{str(t):55s} -> \"{res}\"")
print("="*70 + "\n")
