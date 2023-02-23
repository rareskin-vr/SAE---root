import math
import re

import pyrebase
import requests
from fuzzywuzzy import fuzz

import cosine_similarity as keywordVal
import configurations
import nav_test

# TODO- Accuracy prediction library
'''
e = 1
vg = 2
g = 3
o = 4
p = 5
vp = 6
Grammar:
y = 1
n = 0
'''
def map_score_to_class(score):
    if score == 100:
        return 1
    elif score >= 80:
        return 2
    elif score >= 60:
        return 3
    elif score >= 40:
        return 4
    elif score >= 20:
        return 5
    else:
        return 6

def givVal(model_answer, keywords, answer, out_of):
    # KEYWORDS
    if len(answer.split()) <= 5:
        return 0

    k = keywordVal.givKeywordsValue(model_answer, answer)
    
    # QST
    score =fuzz.token_set_ratio(model_answer, answer)
    q = map_score_to_class(score)

    # GRAMMAR
    req = requests.get("https://api.textgears.com/check.php?text=" + answer + "&key=JmcxHCCPZ7jfXLF6")
    if 'errors' in req.json():
      no_of_errors = len(req.json()['errors'])
    else:
        no_of_errors=0
   

    if no_of_errors > 5 or k >= 5 or q >=5:
        g = 0
    else:
        g = 1

    print("Keywords : ", k)
    print("Grammar  : ", g)
    print("QST      : ", q)

    predicted = nav_test.predict(k, g, q)
    result = predicted * out_of / 10
    return result[0]


def main():
    # Initialize Firebase app
    firebase = pyrebase.initialize_app(config=configurations.config)
    db = firebase.database()

    # Get model answers from Firebase
    model_answers = db.child("model_answers").get().val()
    # Process each model answer
    for i, model_answer in enumerate(model_answers[:3]):
        model_answer_text = db.child("model_answers").get().val()[i+1]['answer']
        out_of =db.child("model_answers").get().val()[i+1]['out_of']
        keywords = re.findall(r"[a-zA-Z]+", db.child("model_answers").get().val()[i+1]['keywords'])

        print(f"\nProcessing model answer {i+1} with {out_of} marks...")
        # print(f"Model answer: {model_answer_text}")
        # print(f"Keywords: {keywords}")

        # Get all user answers from Firebase
        all_answers = db.child("answers").get()

        # Process each user answer
        for user_answer in all_answers.each():
            email = user_answer.val()['email']
            print(f"\nProcessing user answer for {email}")

            # Get answer text
            answer_text = user_answer.val()[f'a{i+1}']

            # Calculate and update result in Firebase
            result = givVal(model_answer_text, keywords, answer_text, out_of)
            print(f"Marks: {result}")
            db.child("answers").child(user_answer.key()).update({f"result{i+1}": result})


if __name__ == "__main__":
    main()