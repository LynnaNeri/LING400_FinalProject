from joblib import load

vectorize, svm = load('vectorizer_svm.joblib')
test_content = 'I think we should just let it happen. This world is a complete shithouse with no redeeming qualities and things are CONSTANTLY and ACTIVELY getting worse and worse by the day. What\'s the point in staying alive if the climates gonna kill me in 30 years?? If the US decides that I\'m no longer a human being and can be legally hunted for sport?? Keeping someone alive who already knows how terrible everything is is just cruelty.'
test_vector = vectorize.transform(test_content)
print("SVM Prediction:", svm.predict_proba)