from sklearn.feature_extraction.text import CountVectorizer
import pickle


# Load the logistic regression model
with open('logistic_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the CountVectorizer
with open('count_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)
    
    
    
def predict_speaker(sentence):
    sentence_vec = vectorizer.transform([sentence])

    predicted_speaker = model.predict(sentence_vec)[0]

    return predicted_speaker, sentence

speaker, sentence = predict_speaker("hello i feel pain in my head since 10 days, help me doctor")
# speaker, sentence = predict_speaker("what is your problem?")
# print(p)
if speaker:
    print(f'patient: {sentence}')
else:
    print(f'Doctor: {sentence}')