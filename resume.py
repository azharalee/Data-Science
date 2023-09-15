import streamlit as st
import pickle
import re
import nltk
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier

nltk.download('punkt')
nltk.download('stopwords')

# Load the CSV file
df = pd.read_csv('UpdatedResumeDataSet.csv')

# Label Encoding
var_mod = ['Category']
le = LabelEncoder()
for i in var_mod:
    df[i] = le.fit_transform(df[i])

# Vectorization
requiredText = df['Resume'].values
requiredTarget = df['Category'].values

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    stop_words='english',
    max_features=1500)
word_vectorizer.fit(requiredText)
WordFeatures = word_vectorizer.transform(requiredText)

# Training model
X_train, X_test, y_train, y_test = train_test_split(WordFeatures, requiredTarget, random_state=0, test_size=0.2)
clf = OneVsRestClassifier(KNeighborsClassifier())
clf.fit(X_train, y_train)

# Pickle the model and vectorizer
with open('clf.pkl', 'wb') as file:
    pickle.dump(clf, file)

with open('word_vectorizer.pkl', 'wb') as file1:
    pickle.dump(word_vectorizer, file1)

# Load the model and vectorizer
clf = pickle.load(open('clf.pkl', 'rb'))
word_vectorizer = pickle.load(open('word_vectorizer.pkl', 'rb'))


# Function to clean resume text
def clean_resume(resume_text):
    clean_text = re.sub('http\S+\s*', ' ', resume_text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+', '', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text


# Function to predict resume category
def predict_category(resume_text):
    cleaned_resume = clean_resume(resume_text)
    input_features = word_vectorizer.transform([cleaned_resume])
    prediction_id = clf.predict(input_features)[0]
    return prediction_id


# Category mapping
category_mapping = {
    15: "Java Developer",
    23: "Testing",
    8: "DevOps Engineer",
    20: "Python Developer",
    24: "Web Designing",
    12: "HR",
    13: "Hadoop",
    3: "Blockchain",
    10: "ETL Developer",
    18: "Operations Manager",
    6: "Data Science",
    22: "Sales",
    16: "Mechanical Engineer",
    1: "Arts",
    7: "Database",
    11: "Electrical Engineering",
    14: "Health and fitness",
    19: "PMO",
    4: "Business Analyst",
    9: "DotNet Developer",
    2: "Automation Testing",
    17: "Network Security Engineer",
    21: "SAP Developer",
    5: "Civil Engineer",
    0: "Advocate",
}

# Streamlit web app
def main():
    st.title(":blue[Resume] :blue[Screening] Web App")
    st.image("Resume-screening.jpg")
    uploaded_file = st.file_uploader('Upload Resume', type=['TXT', 'PDF', 'JPEG'])
    if uploaded_file is not None:
        try:
            file_extension = uploaded_file.name.split('.')[-1]

            if file_extension == 'pdf':
                # Use pdfplumber to extract text from PDF
                import pdfplumber
                with pdfplumber.open(uploaded_file) as pdf:
                    pdf_text = ""
                    for page in pdf.pages:
                        pdf_text += page.extract_text()
                prediction_id = predict_category(pdf_text)
            else:
                resume_bytes = uploaded_file.read()
                resume_text = resume_bytes.decode('utf-8')
                prediction_id = predict_category(resume_text)

            st.write("Predicted Category:", category_mapping.get(prediction_id, "Unknown"))
            st.write(prediction_id)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
