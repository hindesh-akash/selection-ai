# streamlit_app.py
import streamlit as st
import os
import tempfile
import pytesseract
import cv2
import numpy as np
from pdf2image import convert_from_path
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK resources
nltk_data_path = os.path.expanduser('~/nltk_data')
if not os.path.exists(os.path.join(nltk_data_path, 'tokenizers/punkt')):
    nltk.download('punkt')
if not os.path.exists(os.path.join(nltk_data_path, 'corpora/stopwords')):
    nltk.download('stopwords')

# Page config
st.set_page_config(
    page_title="SelectionAI - Handwritten Answer Evaluation",
    page_icon="📝",
    layout="wide"
)

# Sample answer sheet - in a real app, this would be loaded from a database or file
ANSWER_SHEET = {
    "question1": "The mitochondria is the powerhouse of the cell. It's responsible for cellular respiration and ATP production.",
    "question2": "Photosynthesis is the process by which plants convert light energy into chemical energy."
}

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using OCR"""
    # Convert PDF to images
    with st.spinner('Converting PDF to images...'):
        images = convert_from_path(pdf_path)
    
    # Extract text from each image using OCR
    extracted_text = ""
    with st.spinner('Performing OCR on images...'):
        for i, image in enumerate(images):
            # Show progress
            progress_bar.progress((i + 1) / len(images))
            
            # Convert PIL image to OpenCV format
            open_cv_image = np.array(image) 
            open_cv_image = open_cv_image[:, :, ::-1].copy() 
            
            # Preprocess the image for better OCR results
            gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            
            # Perform OCR
            text = pytesseract.image_to_string(thresh)
            extracted_text += text + "\n"
    
    return extracted_text

def evaluate_answer(extracted_text):
    """Compare the extracted text with the answer sheet"""
    # Create TF-IDF vectors for similarity comparison
    vectorizer = TfidfVectorizer()
    
    # Find the best matching question
    best_match = {"question_id": None, "similarity": 0, "reference_answer": None}
    
    for question_id, reference_answer in ANSWER_SHEET.items():
        # Create corpus with just these two documents
        corpus = [reference_answer, extracted_text]
        tfidf_matrix = vectorizer.fit_transform(corpus)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        if similarity > best_match["similarity"]:
            best_match = {
                "question_id": question_id,
                "similarity": similarity,
                "reference_answer": reference_answer
            }
    
    return best_match

def generate_suggestions(student_answer, reference_answer):
    """Generate suggestions for improving the answer"""
    # Tokenize and clean both answers
    stop_words = set(stopwords.words('english'))
    
    student_tokens = [w.lower() for w in word_tokenize(student_answer) if w.isalnum()]
    reference_tokens = [w.lower() for w in word_tokenize(reference_answer) if w.isalnum()]
    
    # Remove stop words
    student_tokens = [w for w in student_tokens if w not in stop_words]
    reference_tokens = [w for w in reference_tokens if w not in stop_words]
    
    # Find missing key terms
    key_terms = [term for term in reference_tokens if term not in student_tokens 
                 and len(term) > 3]  # Only consider terms with length > 3
    
    # Deduplicate the list of key terms
    key_terms = list(set(key_terms))
    
    suggestions = []
    
    # Length-based suggestions
    if len(student_tokens) < len(reference_tokens) * 0.7:
        suggestions.append("Your answer is too brief. Consider elaborating more on key concepts.")
    
    # Missing terms suggestions
    if key_terms:
        suggestions.append(f"Include these important concepts that were missing: {', '.join(key_terms[:5])}.")
    
    # Quality suggestion based on similarity score
    similarity = evaluate_answer(student_answer)["similarity"]
    if similarity < 0.4:
        suggestions.append("Your answer differs significantly from the reference. Review the subject material again.")
    elif similarity < 0.7:
        suggestions.append("Your answer covers some key points but needs improvement. Focus on precision and completeness.")
    else:
        suggestions.append("Good answer! To improve further, ensure you're using precise terminology.")
    
    return suggestions

# Main App UI
st.title("SelectionAI")
st.subheader("Handwritten Answer Evaluation Platform")

with st.expander("About SelectionAI", expanded=False):
    st.write("""
    SelectionAI helps you evaluate handwritten answers by:
    1. Extracting text using OCR
    2. Comparing with reference answers
    3. Providing suggestions for improvement
    
    Simply upload a PDF with your handwritten answer to get started!
    """)

# File uploader
uploaded_file = st.file_uploader("Upload a PDF with your handwritten answer", type=["pdf"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        pdf_path = tmp_file.name
    
    # Add a button to process the file
    if st.button("Process and Evaluate"):
        try:
            # Create a progress bar
            progress_bar = st.progress(0)
            
            # Process the PDF
            extracted_text = extract_text_from_pdf(pdf_path)
            
            # Evaluate the answer
            with st.spinner('Evaluating answer...'):
                evaluation = evaluate_answer(extracted_text)
            
            # Generate suggestions
            with st.spinner('Generating suggestions...'):
                suggestions = generate_suggestions(extracted_text, evaluation["reference_answer"])
            
            # Display results in columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Extracted Text")
                st.text_area("", value=extracted_text, height=200)
            
            with col2:
                st.subheader("Evaluation Results")
                st.markdown(f"**Matched Question:** {evaluation['question_id']}")
                st.markdown(f"**Similarity Score:** {round(evaluation['similarity'] * 100)}%")
                st.markdown("**Reference Answer:**")
                st.info(evaluation["reference_answer"])
            
            # Show suggestions
            st.subheader("Suggestions for Improvement")
            for suggestion in suggestions:
                st.success(suggestion)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        
        finally:
            # Clean up - delete the temporary file
            if os.path.exists(pdf_path):
                os.remove(pdf_path)

# Add a footer
st.markdown("---")
st.markdown("© 2025 SelectionAI - Developed with Streamlit")
