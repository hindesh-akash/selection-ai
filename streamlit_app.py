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
from nltk.corpus import stopwords# streamlit_app.py
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

# Check if NLTK data exists before downloading
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

# UPSC Questions and Answers
upsc_qa = {
    1: {
        "question": "Discuss the significance of the 73rd Constitutional Amendment Act in strengthening grassroots democracy in India.",
        "answer": "The 73rd Constitutional Amendment Act of 1992 institutionalized Panchayati Raj in India, strengthening grassroots democracy. It granted constitutional status to rural local bodies and mandated the creation of Panchayati Raj Institutions (PRIs) at village, intermediate, and district levels. Regular elections every five years, financial devolution, and reservation of seats for Scheduled Castes, Scheduled Tribes, and women were introduced to ensure inclusivity and empowerment. By facilitating citizen participation in governance, PRIs have enhanced transparency, accountability, and development at the local level. They have also fostered leadership among marginalized sections and women, contributing to social justice. However, challenges like limited financial autonomy, political interference, and bureaucratic dominance still persist, demanding further reforms to realize the full potential of grassroots democracy."
    },
    2: {
        "question": "Analyze the impact of globalization on India's cultural diversity.",
        "answer": "Globalization has significantly influenced India's cultural landscape. Increased exposure to global ideas has enriched Indian traditions, cuisine, music, and cinema. Fusion forms in art, fashion, and lifestyle symbolize cultural amalgamation. Indian culture, through Bollywood, yoga, and festivals, has gained international acceptance. However, globalization has also triggered fears of cultural homogenization, where Western influences sometimes overshadow indigenous languages, crafts, and traditional values. There is a growing trend among urban youth to adopt Western lifestyles, sometimes at the cost of their cultural roots. Yet, globalization has also revived interest in India's heritage among diasporic communities and bolstered the economy through cultural tourism. Hence, globalization's impact on India's cultural diversity is complex — both enriching and challenging."
    },
    3: {
        "question": "Evaluate the role of civil services in a democracy.",
        "answer": "Civil services form the backbone of administrative machinery in a democracy like India. They implement government policies, maintain law and order, and deliver public services. As permanent executives, they provide continuity across political changes. Civil servants are expected to be neutral, efficient, and accountable, ensuring that democratic principles are upheld in administration. Their role in rural development, disaster management, health, education, and social justice is crucial. However, issues like bureaucratic inertia, corruption, and lack of accountability have sometimes undermined their credibility. Reforms like lateral entry, performance appraisal, and capacity building are being introduced to enhance their efficiency. A responsive, ethical, and citizen-centric civil service is essential to strengthen democracy."
    },
    4: {
        "question": "Critically examine the concept of 'One Nation, One Election'.",
        "answer": "The idea of 'One Nation, One Election' proposes simultaneous elections to the Lok Sabha and State Legislative Assemblies. Advocates argue it would reduce election costs, administrative burden, and political disruptions caused by frequent polls. It could lead to better policy continuity and governance focus. However, critics highlight constitutional and practical challenges, such as premature dissolution of assemblies, undermining federalism, and logistical complexities. Implementing this idea would require significant constitutional amendments and political consensus. It is essential to balance electoral efficiency with democratic vibrancy to ensure that such a reform strengthens, not weakens, India's democracy."
    },
    5: {
        "question": "Discuss the challenges and opportunities of India's demographic dividend.",
        "answer": "India's demographic dividend — a large working-age population — offers immense potential for economic growth. If harnessed effectively, it can lead to higher productivity, innovation, and consumption. However, challenges like unemployment, skill mismatch, poor education and healthcare infrastructure, and regional disparities could turn the dividend into a demographic disaster. To realize this opportunity, India must invest heavily in education, skill development, health, and job creation, particularly in sunrise sectors like technology, renewable energy, and services. Empowering women, promoting entrepreneurship, and encouraging labor-intensive industries are crucial. Timely policy interventions are key to converting this demographic potential into a demographic advantage."
    },
    6: {
        "question": "Examine the relevance of Gandhian philosophy in contemporary India.",
        "answer": "Gandhian philosophy — centered on non-violence, truth, self-reliance, and rural development — remains profoundly relevant today. In an era marked by violence, polarization, and environmental degradation, Gandhi's emphasis on non-violent conflict resolution, communal harmony, and sustainable living offers timeless lessons. His vision of Gram Swaraj aligns with modern ideas of decentralized governance. Concepts like Swadeshi promote self-sufficiency and local economies, vital for rural development. However, the commercialization of society and rapid technological changes pose challenges to fully implementing Gandhian ideals. Nevertheless, selective adoption of his principles can guide India's socio-economic and moral rejuvenation."
    },
    7: {
        "question": "Analyze the implications of climate change for India's food security.",
        "answer": "Climate change poses a serious threat to India's food security. Rising temperatures, erratic rainfall, frequent droughts, and floods adversely affect agricultural productivity. Staple crops like rice and wheat are vulnerable, leading to potential yield declines. Changing pest patterns and soil degradation further exacerbate the problem. Small and marginal farmers, who form the backbone of Indian agriculture, are particularly at risk. Adaptive strategies like crop diversification, climate-resilient seeds, improved irrigation, and better early warning systems are necessary. Strengthening agricultural research, extension services, and climate-smart policies are critical to ensure food security for India's growing population."
    },
    8: {
        "question": "Discuss the ethical challenges in artificial intelligence applications.",
        "answer": "Artificial Intelligence (AI) raises profound ethical challenges. Issues like bias in algorithms, lack of transparency (black box problem), and discrimination in automated decision-making systems highlight the risks. Data privacy, surveillance, and autonomy are also major concerns. In fields like healthcare, finance, and criminal justice, AI decisions can have life-altering consequences. There are fears of job losses, deepfakes, and weaponization of AI. Ethical frameworks that ensure fairness, accountability, transparency, and human oversight are essential. Multistakeholder involvement — governments, corporations, civil society — is necessary to develop ethical AI governance globally."
    },
    9: {
        "question": "Evaluate the importance of federalism in India's polity.",
        "answer": "Federalism is vital for India's vast and diverse society. It ensures power-sharing between the Centre and the States, balancing unity with regional autonomy. It accommodates linguistic, cultural, and ethnic diversity, thus strengthening national integration. Cooperative federalism promotes collaboration in policymaking, essential in areas like health, education, and disaster management. However, issues like financial centralization, misuse of central agencies, and political friction challenge federal principles. Strengthening institutions like the Inter-State Council and promoting fiscal federalism are necessary to rejuvenate India's federal spirit."
    },
    10: {
        "question": "Analyze the role of technology in enhancing governance in India.",
        "answer": "Technology has revolutionized governance in India. Initiatives like Digital India, Aadhaar, e-Governance services, and online grievance redressal have made governance more accessible, transparent, and efficient. Digital platforms for education, healthcare, and welfare delivery have expanded citizen engagement. Technologies like blockchain, AI, and GIS are being integrated into policymaking. However, digital divides, cybersecurity threats, and concerns about privacy persist. Inclusive digital literacy, robust data protection laws, and citizen-centric technology design are vital to maximize technology's benefits for governance."
    },
    11: {
        "question": "Critically examine the effectiveness of the Right to Information (RTI) Act.",
        "answer": "The RTI Act, 2005, empowered citizens to seek information from public authorities, promoting transparency and accountability. It has exposed corruption, ensured better service delivery, and strengthened democracy. However, challenges like bureaucratic resistance, lack of proactive disclosure, pendency of cases in Information Commissions, and recent amendments weakening its autonomy have affected its effectiveness. Strengthening Information Commissions, protecting whistleblowers, and creating a culture of openness within government departments are essential to realize the true potential of the RTI Act."
    },
    12: {
        "question": "Discuss the challenges faced by the Indian judiciary.",
        "answer": "The Indian judiciary plays a crucial role in upholding constitutional rights and the rule of law. However, it faces significant challenges such as case pendency, shortage of judges, procedural delays, and access to justice for marginalized groups. Judicial overreach and lack of transparency in appointments through the collegium system have also raised concerns. Reforms like digitization, faster appointments, alternative dispute resolution mechanisms, and a robust judicial accountability framework are needed to enhance efficiency and public trust."
    },
    13: {
        "question": "Analyze the role of women in India's freedom struggle.",
        "answer": "Women played a vital and inspiring role in India's freedom movement. Leaders like Rani Lakshmi Bai, Sarojini Naidu, Annie Besant, Aruna Asaf Ali, and Kasturba Gandhi led protests, organized movements, and inspired masses. Women participated in picketing, boycotts, satyagraha, and armed resistance. The freedom movement also catalyzed women's social empowerment, challenging traditional gender roles. Although often marginalized in historical narratives, their contributions laid the foundation for post-independence constitutional guarantees of gender equality. Recognizing their sacrifices is crucial for an inclusive understanding of India's independence struggle."
    },
    14: {
        "question": "Examine the importance of ethical leadership in public administration.",
        "answer": "Ethical leadership in public administration ensures fairness, integrity, transparency, and accountability in governance. It builds public trust, enhances service delivery, and creates a culture of ethical behavior among subordinates. In India, where corruption remains a concern, ethical leadership can combat systemic inefficiencies. Ethical leaders set personal examples, making institutions resilient and citizen-centric. However, pressures like political interference, vested interests, and institutional inertia often challenge ethical conduct. Training, incentives, and a strong legal framework are necessary to nurture ethical leadership."
    },
    15: {
        "question": "Critically analyze the challenges of urbanization in India.",
        "answer": "Urbanization in India has led to economic growth, innovation, and improved living standards. However, it has also created challenges like slum proliferation, traffic congestion, pollution, inadequate infrastructure, and social inequalities. Cities face stress on water, waste management, and housing. Unplanned urban expansion worsens disaster vulnerability and environmental degradation. Policies like Smart Cities Mission, AMRUT, and urban renewal programs aim to address these issues. Sustainable urban planning, affordable housing, inclusive growth, and resilient infrastructure are vital to make Indian cities livable and equitable."
    }
}

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using OCR"""
    # Convert PDF to images
    with st.spinner('Converting PDF to images...'):
        images = convert_from_path(pdf_path)
    
    # Extract text from each image using OCR
    extracted_text = ""
    with st.spinner('Performing OCR on images...'):
        progress_bar = st.progress(0)
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

def evaluate_answer(extracted_text, reference_answer):
    """Compare the extracted text with the reference answer"""
    # Create TF-IDF vectors for similarity comparison
    vectorizer = TfidfVectorizer()
    
    # Create corpus with just these two documents
    corpus = [reference_answer, extracted_text]
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    # Calculate cosine similarity
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    return {
        "similarity": similarity,
        "reference_answer": reference_answer
    }

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
    
    # Get unique terms (deduplicate)
    key_terms = list(set(key_terms))
    
    # Sort by possible importance (length can be a rough proxy for specialized terms)
    key_terms.sort(key=len, reverse=True)
    
    suggestions = []
    
    # Length-based suggestions
    if len(student_tokens) < len(reference_tokens) * 0.7:
        suggestions.append("Your answer is too brief. Consider elaborating more on key concepts.")
    
    # Missing terms suggestions
    if key_terms:
        suggestions.append(f"Consider including these important concepts that were missing: {', '.join(key_terms[:5])}.")
    
    # Structure and content suggestions
    if len(student_tokens) > 20:  # Only if we have a substantive answer
        # Check if the answer covers multiple aspects (simple check based on paragraph breaks)
        if student_answer.count('\n\n') < 2:
            suggestions.append("Try structuring your answer with clear introduction, body, and conclusion paragraphs.")
        
        # Check for examples
        example_indicators = ['example', 'instance', 'case', 'illustration', 'such as', 'for instance']
        has_examples = any(indicator in student_answer.lower() for indicator in example_indicators)
        if not has_examples:
            suggestions.append("Include specific examples to strengthen your arguments.")
    
    # Quality suggestion based on similarity score
    similarity = evaluate_answer(student_answer, reference_answer)["similarity"]
    if similarity < 0.3:
        suggestions.append("Your answer differs significantly from the reference. Review the subject material again.")
    elif similarity < 0.5:
        suggestions.append("Your answer covers some key points but needs improvement in accuracy and completeness.")
    elif similarity < 0.7:
        suggestions.append("Good attempt! To improve further, ensure you're addressing all aspects of the question.")
    else:
        suggestions.append("Excellent answer! For perfection, focus on precision of terminology and conciseness.")
    
    return suggestions

def extract_key_concepts(text, n=8):
    """Extract key concepts from text for focused feedback"""
    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [w.lower() for w in word_tokenize(text) if w.isalpha() and w.lower() not in stop_words and len(w) > 3]
    
    # Count word frequencies
    from collections import Counter
    word_counts = Counter(words)
    
    # Get most common words as key concepts
    return [word for word, _ in word_counts.most_common(n)]

# Main App UI
st.title("SelectionAI")
st.subheader("UPSC Answer Evaluation Platform")

with st.expander("About SelectionAI", expanded=False):
    st.write("""
    SelectionAI helps you evaluate your UPSC exam answers by:
    1. Allowing you to select a question to answer
    2. Extracting text from your handwritten answer using OCR
    3. Comparing with reference answers
    4. Providing suggestions for improvement
    
    Simply select a question, upload a PDF with your handwritten answer, or type your answer directly!
    """)

# Two tabs: Choose question, and Upload answer
tab1, tab2 = st.tabs(["Select Question", "Submit & Evaluate"])

# Store question selection in session state
if 'selected_question_id' not in st.session_state:
    st.session_state.selected_question_id = 1
if 'answer_text' not in st.session_state:
    st.session_state.answer_text = ""
if 'evaluation_done' not in st.session_state:
    st.session_state.evaluation_done = False
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = ""
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = None
if 'suggestions' not in st.session_state:
    st.session_state.suggestions = []

# First tab: Question selection
with tab1:
    st.header("Select a UPSC Question")
    
    # Create a dropdown with all questions
    question_options = {
        f"Question {k}: {v['question'][:70]}..." if len(v['question']) > 70 else f"Question {k}: {v['question']}": k 
        for k, v in upsc_qa.items()
    }

    selected_question = st.selectbox(
        "Choose a question to answer:",
        options=list(question_options.keys()),
        index=st.session_state.selected_question_id - 1
    )
    
    # Extract the question number
    question_id = int(selected_question.split(":")[0].replace("Question ", ""))
    st.session_state.selected_question_id = question_id
    
    # Display the full question
    st.subheader("Question:")
    st.write(upsc_qa[question_id]["question"])
    
    st.info("After selecting your question, go to the 'Submit & Evaluate' tab to provide your answer.")

# Second tab: Answer submission and evaluation
with tab2:
    st.header(f"Answer for Question {st.session_state.selected_question_id}")
    st.write(upsc_qa[st.session_state.selected_question_id]["question"])
    
    st.subheader("How would you like to submit your answer?")
    
    submission_method = st.radio(
        "Choose submission method:",
        ["Type your answer", "Upload handwritten PDF"]
    )
    
    if submission_method == "Type your answer":
        # Text input for typed answers
        st.session_state.answer_text = st.text_area(
            "Type your answer here:",
            value=st.session_state.answer_text,
            height=300
        )
        
        if st.button("Evaluate Typed Answer"):
            if st.session_state.answer_text.strip() == "":
                st.error("Please type an answer before evaluation.")
            else:
                # Evaluate the typed answer
                reference_answer = upsc_qa[st.session_state.selected_question_id]["answer"]
                evaluation = evaluate_answer(st.session_state.answer_text, reference_answer)
                suggestions = generate_suggestions(st.session_state.answer_text, reference_answer)
                
                # Store results in session state
                st.session_state.extracted_text = st.session_state.answer_text
                st.session_state.evaluation_results = evaluation
                st.session_state.suggestions = suggestions
                st.session_state.evaluation_done = True
                
                # Force rerun to show results
                st.experimental_rerun()
    
    else:  # Upload handwritten PDF
        # File uploader
        uploaded_file = st.file_uploader("Upload a PDF with your handwritten answer", type=["pdf"])
        
        if uploaded_file is not None:
            # Display a thumbnail of the uploaded PDF
            st.write("PDF Preview (first page):")
            
            # Save the uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                pdf_path = tmp_file.name
            
            # Add a button to process the file
            if st.button("Process and Evaluate PDF"):
                try:
                    # Create a progress bar
                    progress_bar = st.progress(0)
                    
                    # Process the PDF
                    extracted_text = extract_text_from_pdf(pdf_path)
                    
                    # Evaluate the answer
                    with st.spinner('Evaluating answer...'):
                        reference_answer = upsc_qa[st.session_state.selected_question_id]["answer"]
                        evaluation = evaluate_answer(extracted_text, reference_answer)
                    
                    # Generate suggestions
                    with st.spinner('Generating suggestions...'):
                        suggestions = generate_suggestions(extracted_text, reference_answer)
                    
                    # Store results in session state
                    st.session_state.extracted_text = extracted_text
                    st.session_state.evaluation_results = evaluation
                    st.session_state.suggestions = suggestions
                    st.session_state.evaluation_done = True
                    
                    # Force rerun to show results
                    st.experimental_rerun()
                        
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                
                finally:
                    # Clean up - delete the temporary file
                    if os.path.exists(pdf_path):
                        os.remove(pdf_path)
    
    # Display evaluation results if available
    if st.session_state.evaluation_done:
        st.header("Evaluation Results")
        
        # Create tabs for Results and Suggestions
        results_tab, suggestions_tab, reference_tab = st.tabs(["Evaluation", "Improvement Suggestions", "Reference Answer"])
        
        with results_tab:
            st.subheader("Your Answer")
            st.text_area("Extracted/Submitted Text:", value=st.session_state.extracted_text, height=200)
            
            st.subheader("Score")
            similarity_score = st.session_state.evaluation_results["similarity"] * 100
            
            # Display score with color coding
            if similarity_score >= 70:
                st.success(f"Similarity Score: {similarity_score:.1f}%")
            elif similarity_score >= 50:
                st.warning(f"Similarity Score: {similarity_score:.1f}%")
            else:
                st.error(f"Similarity Score: {similarity_score:.1f}%")
            
            # Visualize score
            st.progress(min(float(similarity_score/100), 1.0))
            
            # Performance interpretation
            if similarity_score >= 80:
                st.success("Excellent! Your answer aligns very well with the expected response.")
            elif similarity_score >= 70:
                st.success("Very good! Your answer covers most key points effectively.")
            elif similarity_score >= 60:
                st.info("Good attempt! Your answer addresses the question but could be improved.")
            elif similarity_score >= 50:
                st.warning("Fair attempt. Your answer needs more development and precision.")
            else:
                st.error("Your answer needs significant improvement in content and approach.")
            
            # Key concepts covered/missed
            reference_answer = st.session_state.evaluation_results["reference_answer"]
            ref_key_concepts = extract_key_concepts(reference_answer)
            student_key_concepts = extract_key_concepts(st.session_state.extracted_text)
            
            # Find overlap
            common_concepts = set(student_key_concepts).intersection(set(ref_key_concepts))
            missed_concepts = set(ref_key_concepts) - set(student_key_concepts)
            
            st.subheader("Key Concepts Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("Concepts You Covered Well:")
                for concept in common_concepts:
                    st.success(f"✓ {concept.capitalize()}")
            
            with col2:
                st.write("Important Concepts to Include:")
                for concept in missed_concepts:
                    st.error(f"✗ {concept.capitalize()}")
        
        with suggestions_tab:
            st.subheader("Suggestions for Improvement")
            for i, suggestion in enumerate(st.session_state.suggestions, 1):
                st.info(f"{i}. {suggestion}")
            
            st.subheader("General Writing Tips")
            st.write("""
            1. **Structure**: Use a clear introduction, body and conclusion
            2. **Precision**: Use specific facts, figures, and examples
            3. **Balance**: Present multiple perspectives for critical questions
            4. **Relevance**: Stay focused on the exact question asked
            5. **Conciseness**: Aim for quality over quantity in your response
            """)
        
        with reference_tab:
            st.subheader("Reference Answer")
            st.write("""
            Note: This is provided as a learning resource. There is no single 'perfect' answer, and examinations reward 
            original thinking that demonstrates understanding of core concepts.
            """)
            st.info(upsc_qa[st.session_state.selected_question_id]["answer"])

        # Reset button
        if st.button("Answer Another Question"):
            st.session_state.evaluation_done = False
            st.session_state.answer_text = ""
            st.experimental_rerun()

# Add a footer
st.markdown("---")
st.markdown("© 2025 SelectionAI - Developed for UPSC Answer Evaluation")
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
