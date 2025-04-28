# streamlit_app_mvp.py
import streamlit as st
import os
import tempfile
from pdf2image import convert_from_path
import torch

# --- Hugging Face Imports (Basic) ---
try:
    from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
    from sentence_transformers import SentenceTransformer, util
    hf_import_success = True
except ImportError:
    st.error("""
        Failed to import necessary libraries. Please install them:
        `pip install streamlit pdf2image transformers sentence-transformers torch`
        You might also need system dependencies like poppler (for PDF conversion).
    """)
    hf_import_success = False

# --- Page Config ---
st.set_page_config(
    page_title="SelectionAI - MVP Evaluation",
    page_icon="🤖",
    layout="wide"
)

st.title("SelectionAI - MVP Evaluation")
st.caption("Simplified version focusing on core functionality.")

# --- Constants & Models ---
# Choose smaller models for faster loading & lower resource usage in MVP
OCR_MODEL = "microsoft/trocr-base-handwritten" # Or "microsoft/trocr-base-printed"
SIMILARITY_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
FEEDBACK_MODEL = 'google/flan-t5-small' # Smaller T5 model for faster feedback (less capable)

# Determine device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Cache models
@st.cache_resource(show_spinner="Loading Models...")
def load_models():
    """Loads all necessary models."""
    if not hf_import_success:
        return None, None, None, None
    try:
        ocr_pipe = pipeline("image-to-text", model=OCR_MODEL, device=0 if DEVICE == "cuda" else -1)
        similarity_model = SentenceTransformer(SIMILARITY_MODEL, device=DEVICE)
        feedback_tokenizer = AutoTokenizer.from_pretrained(FEEDBACK_MODEL)
        feedback_model = AutoModelForSeq2SeqLM.from_pretrained(FEEDBACK_MODEL).to(DEVICE)
        st.success("All models loaded successfully.")
        return ocr_pipe, similarity_model, feedback_model, feedback_tokenizer
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.warning("Check your internet connection and library installations.")
        return None, None, None, None

# Load models upfront
ocr_pipe, similarity_model, feedback_model, feedback_tokenizer = load_models()

# --- Check if models are available ---
models_available = all([ocr_pipe, similarity_model, feedback_model, feedback_tokenizer])
if not models_available:
    st.stop() # Stop if models failed to load

# --- UPSC Questions and Answers (Simplified, can be moved to a config file) ---
upsc_qa = {
    1: {
        "question": "Discuss the significance of the 73rd Constitutional Amendment Act in strengthening grassroots democracy in India.",
        "answer": "The 73rd Constitutional Amendment Act of 1992 institutionalized Panchayati Raj in India, strengthening grassroots democracy. It granted constitutional status to rural local bodies and mandated the creation of Panchayati Raj Institutions (PRIs) at village, intermediate, and district levels. Regular elections every five years, financial devolution, and reservation of seats for Scheduled Castes, Scheduled Tribes, and women were introduced to ensure inclusivity and empowerment. By facilitating citizen participation in governance, PRIs have enhanced transparency, accountability, and development at the local level. They have also fostered leadership among marginalized sections and women, contributing to social justice. However, challenges like limited financial autonomy, political interference, and bureaucratic dominance still persist, demanding further reforms to realize the full potential of grassroots democracy."
    },
    2: {
        "question": "Analyze the impact of globalization on India's cultural diversity.",
        "answer": "Globalization has significantly influenced India's cultural landscape. Increased exposure to global ideas has enriched Indian traditions, cuisine, music, and cinema. Fusion forms in art, fashion, and lifestyle symbolize cultural amalgamation. Indian culture, through Bollywood, yoga, and festivals, has gained international acceptance. However, globalization has also triggered fears of cultural homogenization, where Western influences sometimes overshadow indigenous languages, crafts, and traditional values. There is a growing trend among urban youth to adopt Western lifestyles, sometimes at the cost of their cultural roots. Yet, globalization has also revived interest in India's heritage among diasporic communities and bolstered the economy through cultural tourism. Hence, globalization's impact on India's cultural diversity is complex — both enriching and challenging."
    }
}

# --- Core Functions (Simplified) ---

def extract_text_from_pdf(pdf_path, _ocr_pipeline):
    """Extract text from PDF using Hugging Face OCR pipeline (simplified)."""
    try:
        images = convert_from_path(pdf_path)
        extracted_texts = []
        with st.spinner("Performing OCR..."):
             for img in images:
                 # OCR pipeline expects PIL Image or numpy array
                 text = _ocr_pipeline(img)[0]['generated_text']
                 extracted_texts.append(text)
        return "\n\n".join(extracted_texts)
    except Exception as e:
        st.error(f"Error during PDF processing or OCR: {e}")
        return ""

def evaluate_answer(extracted_text, reference_answer, _similarity_model):
    """Compare texts using Sentence Transformers (simplified)."""
    if not extracted_text or not reference_answer:
        return 0.0 # Return 0 if texts are empty
    try:
        # Embeddings are calculated on the model's device (GPU if available)
        embedding1 = _similarity_model.encode(reference_answer, convert_to_tensor=True)
        embedding2 = _similarity_model.encode(extracted_text, convert_to_tensor=True)
        # Cosine similarity calculation handles device
        similarity_score = util.pytorch_cos_sim(embedding1, embedding2).item()
        return max(0.0, min(1.0, similarity_score)) # Ensure score is between 0 and 1
    except Exception as e:
        st.error(f"Error calculating similarity: {e}")
        return 0.0

def generate_suggestions(question, reference_answer, student_answer, _feedback_model, _feedback_tokenizer):
    """Generate simple suggestions using a generative model (simplified)."""
    if not student_answer:
        return ["No answer provided to generate feedback."]
    try:
        # Simplified prompt
        prompt = f"""Question: {question}
Reference Answer (Key Points): {reference_answer}
Student Answer: {student_answer}

Provide 3 brief suggestions for improvement based on comparing the student answer to the reference answer.
Suggestions:"""

        # Model should already be on DEVICE
        inputs = _feedback_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(DEVICE)

        # Simplified generation parameters
        outputs = _feedback_model.generate(
            inputs.input_ids,
            max_length=150,
            num_beams=1, # Simple greedy search
            temperature=1.0 # Default temperature
        )

        suggestions_text = _feedback_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Basic splitting by newline or period
        suggestions_list = [s.strip() for s in suggestions_text.split('\n') if s.strip()]
        if not suggestions_list:
             suggestions_list = [s.strip() for s in suggestions_text.split('.') if s.strip()]

        return suggestions_list if suggestions_list else ["Could not generate specific suggestions."]

    except Exception as e:
        st.error(f"Error generating feedback: {e}")
        return ["An error occurred while generating feedback."]


# --- Streamlit UI ---

# Question Selection
st.subheader("1. Select a UPSC Question")
question_options = {f"Q{k}: {v['question'][:80]}...": k for k, v in upsc_qa.items()}
selected_question_preview = st.selectbox(
    "Choose a question:",
    options=list(question_options.keys()),
    key="question_selector"
)
selected_qid = question_options[selected_question_preview]
st.markdown(f"**Full Question:** {upsc_qa[selected_qid]['question']}")

st.divider()

# Answer Submission
st.subheader("2. Submit Your Answer")

submission_method = st.radio(
    "How will you submit your answer?",
    ["Type directly", "Upload handwritten PDF"],
    key="submission_method",
    horizontal=True
)

answer_text = ""
uploaded_file = None
pdf_path_to_process = None # Track temp path

if submission_method == "Type directly":
    answer_text = st.text_area("Type your answer here:", height=300, key="typed_answer")
else: # Upload handwritten PDF
    uploaded_file = st.file_uploader("Upload a PDF of your handwritten answer", type=["pdf"], key="pdf_uploader")
    if uploaded_file is not None:
         # Save uploaded file temporarily
         with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
             tmp_file.write(uploaded_file.getvalue())
             pdf_path_to_process = tmp_file.name
             st.info("PDF uploaded. Click 'Evaluate Answer' to process.")


# --- Evaluation Button ---
st.divider()
if st.button("Evaluate Answer", key="evaluate_button"):
    student_answer = ""
    if submission_method == "Type directly":
        student_answer = answer_text.strip()
    elif uploaded_file is not None and pdf_path_to_process:
        student_answer = extract_text_from_pdf(pdf_path_to_process, ocr_pipe)

    if not student_answer:
        st.warning("Please provide an answer (type or upload PDF) before evaluating.")
    else:
        st.subheader("Evaluation Results")
        reference_answer = upsc_qa[selected_qid]["answer"]
        question_text = upsc_qa[selected_qid]["question"]

        # Perform evaluation
        similarity_score = evaluate_answer(student_answer, reference_answer, similarity_model)
        st.metric(label="Semantic Similarity Score", value=f"{similarity_score*100:.1f}%")

        # Display extracted/typed text (optional for MVP, but helpful)
        with st.expander("View Submitted Answer Text"):
             st.text_area("Your Answer:", value=student_answer, height=200, disabled=True)


        # Generate feedback
        suggestions = generate_suggestions(question_text, reference_answer, student_answer, feedback_model, feedback_tokenizer)
        st.subheader("AI Feedback & Suggestions")
        for suggestion in suggestions:
            st.markdown(f"- {suggestion}")

        # Add reference answer for comparison
        with st.expander("View Reference Answer"):
             st.text_area("Reference Answer:", value=reference_answer, height=200, disabled=True)

    # Clean up temporary file after processing
    if pdf_path_to_process and os.path.exists(pdf_path_to_process):
        try:
            os.remove(pdf_path_to_process)
            # st.caption(f"Cleaned up temporary file: {pdf_path_to_process}") # Optional debug message
        except Exception as e:
            st.warning(f"Could not delete temporary file: {e}")
