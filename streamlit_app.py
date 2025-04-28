# streamlit_app.py
import streamlit as st
import os
import tempfile
from pdf2image import convert_from_path
import numpy as np
from PIL import Image
import torch # PyTorch is often a dependency for transformers

# --- Hugging Face Imports ---
# Using try-except for optional dependencies or different environments
try:
    from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
    from sentence_transformers import SentenceTransformer, util
    hf_import_success = True
except ImportError:
    st.error("""
        Failed to import Hugging Face libraries. Please install them:
        `pip install streamlit torch transformers sentence-transformers Pillow pdf2image python-dotenv`
        You might also need system dependencies like poppler:
        - Ubuntu/Debian: `sudo apt-get update && sudo apt-get install -y poppler-utils`
        - macOS: `brew install poppler`
        - Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases/
    """)
    hf_import_success = False
    st.stop() # Stop execution if core libraries are missing

# --- Page Config ---
st.set_page_config(
    page_title="SelectionAI HF - Answer Evaluation",
    page_icon="🤖",
    layout="wide"
)

# --- Constants ---
# Choose models (consider smaller models if resource-constrained)
OCR_MODEL = "microsoft/trocr-base-handwritten" # Good for handwritten text
# OCR_MODEL = "microsoft/trocr-base-printed" # Better for printed text
SIMILARITY_MODEL = 'sentence-transformers/all-MiniLM-L6-v2' # Efficient & effective
FEEDBACK_MODEL = 'google/flan-t5-base' # Good balance of capability and size

# --- Model Loading (Cached) ---
# Cache models to avoid reloading on every interaction
@st.cache_resource
def load_ocr_pipeline():
    """Loads the Hugging Face OCR pipeline."""
    try:
        # Ensure device is set correctly (use GPU if available)
        device = 0 if torch.cuda.is_available() else -1
        ocr_pipeline = pipeline("image-to-text", model=OCR_MODEL, device=device)
        st.success(f"OCR Model ({OCR_MODEL}) loaded successfully.")
        return ocr_pipeline
    except Exception as e:
        st.error(f"Error loading OCR model ({OCR_MODEL}): {e}")
        st.info("This might be due to network issues, model availability, or incompatible libraries.")
        return None

@st.cache_resource
def load_similarity_model():
    """Loads the Sentence Transformer model."""
    try:
        model = SentenceTransformer(SIMILARITY_MODEL)
        st.success(f"Similarity Model ({SIMILARITY_MODEL}) loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading Similarity model ({SIMILARITY_MODEL}): {e}")
        return None

@st.cache_resource
def load_feedback_model_and_tokenizer():
    """Loads the Flan-T5 model and tokenizer for feedback generation."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(FEEDBACK_MODEL)
        model = AutoModelForSeq2SeqLM.from_pretrained(FEEDBACK_MODEL)
        st.success(f"Feedback Model ({FEEDBACK_MODEL}) loaded successfully.")
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading Feedback model ({FEEDBACK_MODEL}): {e}")
        return None, None

# Load models only if Hugging Face libraries were imported successfully
if hf_import_success:
    ocr_pipe = load_ocr_pipeline()
    similarity_model = load_similarity_model()
    feedback_model, feedback_tokenizer = load_feedback_model_and_tokenizer()
else:
    ocr_pipe, similarity_model, feedback_model, feedback_tokenizer = None, None, None, None


# --- UPSC Questions and Answers (Keep as is) ---
upsc_qa = {
    1: {
        "question": "Discuss the significance of the 73rd Constitutional Amendment Act in strengthening grassroots democracy in India.",
        "answer": "The 73rd Constitutional Amendment Act of 1992 institutionalized Panchayati Raj in India, strengthening grassroots democracy. It granted constitutional status to rural local bodies and mandated the creation of Panchayati Raj Institutions (PRIs) at village, intermediate, and district levels. Regular elections every five years, financial devolution, and reservation of seats for Scheduled Castes, Scheduled Tribes, and women were introduced to ensure inclusivity and empowerment. By facilitating citizen participation in governance, PRIs have enhanced transparency, accountability, and development at the local level. They have also fostered leadership among marginalized sections and women, contributing to social justice. However, challenges like limited financial autonomy, political interference, and bureaucratic dominance still persist, demanding further reforms to realize the full potential of grassroots democracy."
    },
    2: {
        "question": "Analyze the impact of globalization on India's cultural diversity.",
        "answer": "Globalization has significantly influenced India's cultural landscape. Increased exposure to global ideas has enriched Indian traditions, cuisine, music, and cinema. Fusion forms in art, fashion, and lifestyle symbolize cultural amalgamation. Indian culture, through Bollywood, yoga, and festivals, has gained international acceptance. However, globalization has also triggered fears of cultural homogenization, where Western influences sometimes overshadow indigenous languages, crafts, and traditional values. There is a growing trend among urban youth to adopt Western lifestyles, sometimes at the cost of their cultural roots. Yet, globalization has also revived interest in India's heritage among diasporic communities and bolstered the economy through cultural tourism. Hence, globalization's impact on India's cultural diversity is complex — both enriching and challenging."
    }
    # Add more questions as needed
}

# --- Core Functions (Modified) ---

# Cache data processing functions
@st.cache_data(show_spinner=False) # Show spinner manually inside
def extract_text_from_pdf_hf(pdf_path, _ocr_pipeline):
    """Extract text from PDF using Hugging Face OCR pipeline"""
    if _ocr_pipeline is None:
        st.error("OCR Pipeline not loaded. Cannot extract text.")
        return ""

    extracted_text = ""
    try:
        with st.spinner('Converting PDF to images...'):
            # Use thread_count for potential speedup, adjust based on system
            images = convert_from_path(pdf_path, thread_count=4)

        st.info(f"Found {len(images)} page(s) in the PDF.")

        with st.spinner(f'Performing OCR using {OCR_MODEL}...'):
            progress_bar = st.progress(0)
            for i, image in enumerate(images):
                try:
                    # Convert PIL Image to format suitable for pipeline if needed
                    # Most pipelines handle PIL images directly
                    result = _ocr_pipeline(image)
                    # The output format might vary slightly depending on the pipeline/model
                    page_text = result[0]['generated_text'] if isinstance(result, list) and result else \
                                result['generated_text'] if isinstance(result, dict) else str(result)

                    extracted_text += page_text + "\n\n" # Add space between pages
                except Exception as page_e:
                    st.warning(f"Could not process page {i+1}: {page_e}")
                finally:
                    progress_bar.progress((i + 1) / len(images))
            progress_bar.empty() # Remove progress bar after completion

    except Exception as e:
        st.error(f"An error occurred during PDF processing or OCR: {e}")
        # Provide more specific guidance if possible (e.g., Poppler path issues)
        if "poppler" in str(e).lower():
            st.error("This might indicate an issue with the Poppler installation or its path configuration.")
        return "" # Return empty string on failure

    if not extracted_text.strip():
        st.warning("OCR process completed, but no text was extracted. The PDF might be image-only with no machine-readable text, or the OCR model struggled with the content.")

    return extracted_text.strip()

@st.cache_data(show_spinner="Calculating semantic similarity...")
def evaluate_answer_hf(_similarity_model, extracted_text, reference_answer):
    """Compare the extracted text with the reference answer using Sentence Transformers"""
    if _similarity_model is None:
        st.error("Similarity Model not loaded. Cannot evaluate.")
        return {"similarity": 0.0, "reference_answer": reference_answer}
    if not extracted_text or not reference_answer:
        return {"similarity": 0.0, "reference_answer": reference_answer}

    try:
        # Encode texts into embeddings
        embedding1 = _similarity_model.encode(reference_answer, convert_to_tensor=True)
        embedding2 = _similarity_model.encode(extracted_text, convert_to_tensor=True)

        # Calculate cosine similarity
        similarity_score = util.pytorch_cos_sim(embedding1, embedding2).item() # Get scalar value

        # Clamp score between 0 and 1 (sometimes similarity can be slightly outside)
        similarity_score = max(0.0, min(1.0, similarity_score))

        return {
            "similarity": similarity_score,
            "reference_answer": reference_answer
        }
    except Exception as e:
        st.error(f"Error calculating similarity: {e}")
        return {"similarity": 0.0, "reference_answer": reference_answer}


@st.cache_data(show_spinner="Generating feedback with AI...")
def generate_suggestions_hf(_feedback_model, _feedback_tokenizer, question, reference_answer, student_answer):
    """Generate suggestions for improving the answer using a generative AI model"""
    if _feedback_model is None or _feedback_tokenizer is None:
        st.error("Feedback Model/Tokenizer not loaded. Cannot generate suggestions.")
        return ["Feedback generation failed: Model not available."]
    if not student_answer:
         return ["Cannot generate feedback for an empty answer."]

    try:
        # Construct a detailed prompt for the T5 model
        prompt = f"""
        Context: Evaluate a student's answer for a UPSC (Indian Civil Services Exam) question.
        Question: {question}

        Reference Answer (Ideal Key Points): {reference_answer}

        Student's Answer: {student_answer}

        Task: Provide constructive feedback for the student. Analyze the student's answer based on the question and reference answer. Identify strengths and weaknesses regarding:
        1. Relevance: Does the answer directly address the question?
        2. Completeness: Are the key points from the reference answer covered?
        3. Accuracy: Is the information presented correct?
        4. Structure: Is the answer well-organized?
        5. Clarity: Is the language clear and concise?

        Output: Provide a bulleted list of specific, actionable suggestions for improvement. Start with a brief overall assessment. Do not just repeat the reference answer. Focus on *how* the student can improve.
        Feedback:
        """

        # Ensure CUDA availability check for generation if applicable
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _feedback_model.to(device) # Ensure model is on the correct device

        inputs = _feedback_tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(device)

        # Adjust generation parameters as needed
        outputs = _feedback_model.generate(
            inputs.input_ids,
            max_length=300,  # Max length of the generated feedback
            num_beams=4,     # Beam search for better quality
            early_stopping=True,
            no_repeat_ngram_size=2 # Avoid repetitive phrases
        )

        suggestions_text = _feedback_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Basic formatting (split into bullet points if model followed instructions)
        suggestions_list = [s.strip() for s in suggestions_text.split('\n') if s.strip() and s.strip().startswith(('*', '-', '•'))]
        if not suggestions_list: # Fallback if model didn't use bullets
             suggestions_list = [suggestions_text]

        return suggestions_list

    except Exception as e:
        st.error(f"Error generating feedback: {e}")
        return ["An error occurred while generating feedback."]


# --- Main App UI (Similar Structure, Updated Logic) ---
st.title("SelectionAI - Enhanced Evaluation")
st.subheader("UPSC Answer Evaluation using Hugging Face Models")

with st.expander("About SelectionAI & Models Used", expanded=False):
    st.write(f"""
    SelectionAI helps you evaluate your UPSC exam answers by:
    1. Allowing you to select a question to answer.
    2. Extracting text from your handwritten answer using **{OCR_MODEL}** from Hugging Face.
    3. Comparing your answer with a reference answer using semantic similarity calculated by **{SIMILARITY_MODEL}**.
    4. Providing AI-generated feedback and suggestions for improvement using **{FEEDBACK_MODEL}**.

    Simply select a question, upload a PDF with your handwritten answer, or type your answer directly!
    """)
    st.caption(f"Models loaded: OCR ({'Yes' if ocr_pipe else 'No'}), Similarity ({'Yes' if similarity_model else 'No'}), Feedback ({'Yes' if feedback_model else 'No'})")


# Check if models loaded successfully before proceeding
if not all([ocr_pipe, similarity_model, feedback_model, feedback_tokenizer]):
     st.warning("One or more AI models failed to load. Functionality will be limited. Please check the errors above and your environment setup.")
     # Optionally disable parts of the UI if models are missing
     # st.stop() # Or stop completely if core models are missing

# Initialize session state variables
default_question_id = list(upsc_qa.keys())[0]
if 'selected_question_id' not in st.session_state:
    st.session_state.selected_question_id = default_question_id
if 'answer_text' not in st.session_state:
    st.session_state.answer_text = "" # Stores typed text or extracted OCR text
if 'evaluation_done' not in st.session_state:
    st.session_state.evaluation_done = False
if 'evaluation_results' not in st.session_state: # Stores similarity score etc.
    st.session_state.evaluation_results = None
if 'suggestions' not in st.session_state: # Stores AI generated feedback
    st.session_state.suggestions = []

# --- Tabs for Workflow ---
tab1, tab2 = st.tabs(["1. Select Question", "2. Submit & Evaluate"])

# --- Tab 1: Question Selection ---
with tab1:
    st.header("Select a UPSC Question")

    # Create a dropdown with truncated question previews
    question_options = {
        f"Q{k}: {v['question'][:70]}...": k
        for k, v in upsc_qa.items()
    }
    # Find the index corresponding to the current session state ID
    current_index = list(question_options.values()).index(st.session_state.selected_question_id)

    selected_question_preview = st.selectbox(
        "Choose a question to answer:",
        options=list(question_options.keys()),
        index=current_index # Use the calculated index
    )

    # Update session state if selection changes
    new_question_id = question_options[selected_question_preview]
    if st.session_state.selected_question_id != new_question_id:
        st.session_state.selected_question_id = new_question_id
        # Reset previous evaluation results when question changes
        st.session_state.evaluation_done = False
        st.session_state.answer_text = ""
        st.session_state.evaluation_results = None
        st.session_state.suggestions = []
        st.rerun() # Rerun to update the displayed question below

    # Display the full selected question
    question_id = st.session_state.selected_question_id
    st.subheader("Selected Question:")
    st.markdown(f"**Q{question_id}:** {upsc_qa[question_id]['question']}")

    st.info("➡️ After selecting your question, go to the 'Submit & Evaluate' tab.")


# --- Tab 2: Submission and Evaluation ---
with tab2:
    selected_qid = st.session_state.selected_question_id
    st.header(f"Answer for Question {selected_qid}")
    st.write(f"**Question:** {upsc_qa[selected_qid]['question']}")
    st.divider()

    st.subheader("Submit Your Answer")

    submission_method = st.radio(
        "Choose submission method:",
        ["Type your answer", "Upload handwritten PDF"],
        key="submission_method",
        horizontal=True
    )

    pdf_path_to_delete = None # To track temporary file for deletion

    # --- Handling Typed Input ---
    if submission_method == "Type your answer":
        st.session_state.answer_text = st.text_area(
            "Type your answer here:",
            value=st.session_state.get("answer_text", ""), # Use get for safety
            height=300,
            key="typed_answer_area"
        )

        if st.button("Evaluate Typed Answer", key="eval_typed_btn", disabled=(not all([similarity_model, feedback_model]))):
            if not st.session_state.answer_text.strip():
                st.error("Please type an answer before evaluation.")
            else:
                # Models are needed here
                if similarity_model and feedback_model and feedback_tokenizer:
                    reference_answer = upsc_qa[selected_qid]["answer"]
                    question_text = upsc_qa[selected_qid]["question"]

                    # Perform evaluation and generate suggestions
                    evaluation = evaluate_answer_hf(similarity_model, st.session_state.answer_text, reference_answer)
                    suggestions = generate_suggestions_hf(feedback_model, feedback_tokenizer, question_text, reference_answer, st.session_state.answer_text)

                    # Store results
                    st.session_state.evaluation_results = evaluation
                    st.session_state.suggestions = suggestions
                    st.session_state.evaluation_done = True
                    st.rerun() # Rerun to display results section
                else:
                    st.error("Models required for evaluation are not loaded. Cannot proceed.")

    # --- Handling PDF Upload ---
    else: # Upload handwritten PDF
        uploaded_file = st.file_uploader(
            f"Upload a PDF of your handwritten answer (OCR Model: {OCR_MODEL})",
            type=["pdf"],
            key="pdf_uploader"
            )

        if uploaded_file is not None:
            st.info(f"File '{uploaded_file.name}' uploaded. Click below to process.")

            if st.button("Process and Evaluate PDF", key="eval_pdf_btn", disabled=(not all([ocr_pipe, similarity_model, feedback_model]))):
                 # Ensure all models are loaded before proceeding
                if ocr_pipe and similarity_model and feedback_model and feedback_tokenizer:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        pdf_path = tmp_file.name
                        pdf_path_to_delete = pdf_path # Mark for deletion later

                    try:
                        # 1. Extract text using HF OCR pipeline
                        extracted_text = extract_text_from_pdf_hf(pdf_path, ocr_pipe)
                        st.session_state.answer_text = extracted_text # Store extracted text

                        if not extracted_text:
                            st.error("Text extraction failed or yielded no text. Cannot evaluate.")
                            st.session_state.evaluation_done = False
                        else:
                             # 2. Evaluate the answer
                            reference_answer = upsc_qa[selected_qid]["answer"]
                            evaluation = evaluate_answer_hf(similarity_model, extracted_text, reference_answer)

                            # 3. Generate suggestions
                            question_text = upsc_qa[selected_qid]["question"]
                            suggestions = generate_suggestions_hf(feedback_model, feedback_tokenizer, question_text, reference_answer, extracted_text)

                            # Store results
                            st.session_state.evaluation_results = evaluation
                            st.session_state.suggestions = suggestions
                            st.session_state.evaluation_done = True
                            st.rerun() # Rerun to display results

                    except Exception as e:
                        st.error(f"An unexpected error occurred during PDF processing/evaluation: {e}")
                        st.session_state.evaluation_done = False

                    finally:
                        # Clean up the temporary file
                        if pdf_path_to_delete and os.path.exists(pdf_path_to_delete):
                            try:
                                os.remove(pdf_path_to_delete)
                            except PermissionError:
                                st.warning(f"Could not delete temporary file {pdf_path_to_delete}. It might be locked.")
                else:
                    st.error("One or more required AI models are not loaded. Cannot process PDF.")


    st.divider()

    # --- Display Evaluation Results ---
    if st.session_state.get("evaluation_done", False):
        st.header("📊 Evaluation Results & Feedback")

        eval_results = st.session_state.evaluation_results
        suggestions = st.session_state.suggestions
        submitted_text = st.session_state.answer_text

        if eval_results:
            results_tab, feedback_tab, reference_tab = st.tabs(["Evaluation Score", "AI Feedback", "Reference Answer"])

            with results_tab:
                st.subheader("Your Answer (Submitted/Extracted)")
                st.text_area("Submitted Text:", value=submitted_text, height=200, disabled=True)

                st.subheader("Semantic Similarity Score")
                similarity_score = eval_results.get("similarity", 0.0) * 100

                # Display score with color coding
                score_color = "green" if similarity_score >= 70 else "orange" if similarity_score >= 50 else "red"
                st.markdown(f"**Score:** <span style='color:{score_color}; font-size: 1.2em;'>{similarity_score:.1f}%</span> (Compared to reference answer using '{SIMILARITY_MODEL}')", unsafe_allow_html=True)

                # Visualize score
                st.progress(min(float(similarity_score/100), 1.0))

                # Performance interpretation based on similarity
                if similarity_score >= 80:
                    st.success("✅ Excellent! Your answer shows strong semantic similarity to the reference.")
                elif similarity_score >= 70:
                    st.success("👍 Very Good! Your answer covers most key concepts effectively.")
                elif similarity_score >= 60:
                    st.info("🙂 Good! Your answer addresses the main points but could be more aligned.")
                elif similarity_score >= 50:
                    st.warning("🤔 Fair. There's overlap, but significant differences exist.")
                else:
                    st.error("⚠️ Needs Improvement. Your answer differs significantly from the reference.")
                st.caption("Note: Semantic similarity measures how close the *meaning* is, not just keyword overlap. A high score indicates conceptual alignment.")


            with feedback_tab:
                st.subheader(f"AI-Generated Feedback (using {FEEDBACK_MODEL})")
                if suggestions:
                    st.info("Here's feedback based on the question, reference, and your answer:")
                    for i, suggestion in enumerate(suggestions, 1):
                        st.markdown(f"{suggestion}") # Use markdown for potential formatting from model
                else:
                    st.warning("Could not generate feedback.")

                st.subheader("General Writing Tips (UPSC Context)")
                st.write("""
                * **Structure:** Clear Intro-Body-Conclusion is vital. Use paragraphs effectively.
                * **Relevance:** Directly address *all* parts of the question. Avoid tangents.
                * **Content:** Use specific facts, data, examples, and keywords relevant to the syllabus.
                * **Analysis:** Don't just describe; analyze, critique, and provide balanced viewpoints where required.
                * **Clarity & Conciseness:** Write clearly. Avoid jargon where simpler terms suffice. Be mindful of word limits.
                """)

            with reference_tab:
                st.subheader("Reference Answer")
                st.info("Use this as a guide to understand the key points expected. Your answer might still be valid if structured differently but covering core concepts.")
                ref_answer = eval_results.get("reference_answer", "Reference answer not available.")
                st.markdown(ref_answer)
        else:
            st.warning("Evaluation results are not available.")

        # Reset button
        if st.button("Evaluate Another Answer / New Question"):
            # Clear relevant state
            st.session_state.evaluation_done = False
            st.session_state.answer_text = ""
            st.session_state.evaluation_results = None
            st.session_state.suggestions = []
            # Optionally navigate back to question selection or just clear the current state
            # st.session_state.selected_question_id = default_question_id # Uncomment to reset question too
            st.rerun()

st.divider()

# --- Section on Evaluating the AI's Evaluation ---
with st.expander("Evaluating the AI's Evaluation (Meta-Evaluation)"):
    st.markdown("""
    It's important to critically assess the AI's output (both the score and the feedback). Here's how you can think about it, along with prompts you could potentially use with another AI model (like GPT-4, Claude, or even another instance of Flan-T5) for assessment:

    **1. Assessing the Similarity Score:**
    * **Does the score *feel* right?** Based on your own understanding, does the percentage reasonably reflect how close your answer was to the reference?
    * **Limitations:** Semantic similarity doesn't always capture factual correctness perfectly or adherence to specific instructions (like word count). A high similarity score might still hide factual errors if the *structure* and *topic* are similar.

    * **Example Evaluation Prompt:**
        ```
        Question: [Paste the UPSC Question Here]
        Reference Answer: [Paste the Reference Answer Here]
        Student Answer: [Paste the Student's Answer Here]
        AI Similarity Score: [Paste the Score, e.g., 75.3%]

        Task: Critically evaluate the AI's similarity score. Does this score accurately reflect the semantic closeness and overall quality of the student's answer compared to the reference? Explain your reasoning, considering relevance, completeness, and potential limitations of semantic similarity. Is the score too high, too low, or about right? Why?
        ```

    **2. Assessing the AI-Generated Feedback:**
    * **Is it Relevant?** Does the feedback directly relate to the question, your answer, and the reference points?
    * **Is it Specific?** Does it point out concrete areas (e.g., "missing discussion on financial devolution", "needs clearer examples") or is it too generic (e.g., "improve your answer")?
    * **Is it Actionable?** Does the feedback give you clear steps or directions on *how* to improve?
    * **Is it Accurate?** Does the feedback correctly identify strengths/weaknesses? Did it hallucinate or misunderstand something?
    * **Is it Constructive?** Is the tone helpful?

    * **Example Evaluation Prompt:**
        ```
        Question: [Paste the UPSC Question Here]
        Reference Answer: [Paste the Reference Answer Here]
        Student Answer: [Paste the Student's Answer Here]
        AI Generated Feedback: [Paste the Bulleted List of Feedback Here]

        Task: Evaluate the quality of the AI-generated feedback based on the provided context. Rate its helpfulness on a scale of 1 (Not Helpful) to 5 (Very Helpful). Justify your rating by assessing its relevance, specificity, actionability, accuracy, and constructive tone. What are the strengths and weaknesses of this feedback?
        ```

    By performing this kind of meta-evaluation, you can better understand the capabilities and limitations of the AI tools used here.
    """)


# Add a footer
st.markdown("---")
st.markdown("© 2025 SelectionAI (Hugging Face Edition) - AI models for educational purposes.")

# --- Final Cleanup ---
# Ensure temp file is deleted if an error occurred before the 'finally' block in the PDF section
if pdf_path_to_delete and os.path.exists(pdf_path_to_delete):
     try:
         os.remove(pdf_path_to_delete)
     except Exception:
         pass # Ignore if deletion fails here, already warned user potentially
