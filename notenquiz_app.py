import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import os
import time
import io
import re
from pypdf import PdfReader
import nltk
from nltk.tokenize import sent_tokenize
import random
import re
import requests
import zipfile
from io import BytesIO 
import shutil # Needed for folder deletion in extraction

# --- Setup and Constants ---
# FIX: Rely on simple LookupError or generic Exception which Python can always resolve.
try:
    nltk.data.find('tokenizers/punkt')
except LookupError: # Using LookupError as the safest, standard Python exception for missing resources
    nltk.download('punkt')

# --- Model and Generation Configuration ---
MODEL_PATH = "./final_notes_quiz_model"
MAX_INPUT_LENGTH = 512
MAX_OUTPUT_LENGTH = 200   
QA_MAX_LENGTH = 30        
QUESTION_WORDS = ["who", "what", "where", "when", "why", "how", "which"] 

# Delimiters (Must match training and parsing logic)
MCQ_DELIMITER = "[MCQ]"
OPTIONS_DELIMITER = "[OPT]"
ANSWER_DELIMITER = "[ANS]"
SUM_DELIMITER = "[SUM]"

# ----------------------------------------------------------------------
# 🌟 CRITICAL FIX: st.set_page_config() MUST BE THE FIRST STREAMLIT COMMAND
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Study Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded" 
)
# ----------------------------------------------------------------------

# --- Core Helper Functions ---

def get_confirm_token(response):
    """Parses Google Drive's redirect page for the confirmation token."""
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

@st.cache_resource
def load_model():
    """
    Downloads, unzips, and loads the T5 model from a public ZIP file using 
    a robust method that handles Google Drive's security checks.
    """
    
    # 🛑 CRITICAL: YOUR CONSTRUCTED DIRECT DOWNLOAD LINK IS NOW HERE
    FILE_ID = "1dZfxKtpok84u2QI2egnuR39j8GclYdoC"
    DOWNLOAD_URL = "https://docs.google.com/uc?export=download" # Base URL for security check
    # -----------------------------------------------------------
    
    # 1. Check if the model is already downloaded/extracted
    if not os.path.exists(MODEL_PATH) or not os.listdir(MODEL_PATH):
        st.info("Attempting robust download from Google Drive...")

        try:
            # --- PHASE 1: Initiate Download and Get Confirmation Token ---
            session = requests.Session()
            response = session.get(DOWNLOAD_URL, params={'id': FILE_ID}, stream=True, timeout=300)
            token = get_confirm_token(response)

            # --- PHASE 2: Execute Final Download with Token ---
            if token:
                params = {'id': FILE_ID, 'confirm': token}
                response = session.get(DOWNLOAD_URL, params=params, stream=True, timeout=300)
            
            # Final check on response content type (should be application/zip)
            if 'html' in response.headers.get('Content-Type', '').lower() or response.status_code != 200:
                 raise Exception("Download received HTML content instead of a ZIP file.")

            # --- CRITICAL EXTRACTION FIX ---
            os.makedirs(MODEL_PATH, exist_ok=True)
            
            with st.spinner("Extracting model files..."):
                zip_content = BytesIO(response.content)
                with zipfile.ZipFile(zip_content) as zip_ref:
                    
                    # Extract the contents into a temporary directory
                    temp_dir = "./temp_model_extract"
                    zip_ref.extractall(temp_dir)
                    
                    # Determine the source directory inside the zip (handles nested folders)
                    source_dir = temp_dir
                    if len(os.listdir(temp_dir)) == 1 and os.path.isdir(os.path.join(temp_dir, os.listdir(temp_dir)[0])):
                        source_dir = os.path.join(temp_dir, os.listdir(temp_dir)[0])
                    
                    # Move files from the source directory to the final MODEL_PATH
                    for item_name in os.listdir(source_dir):
                        shutil.move(os.path.join(source_dir, item_name), MODEL_PATH)
                    
                    shutil.rmtree(temp_dir)
            
            st.success("Model downloaded and extracted successfully!")
        
        except Exception as e:
            st.error(f"FATAL DOWNLOAD ERROR: {e}. Please ensure the Google Drive file is shared as 'Anyone with the link' and the ID is correct.")
            st.stop()

    # 2. Load the Model
    device = torch.device("cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu")
    
    with st.spinner(f"Loading T5 model onto {device}..."):
        tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
        model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)
    
    return tokenizer, model, device

def extract_text_from_file(uploaded_file):
    """Extracts text content from a PDF or TXT file."""
    if uploaded_file.type == "text/plain":
        string_io = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
        text = string_io.read()
        return text.strip()
    
    elif uploaded_file.type == "application/pdf":
        try:
            reader = PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {e}. Ensure the PDF is not encrypted.")
            return None
    else:
        st.warning("Unsupported file type. Please upload a PDF or TXT file.")
        return None

def generate_output(input_text, tokenizer, model, device, max_length, temperature=0.7):
    """Generates the output using the T5 model, with tunable temperature and better beams."""
    try:
        input_ids = tokenizer.encode(
            input_text, 
            return_tensors="pt", 
            max_length=MAX_INPUT_LENGTH, 
            truncation=True
        ).to(device)

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=max_length,
                num_beams=8, 
                early_stopping=True,
                temperature=temperature, 
                top_k=50,
                no_repeat_ngram_size=3
            )
        
        return tokenizer.decode(output[0], skip_special_tokens=True)

    except Exception as e:
        return f"An error occurred during generation: {e}"

def clean_answer(answer):
    """Cleans up T5-generated answers."""
    answer = answer.strip()
    answer = re.sub(r'^\W+', '', answer)
    answer = re.sub(r'\.$', '', answer)
    return answer.capitalize()

def parse_mcq_output(result_raw):
    """Parses MCQs with maximum forgiveness and fallbacks."""
    
    answer = "N/A"
    answer_match = re.search(r'(?:answer:|' + re.escape(ANSWER_DELIMITER) + r')\s*(.*)', result_raw, re.IGNORECASE)
    if answer_match:
        answer = clean_answer(answer_match.group(1).strip())
    
    options_raw = result_raw
    if answer_match:
        options_raw = result_raw[:answer_match.start()].strip()
    
    if "options:" in options_raw:
        options_raw = options_raw.split("options:", 1)[-1].strip()

    options_list = []
    
    if OPTIONS_DELIMITER in options_raw:
        options_list = [clean_answer(opt) for opt in options_raw.split(OPTIONS_DELIMITER) if opt.strip()]
    elif '|' in options_raw:
        options_list = [clean_answer(opt) for opt in options_raw.split('|') if opt.strip()]
    else:
         options_list = [clean_answer(options_raw)]
        
    return options_list[:4], answer

def generate_single_quiz_item(context_chunk, question_type, tokenizer, model, device):
    """Generates a quiz item, RELAXES filters to increase count."""
    if len(context_chunk.split()) < 40:
        return None, None, None
        
    sentences = sent_tokenize(context_chunk)
    if not sentences:
        return None, None, None
    
    try:
        if question_type == "QG":
            # --- QG: Generate Question and Chain to Answer ---
            qg_prefix = f"generate question: context: {context_chunk} answer: a key detail"
            generated_question = generate_output(qg_prefix, tokenizer, model, device, max_length=64, temperature=0.8)

            # --- HEURISTIC 1 (MAXIMUM RELAXATION): Only check for question mark and min length ---
            if "?" not in generated_question or len(generated_question.split()) < 3:
                 return None, None, None 

            # --- Step 2: Generate Answer ---
            qa_prefix = f"question: {generated_question} context: {context_chunk}"
            generated_answer = generate_output(qa_prefix, tokenizer, model, device, max_length=QA_MAX_LENGTH, temperature=0.1)
            
            # --- HEURISTIC 2 (RELAXED): Answer length check is more forgiving ---
            cleaned_answer = clean_answer(generated_answer)
            if not cleaned_answer or len(cleaned_answer.split()) > 15:
                 return None, None, None

            return "QG_QA", generated_question, cleaned_answer

        elif question_type == "MCQ":
            # --- MCQ: Generate the full structured string ---
            input_prefix = f"generate mcq: context: {context_chunk}"
            result_raw = generate_output(input_prefix, tokenizer, model, device, max_length=MAX_OUTPUT_LENGTH, temperature=0.7)
            
            # --- HEURISTIC 3: Must contain at least one separator to be attempted for parsing ---
            if not ("|" in result_raw or "options:" in result_raw or ANSWER_DELIMITER in result_raw):
                 return None, None, None

            return "MCQ", "Select the correct option from this chunk.", result_raw
    
    except Exception:
        return None, None, None
        
    return None, None, None

# --- Main App Execution Starts Here ---

# 1. Load model (cached)
tokenizer, model, device = load_model()

# --- Sidebar Content ---
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("This AI-powered tool helps you:")
    st.markdown("- **Generate concise and detailed notes**")
    st.markdown("- **Create interactive quizzes**")
    st.markdown("- **Test your understanding**")
    
    st.markdown("---")
    
    st.markdown("**Supported formats:**")
    st.markdown("- **PDF** (`.pdf`)")
    st.markdown("- **Text** (`.txt`)")
    
    st.markdown("---")
    
    st.markdown("**How to use:**")
    st.markdown("1. Upload your study material")
    st.markdown("2. Choose Notes or Quiz")
    st.markdown("3. Review and download results")
    
# --- Main Content Section ---

st.title("🧠 Smart Study Assistant")
st.markdown("### Upload your study material and generate Notes or Quizzes")

# 1. File Uploader Section
uploaded_file = st.file_uploader(
    "Choose your file (PDF or TXT)",
    type=["txt", "pdf"],
    key="file_uploader_main"
)
st.caption("Limit 200MB per file · PDF, TXT")
st.markdown("---")

# 2. Configuration
st.subheader("⚙️ Configuration")
col1, col2, col3 = st.columns(3)

with col1:
    output_choice = st.radio(
        "Select Output Type:",
        ('Generate Both (Notes & Quiz)', 'Notes Only', 'Quiz Only'),
        index=0 
    )

with col2:
    difficulty = st.select_slider(
        "Quiz Difficulty:",
        options=['Easy', 'Medium', 'Hard'],
        value='Medium'
    )

with col3:
    num_questions = st.slider(
        "Number of Questions:",
        min_value=1,
        max_value=10,
        value=5
    )

generate_button = st.button("Generate Smart Study Material", type="primary")
st.markdown("---")

# --- Generation Logic ---

if generate_button:
    
    if not uploaded_file:
        st.warning("Please upload a file to begin generating material.")
        st.stop()
    
    # Set parameters based on difficulty
    if difficulty == 'Easy':
        chunk_words = 150
        question_type = "QG" 
    elif difficulty == 'Medium':
        chunk_words = 100
        question_type = "QG"
    else: # Hard
        chunk_words = 60
        question_type = "MCQ"

    with st.status("Processing and Generating...", expanded=True) as status:
        
        status.update(label="Extracting text from file...", state="running", expanded=True)
        full_context = extract_text_from_file(uploaded_file)
        
        if not full_context or len(full_context.split()) < 40:
            status.update(label="Text extraction failed or document is too short (< 40 words).", state="error")
            st.stop()
        
        sentences = sent_tokenize(full_context)
        total_words = len(full_context.split())
        st.success(f"File processed! Extracted {total_words} words.")
        
        start_time = time.time()

        # ----------------------------------------------------
        # PHASE 1: GENERATE SUMMARY NOTES
        # ----------------------------------------------------
        if output_choice in ['Generate Both (Notes & Quiz)', 'Notes Only']:
            status.update(label="Generating Summary Notes (Phase 1/2): Applying Cleaning Heuristics...", state="running")
            
            summary_max_chunk_words = 250 
            summary_chunks = []
            current_chunk = ""
            
            for sent in sentences:
                if len((current_chunk + " " + sent).split()) < summary_max_chunk_words:
                    current_chunk += " " + sent
                else:
                    summary_chunks.append(current_chunk.strip())
                    current_chunk = sent
            if current_chunk:
                summary_chunks.append(current_chunk.strip())

            full_summary_sentences = []
            
            for chunk in summary_chunks:
                input_prefix = f"summarize: {chunk}"
                generated_summary_raw = generate_output(input_prefix, tokenizer, model, device, max_length=MAX_OUTPUT_LENGTH, temperature=0.9)
                
                # Robust summary parsing
                long_summary = generated_summary_raw
                if "| long:" in generated_summary_raw:
                    long_summary = generated_summary_raw.split("| long:")[-1].strip()
                elif "long:" in generated_summary_raw:
                    long_summary = generated_summary_raw.split("long:")[-1].strip()
                
                for sent in sent_tokenize(long_summary):
                    if sent not in full_summary_sentences and len(sent.split()) > 5:
                         full_summary_sentences.append(sent)

            final_summary = " ".join(full_summary_sentences)
            
            st.subheader("📖 Your Concise Notes")
            st.markdown(final_summary)
            st.download_button(
                "Download Notes as TXT",
                data=final_summary.strip(),
                file_name="smart_notes.txt",
                mime="text/plain"
            )

        # ----------------------------------------------------
        # PHASE 2: GENERATE QUIZ
        # ----------------------------------------------------
        if output_choice in ['Generate Both (Notes & Quiz)', 'Quiz Only']:
            status.update(label=f"Generating {num_questions} Quiz Questions (Phase 2/2): Applying Filters...", state="running")
            
            quiz_chunks = []
            current_quiz_chunk = ""
            for sent in sentences:
                if len((current_quiz_chunk + " " + sent).split()) < chunk_words:
                    current_quiz_chunk += " " + sent
                else:
                    quiz_chunks.append(current_quiz_chunk.strip())
                    current_quiz_chunk = sent
            if current_quiz_chunk:
                quiz_chunks.append(current_quiz_chunk.strip())
            
            if len(quiz_chunks) > num_questions * 3: 
                selected_chunks = random.sample(quiz_chunks, num_questions * 3)
            else:
                selected_chunks = quiz_chunks

            quiz_items = []
            
            for i, chunk in enumerate(selected_chunks):
                if len(quiz_items) >= num_questions:
                    break
                
                item_type, q_text_or_dummy, result_raw = generate_single_quiz_item(chunk, question_type, tokenizer, model, device)
                
                if item_type == "QG_QA":
                    quiz_items.append({
                        "type": "QG",
                        "question": f"Question {len(quiz_items) + 1}: {q_text_or_dummy.strip()} (Difficulty: {difficulty})",
                        "answer": result_raw
                    })
                
                elif item_type == "MCQ":
                    # --- ROBUST MCQ PARSING ---
                    options_list, answer = parse_mcq_output(result_raw)
                    
                    try:
                        # Final check for validity: must have 3+ options and a valid answer
                        if len(options_list) >= 3 and answer not in ["N/A", ""]:
                            quiz_items.append({
                                "type": "MCQ",
                                "question": f"Question {len(quiz_items) + 1}: Select the correct option. (Difficulty: {difficulty})",
                                "options": options_list,
                                "answer": answer
                            })
                        # else: item is silently skipped due to parsing/quality failure

                    except Exception:
                        pass 

            # Display Quiz and Final Feedback
            generated_count = len(quiz_items)
            
            if generated_count < num_questions:
                st.warning(f"Could only generate {generated_count} out of {num_questions} requested items due to model instability or strict output filtering.")
                
            st.subheader("📚 Generated Quiz")
            st.markdown(f"**Difficulty:** {difficulty} | **Type:** {question_type} | **Questions Generated:** {generated_count}")

            quiz_text_for_download = ""

            for i, item in enumerate(quiz_items):
                st.markdown(f"**{item.get('question', f'Question {i+1}')}**")
                quiz_text_for_download += f"Q{i+1}: {item.get('question', f'Question {i+1}')}\n"
                
                if item['type'] == 'MCQ':
                    options_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
                    for idx, option in enumerate(item['options']):
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;**{options_map.get(idx, '?')}.** {option}")
                        quiz_text_for_download += f"  {options_map.get(idx, '?')}. {option}\n"
                    st.markdown(f"**Correct Answer:** `{item['answer']}`")
                    quiz_text_for_download += f"Answer: {item['answer']}\n\n"
                    
                elif item['type'] == 'QG':
                    st.markdown(f"**Factual Answer:** `{item['answer']}`")
                    quiz_text_for_download += f"Answer: {item['answer']}\n\n"
                
                st.markdown("---")

            st.download_button(
                "Download Quiz with Answers (TXT)",
                data=quiz_text_for_download,
                file_name=f"smart_quiz_{difficulty}_{len(quiz_items)}q.txt",
                mime="text/plain"
            )

        # Final Status Update
        end_time = time.time()
        status.update(label=f"Generation Complete! Total Time: {end_time - start_time:.2f} seconds.", state="complete")
