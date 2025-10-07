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

import nltk
from nltk.downloader import DownloadError # <--- ADD THIS LINE (CRUCIAL)

# --- Configuration ---
MODEL_PATH = "./final_notes_quiz_model"
# ... other constants ...

# The actual download logic should be made safe now that DownloadError is imported
try:
    nltk.data.find('tokenizers/punkt')
except (DownloadError, LookupError): # USE DownloadError (now imported) instead of nltk.downloader.DownloadError
    nltk.download('punkt')

# --- Configuration ---
MODEL_PATH = "./final_notes_quiz_model"
MAX_INPUT_LENGTH = 512 # Increased max input length (matches T5-Base training)
MAX_OUTPUT_LENGTH = 200 # Increased output length for less truncation in summaries
QA_MAX_LENGTH = 30 # Increased answer length filter
QUESTION_WORDS = ["who", "what", "where", "when", "why", "how", "which"] 

# --- Delimiters (Must match your T5 training target format!) ---
# Assuming your model was trained on the simpler pipe format based on your app code,
# but we will make the parsing robust for either clean pipe or structured delimiters.
# We will use the pipe split for compatibility with the provided app logic.

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

# --- Helper Functions ---

@st.cache_resource
def load_model():
    """Loads the fine-tuned T5 model and tokenizer."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model directory not found at: {MODEL_PATH}")
        st.stop()
    
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
                num_beams=8, # Increased beams for higher quality/coherence (Crucial fix)
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

def generate_single_quiz_item(context_chunk, question_type, tokenizer, model, device):
    """Generates a QG item, forces a factual answer, and filters poor output."""
    if len(context_chunk.split()) < 40:
        return None, None, None
        
    sentences = sent_tokenize(context_chunk)
    if not sentences:
        return None, None, None
    
    # Use 3 random sentences to generate a central point for the question
    base_text = " ".join(random.sample(sentences, min(3, len(sentences))))
    
    try:
        if question_type == "QG":
            # --- Step 1: Generate Question (QG Task) ---
            qg_prefix = f"generate question: context: {base_text} answer: a key fact"
            generated_question = generate_output(qg_prefix, tokenizer, model, device, max_length=64, temperature=0.8)

            # --- HEURISTIC 1 (RELAXED): Only check for question mark and minimum length ---
            if "?" not in generated_question or len(generated_question.split()) < 3:
                 return None, None, None 

            # --- Step 2: Generate Factual Answer (QA Task) ---
            qa_prefix = f"question: {generated_question} context: {context_chunk}"
            generated_answer = generate_output(qa_prefix, tokenizer, model, device, max_length=QA_MAX_LENGTH, temperature=0.1)
            
            # --- HEURISTIC 2 (RELAXED): Answer length check is more forgiving ---
            cleaned_answer = clean_answer(generated_answer)
            # Increased max words from 10 to 15
            if not cleaned_answer or len(cleaned_answer.split()) > 15:
                 return None, None, None

            return "QG_QA", generated_question, cleaned_answer

        elif question_type == "MCQ":
            # NOTE: For MCQ, we rely entirely on the model to generate the full structured string
            input_prefix = f"generate mcq: context: {context_chunk}"
            result_raw = generate_output(input_prefix, tokenizer, model, device, max_length=MAX_OUTPUT_LENGTH, temperature=0.7)
            
            # --- HEURISTIC 3: Ensure MCQ format is parsable by pipes (minimum requirement) ---
            # If the model was trained strictly on [OPT] delimiters, this will still mostly work 
            # if the model uses pipes as fallback, but ideally should match the training output format.
            if "|" not in result_raw:
                 return None, None, None

            return "MCQ", "What is a key detail from this chunk?", result_raw
    
    except Exception as e:
        # st.warning(f"Error during quiz item generation: {e}") # Commented out for cleaner app log
        return None, None, None
        
    return None, None, None

# --- Main App Execution ---

# 1. Load model first (cached)
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
        # PHASE 1: GENERATE SUMMARY NOTES (CLEANED)
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
            
            # Process each chunk
            for chunk in summary_chunks:
                input_prefix = f"summarize: {chunk}"
                
                generated_summary_raw = generate_output(input_prefix, tokenizer, model, device, max_length=MAX_OUTPUT_LENGTH, temperature=0.9)
                
                # Robust summary parsing, falling back to raw output if delimiters fail
                if "| long:" in generated_summary_raw:
                    long_summary = generated_summary_raw.split("| long:")[-1].strip()
                elif "long:" in generated_summary_raw:
                    long_summary = generated_summary_raw.split("long:")[-1].strip()
                else:
                    # Fallback: just use the whole output
                    long_summary = generated_summary_raw
                
                # Post-process: Break into sentences and add unique ones
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
        # PHASE 2: GENERATE QUIZ (QG -> QA CHAINED & FILTERED)
        # ----------------------------------------------------
        if output_choice in ['Generate Both (Notes & Quiz)', 'Quiz Only']:
            status.update(label=f"Generating {num_questions} Quiz Questions (Phase 2/2): Applying Filters...", state="running")
            
            # Chunking and Sampling logic
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
            
            # Ensure we have enough chunks to sample from
            if len(quiz_chunks) > num_questions * 3: # Use a higher multiplier to increase success rate
                selected_chunks = random.sample(quiz_chunks, num_questions * 3)
            else:
                selected_chunks = quiz_chunks

            quiz_items = []
            
            # Generation Loop with Error Handling/Filtering
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
                
                elif item_type == "MCQ" and ("|" in result_raw or "options:" in result_raw):
                    try:
                        # Robust Parsing for MCQ (Handles both pipe and potential strict format fails)
                        
                        # 1. Split by the main option separator
                        parts = result_raw.split("|")
                        
                        # 2. Extract answer part (assumed to be the last part prefixed by 'answer:')
                        answer_part_raw = parts[-1]
                        answer_match = re.search(r'answer:\s*(.*)', answer_part_raw)
                        answer = clean_answer(answer_match.group(1).strip()) if answer_match else "N/A"
                        
                        # 3. Extract options (everything before the last 'answer:' part)
                        options_raw = result_raw.replace(answer_part_raw, "").replace("options:", "").strip()
                        
                        # Re-split/clean remaining options based on pipe or space
                        options_list = [clean_answer(opt) for opt in options_raw.split('|') if opt.strip()]
                        
                        # Final check for validity
                        if len(options_list) < 3 or answer == "N/A":
                            raise ValueError("Bad MCQ format detected after cleaning.")

                        quiz_items.append({
                            "type": "MCQ",
                            "question": f"Question {len(quiz_items) + 1}: Select the correct option. (Difficulty: {difficulty})",
                            "options": options_list,
                            "answer": answer
                        })
                    except Exception as e:
                        # st.warning(f"Skipping badly formatted MCQ item. Error: {e}")
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
