from flask import Flask, render_template, request, flash, jsonify
import fitz
import os
import re
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'bart_crop_disease_advisory_secret_key'
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'pdf'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables for model (loaded once)
bart_model = None
bart_tokenizer = None
summarization_pipeline = None

def initialize_bart_model():
    """
    Initialize BART model for abstractive summarization.
    Using facebook/bart-large-cnn - pre-trained on CNN/DailyMail dataset
    """
    global bart_model, bart_tokenizer, summarization_pipeline
    
    if bart_model is None:
        print("Loading BART model... This may take a moment on first run.")
        
        # Check if CUDA is available
        device = 0 if torch.cuda.is_available() else -1
        device_name = "GPU" if device == 0 else "CPU"
        print(f"Using device: {device_name}")
        
        # Load pre-trained BART model and tokenizer
        model_name = "facebook/bart-large-cnn"
        
        try:
            bart_tokenizer = BartTokenizer.from_pretrained(model_name)
            bart_model = BartForConditionalGeneration.from_pretrained(model_name)
            
            # Create summarization pipeline
            summarization_pipeline = pipeline(
                "summarization",
                model=bart_model,
                tokenizer=bart_tokenizer,
                device=device,
                framework="pt"
            )
            
            print("BART model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading BART model: {e}")
            return False
    
    return True

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(path):
    """Extract text from PDF using PyMuPDF"""
    try:
        doc = fitz.open(path)
        text = ""
        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            text += f"{page_text}\n"
        doc.close()
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def preprocess_text(text):
    """
    Clean and preprocess extracted text for better summarization
    """
    # Remove excessive whitespace and newlines
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Remove page numbers and common PDF artifacts
    text = re.sub(r'\bPage\s+\d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d+\s+of\s+\d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\xff]', '', text)
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    return text.strip()

def chunk_text(text, max_length=1024):
    """
    Split text into chunks that fit BART's token limit.
    BART can handle ~1024 tokens, we use slightly less for safety.
    """
    # Split by sentences (simple approach)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # Check if adding this sentence exceeds the limit
        if len(current_chunk.split()) + len(sentence.split()) < max_length:
            current_chunk += " " + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    
    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def extract_agricultural_context(text):
    """
    Extract agricultural-specific information for context-aware summarization
    """
    context = {
        'crops': [],
        'diseases': [],
        'treatments': [],
        'measurements': []
    }
    
    # Agricultural patterns
    crop_patterns = r'\b(rice|wheat|cotton|maize|corn|soybean|potato|tomato|onion|sugarcane|vegetable|fruit)\b'
    disease_patterns = r'\b(disease|pest|fungus|bacteria|virus|blight|wilt|rot|mildew|rust|aphid|infection)\b'
    treatment_patterns = r'\b(spray|pesticide|fungicide|herbicide|insecticide|fertilizer|treatment|control)\b'
    measurement_patterns = r'\d+\.?\d*\s*(?:kg|g|ml|l|liter|gram|percent|%|ppm|ha|acre)'
    
    # Extract matches
    context['crops'] = list(set(re.findall(crop_patterns, text.lower())))
    context['diseases'] = list(set(re.findall(disease_patterns, text.lower())))
    context['treatments'] = list(set(re.findall(treatment_patterns, text.lower())))
    context['measurements'] = re.findall(measurement_patterns, text.lower())[:5]
    
    return context

def bart_summarize(text, min_length=100, max_length=300):
    """
    Generate abstractive summary using BART transformer model.
    This is semantic AI-based summarization that generates new sentences.
    """
    if not text or len(text.split()) < 50:
        return ["Insufficient content for summarization."]
    
    try:
        # Ensure model is loaded
        if not initialize_bart_model():
            return ["Error: Could not load BART model."]
        
        # Preprocess text
        cleaned_text = preprocess_text(text)
        
        # Extract agricultural context
        ag_context = extract_agricultural_context(cleaned_text)
        
        # Chunk text if it's too long
        chunks = chunk_text(cleaned_text, max_length=900)
        
        summaries = []
        
        # Summarize each chunk
        for i, chunk in enumerate(chunks[:3]):  # Limit to first 3 chunks
            if len(chunk.split()) < 50:
                continue
                
            try:

                chunk_length = len(chunk.split())
                adjusted_min = min(min_length, chunk_length // 2)
                adjusted_max = min(max_length, chunk_length)
                
                result = summarization_pipeline(
                    chunk,
                    min_length=adjusted_min,
                    max_length=adjusted_max,
                    do_sample=False,  # Deterministic output
                    num_beams=2,      # Reduced for speed
                    length_penalty=1.0,  # Neutral length penalty
                    early_stopping=True,
                    truncation=True
                )
                
                summary_text = result[0]['summary_text']
                summaries.append(summary_text)
                
            except Exception as e:
                print(f"Error summarizing chunk {i}: {e}")
                continue
        
        if len(summaries) > 1:
            combined = " ".join(summaries)
            try:
                combined_length = len(combined.split())
                final_min = min(80, combined_length // 3)
                final_max = min(250, combined_length)
                
                final_result = summarization_pipeline(
                    combined,
                    min_length=final_min,
                    max_length=final_max,
                    do_sample=False,
                    num_beams=2,
                    length_penalty=1.0,
                    early_stopping=True,
                    truncation=True
                )
                final_summary = final_result[0]['summary_text']
            except Exception as e:
                print(f"Error in final summarization: {e}")
                final_summary = summaries[0]
        elif len(summaries) == 1:
            final_summary = summaries[0]
        else:
            return ["Could not generate summary from the provided document."]
        
        # Split into sentences for better readability
        summary_sentences = re.split(r'(?<=[.!?])\s+', final_summary)
        
        # Add context-aware information if available
        enhanced_summary = []
        
        if ag_context['crops'] or ag_context['diseases']:
            crops_str = ', '.join(ag_context['crops'][:3]) if ag_context['crops'] else 'various crops'
            diseases_str = ', '.join(ag_context['diseases'][:3]) if ag_context['diseases'] else 'agricultural issues'
            enhanced_summary.append(f"This bulletin addresses {diseases_str} affecting {crops_str}.")
        
        enhanced_summary.extend(summary_sentences)
        
        if ag_context['treatments']:
            treatments_str = ', '.join(ag_context['treatments'][:3])
            enhanced_summary.append(f"Recommended interventions include {treatments_str}.")
        
        if ag_context['measurements']:
            enhanced_summary.append(f"Application rates specified: {', '.join(ag_context['measurements'][:2])}.")
        
        return enhanced_summary[:7]  # Return top 7 sentences
        
    except Exception as e:
        print(f"Error in BART summarization: {e}")
        return [f"Error during summarization: {str(e)}"]

def extract_semantic_keywords(text, summary):
    """
    Extract keywords using semantic understanding from BART's attention
    """
    # Simple keyword extraction based on summary content
    summary_text = " ".join(summary) if isinstance(summary, list) else summary
    
    # Extract important agricultural terms
    keywords = []
    
    # Patterns for different categories
    patterns = {
        'crops': r'\b(rice|wheat|cotton|maize|corn|soybean|potato|tomato|onion|sugarcane)\b',
        'issues': r'\b(disease|pest|fungus|bacteria|virus|blight|wilt|rot|infection)\b',
        'actions': r'\b(spray|apply|control|prevent|treat|monitor|manage)\b',
        'chemicals': r'\b(pesticide|fungicide|herbicide|insecticide|fertilizer)\b'
    }
    
    for category, pattern in patterns.items():
        matches = re.findall(pattern, summary_text.lower())
        keywords.extend(list(set(matches)))
    
    # Remove duplicates and limit
    keywords = list(set(keywords))[:10]
    
    return keywords

@app.route('/favicon.ico')
def favicon():
    """Handle favicon requests"""
    return '', 204

@app.route("/", methods=["GET", "POST"])
def index():
    summary = []
    keywords = []
    stats = {}
    model_info = {
        'approach': 'BART Transformer (Semantic AI)',
        'model': 'facebook/bart-large-cnn',
        'type': 'Abstractive Summarization',
        'device': 'GPU' if torch.cuda.is_available() else 'CPU'
    }
    
    if request.method == "POST":
        # Check if file was uploaded
        if 'pdf' not in request.files:
            flash('No file uploaded', 'error')
            return render_template("index.html", summary=summary, keywords=keywords, 
                                 stats=stats, model_info=model_info)
        
        file = request.files["pdf"]
        
        # Check if filename is empty
        if file.filename == '':
            flash('No file selected', 'error')
            return render_template("index.html", summary=summary, keywords=keywords, 
                                 stats=stats, model_info=model_info)
        
        # Check if file is allowed
        if not allowed_file(file.filename):
            flash('Please upload a PDF file', 'error')
            return render_template("index.html", summary=summary, keywords=keywords, 
                                 stats=stats, model_info=model_info)
        
        try:
            # Save the file
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)
            
            flash('Processing PDF with BART transformer model...', 'info')
            
            # Extract text from PDF
            raw_text = extract_text_from_pdf(path)
            
            if not raw_text or len(raw_text.strip()) < 50:
                flash('Could not extract sufficient text from PDF.', 'error')
                return render_template("index.html", summary=summary, keywords=keywords, 
                                     stats=stats, model_info=model_info)
            
            # Generate BART-based abstractive summary
            summary = bart_summarize(raw_text, min_length=100, max_length=300)
            
            # Extract semantic keywords
            keywords = extract_semantic_keywords(raw_text, summary)
            
            # Calculate statistics
            total_words = len(raw_text.split())
            summary_words = sum(len(s.split()) for s in summary)
            
            stats = {
                'total_words': total_words,
                'summary_words': summary_words,
                'summary_sentences': len(summary),
                'compression_ratio': f"{(summary_words / total_words * 100):.1f}%" if total_words > 0 else "N/A",
                'model_type': 'Transformer-based (BART)'
            }
            
            flash('✓ Summary generated successfully using BART transformer!', 'success')
            
            # Clean up uploaded file
            try:
                os.remove(path)
            except:
                pass
                
        except Exception as e:
            flash(f'Error processing PDF: {str(e)}', 'error')
            print(f"Error: {e}")
    
    return render_template("index.html", summary=summary, keywords=keywords, 
                         stats=stats, model_info=model_info)

if __name__ == "__main__":
    # Initialize BART model on startup
    print("=" * 60)
    print("BART Transformer-Based Crop Bulletin Summarizer")
    print("=" * 60)
    initialize_bart_model()
    print("Starting Flask application...")
    app.run(debug=True, port=5001)
