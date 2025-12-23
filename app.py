from flask import Flask, render_template, request, flash
import fitz
import nltk
import os
import re
import ssl
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict, Counter
import string

# Fix SSL certificate issue for NLTK downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download NLTK data quietly (ignore if already exists)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

app = Flask(__name__)
app.secret_key = 'crop_disease_advisory_secret_key'
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'pdf'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Agricultural/crop disease related keywords for better relevance scoring
AGRICULTURAL_KEYWORDS = {
    'disease', 'crop', 'pest', 'fungus', 'bacteria', 'virus', 'infection',
    'treatment', 'spray', 'control', 'prevention', 'symptom', 'leaf', 'stem',
    'root', 'plant', 'soil', 'fertilizer', 'pesticide', 'fungicide', 'herbicide',
    'insecticide', 'yield', 'damage', 'affected', 'resistance', 'susceptible',
    'pathogen', 'infestation', 'management', 'recommendation', 'advisory',
    'weather', 'rainfall', 'temperature', 'humidity', 'irrigation', 'drought',
    'rice', 'wheat', 'cotton', 'maize', 'sugarcane', 'soybean', 'vegetable',
    'fruit', 'apply', 'application', 'dose', 'stage', 'growth', 'field'
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(path):
    """Extract text from PDF with better formatting preservation"""
    try:
        doc = fitz.open(path)
        text = ""
        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            text += f"\n{page_text}\n"
        doc.close()
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def preprocess(text):
    """Clean text while preserving important agricultural terms"""
    # Remove excessive whitespace but preserve sentence structure
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep periods, commas, and hyphens
    text = re.sub(r'[^\w\s\.\,\-\:\;]', '', text)
    
    # Remove page numbers and common PDF artifacts
    text = re.sub(r'\bPage\s+\d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d+\s+of\s+\d+\b', '', text, flags=re.IGNORECASE)
    
    return text.strip()

def extract_keywords(text, top_n=10):
    """Extract important keywords from the bulletin"""
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    
    # Filter out stopwords and short words
    filtered_words = [
        word for word in words 
        if word not in stop_words 
        and len(word) > 3 
        and word.isalpha()
    ]
    
    # Count frequency
    word_freq = Counter(filtered_words)
    
    # Boost agricultural keywords
    for word in filtered_words:
        if word in AGRICULTURAL_KEYWORDS:
            word_freq[word] *= 2
    
    return [word for word, _ in word_freq.most_common(top_n)]

def calculate_sentence_score(sentence, word_freq, keywords):
    """Calculate relevance score for a sentence"""
    words = word_tokenize(sentence.lower())
    score = 0
    
    # Base score from word frequency
    for word in words:
        if word in word_freq:
            score += word_freq[word]
    
    # Bonus for agricultural keywords
    for word in words:
        if word in AGRICULTURAL_KEYWORDS:
            score += 5
    
    # Bonus for extracted keywords
    for word in words:
        if word in keywords:
            score += 3
    
    # Bonus for sentences with numbers (often indicate measurements, doses, etc.)
    if re.search(r'\d+', sentence):
        score += 2
    
    # Penalty for very short sentences
    if len(words) < 5:
        score *= 0.5
    
    # Penalty for very long sentences
    if len(words) > 40:
        score *= 0.8
    
    return score

def summarize(text, num_sentences=7):
    """Advanced summarization for crop disease bulletins"""
    if not text or len(text.strip()) < 50:
        return ["No sufficient content found in the PDF for summarization."]
    
    try:
        # Tokenize into sentences
        sentences = sent_tokenize(text)
        
        if len(sentences) == 0:
            return ["Unable to extract meaningful sentences from the document."]
        
        # If document is very short, return all sentences
        if len(sentences) <= num_sentences:
            return sentences
        
        # Extract keywords
        keywords = extract_keywords(text)
        
        # Calculate word frequencies (excluding stopwords)
        stop_words = set(stopwords.words("english"))
        words = word_tokenize(text.lower())
        
        word_freq = defaultdict(int)
        for word in words:
            if word not in stop_words and word.isalpha() and len(word) > 2:
                word_freq[word] += 1
        
        # Score each sentence
        sentence_scores = {}
        for sentence in sentences:
            # Skip very short sentences
            if len(sentence.split()) < 4:
                continue
            score = calculate_sentence_score(sentence, word_freq, keywords)
            sentence_scores[sentence] = score
        
        # Rank sentences by score
        ranked_sentences = sorted(
            sentence_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Get top N sentences
        top_sentences = [sent for sent, score in ranked_sentences[:num_sentences]]
        
        # Reorder by appearance in original text for better flow
        ordered_summary = []
        for sentence in sentences:
            if sentence in top_sentences:
                ordered_summary.append(sentence.strip())
        
        return ordered_summary if ordered_summary else sentences[:num_sentences]
    
    except Exception as e:
        print(f"Error during summarization: {e}")
        return ["Error occurred during summarization. Please try a different document."]

@app.route("/", methods=["GET", "POST"])
def index():
    summary = []
    keywords = []
    stats = {}
    
    if request.method == "POST":
        # Check if file was uploaded
        if 'pdf' not in request.files:
            flash('No file uploaded', 'error')
            return render_template("index.html", summary=summary, keywords=keywords, stats=stats)
        
        file = request.files["pdf"]
        
        # Check if filename is empty
        if file.filename == '':
            flash('No file selected', 'error')
            return render_template("index.html", summary=summary, keywords=keywords, stats=stats)
        
        # Check if file is allowed
        if not allowed_file(file.filename):
            flash('Please upload a PDF file', 'error')
            return render_template("index.html", summary=summary, keywords=keywords, stats=stats)
        
        try:
            # Save the file
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)
            
            # Extract and process text
            raw_text = extract_text_from_pdf(path)
            
            if not raw_text or len(raw_text.strip()) < 50:
                flash('Could not extract sufficient text from PDF. Please check if the PDF contains readable text.', 'error')
                return render_template("index.html", summary=summary, keywords=keywords, stats=stats)
            
            cleaned_text = preprocess(raw_text)
            
            # Generate summary
            summary = summarize(cleaned_text, num_sentences=7)
            
            # Extract keywords
            keywords = extract_keywords(cleaned_text, top_n=8)
            
            # Calculate statistics
            sentences = sent_tokenize(cleaned_text)
            words = word_tokenize(cleaned_text)
            stats = {
                'total_sentences': len(sentences),
                'total_words': len(words),
                'summary_sentences': len(summary),
                'compression_ratio': f"{(len(summary) / len(sentences) * 100):.1f}%" if len(sentences) > 0 else "N/A"
            }
            
            flash('Summary generated successfully!', 'success')
            
            # Clean up uploaded file
            try:
                os.remove(path)
            except:
                pass
                
        except Exception as e:
            flash(f'Error processing PDF: {str(e)}', 'error')
            print(f"Error: {e}")
    
    return render_template("index.html", summary=summary, keywords=keywords, stats=stats)

if __name__ == "__main__":
    app.run(debug=True)
