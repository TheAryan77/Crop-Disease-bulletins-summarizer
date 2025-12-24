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

def extract_key_information(text):
    """Extract key entities and information from text"""
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    
    info = {
        'crops': set(),
        'diseases': set(),
        'treatments': set(),
        'actions': set(),
        'numbers': []
    }
    
    # Crop types
    crop_types = {'rice', 'wheat', 'cotton', 'maize', 'sugarcane', 'soybean', 
                  'vegetable', 'fruit', 'potato', 'tomato', 'onion', 'corn'}
    
    # Disease related terms
    disease_terms = {'disease', 'pest', 'fungus', 'bacteria', 'virus', 'infection',
                     'blight', 'wilt', 'rot', 'mildew', 'rust', 'spot', 'aphid'}
    
    # Treatment terms
    treatment_terms = {'spray', 'apply', 'treatment', 'control', 'pesticide',
                       'fungicide', 'herbicide', 'insecticide', 'fertilizer'}
    
    # Action verbs
    action_verbs = {'apply', 'spray', 'use', 'control', 'prevent', 'treat',
                    'monitor', 'manage', 'avoid', 'ensure', 'maintain'}
    
    for word in words:
        if word in crop_types:
            info['crops'].add(word)
        if word in disease_terms:
            info['diseases'].add(word)
        if word in treatment_terms:
            info['treatments'].add(word)
        if word in action_verbs:
            info['actions'].add(word)
    
    # Extract numbers (doses, percentages, etc.)
    for sentence in sentences[:10]:  # Check first 10 sentences
        numbers = re.findall(r'\d+\.?\d*\s*(?:kg|g|ml|l|percent|%|ppm)', sentence.lower())
        info['numbers'].extend(numbers)
    
    return info

def generate_abstractive_summary(text, num_sentences=5):
    """Generate new sentences that summarize the bulletin"""
    if not text or len(text.strip()) < 50:
        return ["No sufficient content found in the PDF for summarization."]
    
    try:
        sentences = sent_tokenize(text)
        
        if len(sentences) == 0:
            return ["Unable to extract meaningful sentences from the document."]
        
        # Extract key information
        key_info = extract_key_information(text)
        keywords = extract_keywords(text, top_n=15)
        
        # Calculate word frequencies
        stop_words = set(stopwords.words("english"))
        words = word_tokenize(text.lower())
        word_freq = defaultdict(int)
        
        for word in words:
            if word not in stop_words and word.isalpha() and len(word) > 2:
                word_freq[word] += 1
        
        # Score and select important sentences
        sentence_scores = {}
        for sentence in sentences:
            if len(sentence.split()) < 4:
                continue
            score = calculate_sentence_score(sentence, word_freq, keywords)
            sentence_scores[sentence] = score
        
        ranked_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
        top_sentences = [sent for sent, score in ranked_sentences[:15]]
        
        # Generate new summary sentences
        summary = []
        
        # Sentence 1: Main topic identification
        crops_mentioned = ', '.join(list(key_info['crops'])[:3]) if key_info['crops'] else 'crops'
        diseases_mentioned = ', '.join(list(key_info['diseases'])[:3]) if key_info['diseases'] else 'agricultural issues'
        
        summary.append(f"This bulletin addresses {diseases_mentioned} affecting {crops_mentioned} cultivation.")
        
        # Sentence 2: Problem description
        problem_keywords = [k for k in keywords if k in AGRICULTURAL_KEYWORDS][:5]
        if problem_keywords:
            summary.append(f"The primary concerns include {', '.join(problem_keywords[:3])} which require immediate attention from farmers.")
        
        # Sentence 3: Treatment recommendations
        treatments = list(key_info['treatments'])
        if treatments:
            treatment_str = ', '.join(treatments[:3])
            summary.append(f"Recommended control measures involve the application of {treatment_str} at appropriate stages.")
        else:
            # Extract treatment info from high-scoring sentences
            for sent in top_sentences[:5]:
                if any(word in sent.lower() for word in ['spray', 'apply', 'treatment', 'control']):
                    # Simplify the sentence
                    words_in_sent = sent.split()
                    if len(words_in_sent) > 15:
                        summary.append(f"Control measures include {' '.join(words_in_sent[:15])}...")
                    else:
                        summary.append(sent)
                    break
        
        # Sentence 4: Dosage/Application details
        if key_info['numbers']:
            dose_info = key_info['numbers'][0]
            summary.append(f"Application rates of {dose_info} are recommended as per field conditions.")
        else:
            # Look for sentences with numbers
            for sent in top_sentences:
                if re.search(r'\d+', sent) and any(word in sent.lower() for word in ['dose', 'rate', 'kg', 'ml', 'gram', 'liter']):
                    words_in_sent = sent.split()
                    if len(words_in_sent) > 20:
                        summary.append(' '.join(words_in_sent[:20]) + '...')
                    else:
                        summary.append(sent)
                    break
        
        # Sentence 5: Prevention/Management
        for sent in top_sentences:
            if any(word in sent.lower() for word in ['prevent', 'monitor', 'manage', 'avoid', 'ensure']):
                words_in_sent = sent.split()
                if len(words_in_sent) > 20:
                    summary.append(f"Preventive measures suggest {' '.join(words_in_sent[:18])}...")
                else:
                    summary.append(sent)
                break
        
        # Sentence 6: Weather/Environmental conditions
        for sent in top_sentences:
            if any(word in sent.lower() for word in ['weather', 'temperature', 'rainfall', 'humidity', 'climate']):
                words_in_sent = sent.split()
                if len(words_in_sent) > 18:
                    summary.append(f"Environmental factors indicate {' '.join(words_in_sent[:16])}...")
                else:
                    summary.append(sent)
                break
        
        # Sentence 7: General recommendation
        actions = list(key_info['actions'])
        if actions and len(summary) < num_sentences:
            summary.append(f"Farmers are advised to {', '.join(actions[:3])} based on current field observations and expert recommendations.")
        
        # Ensure we have enough sentences
        if len(summary) < 3:
            # Fall back to extractive summarization
            return [sent.strip() for sent in top_sentences[:num_sentences]]
        
        return summary[:num_sentences]
    
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
            
            # Generate abstractive summary
            summary = generate_abstractive_summary(cleaned_text, num_sentences=7)
            
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
