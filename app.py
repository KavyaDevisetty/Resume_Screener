import os
import re
import spacy
import nltk
from flask import Flask, render_template, request, send_from_directory
from collections import defaultdict
import base64
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import heapq
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


# Download NLTK data
nltk.download('stopwords')

# Load spaCy NLP model
nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])

def upload_resume():
    if 'resume' not in request.files:
        return "No file uploaded", 400

    file = request.files['resume']
    job_description = request.form.get("job_description", "")

    if file.filename == '':
        return "No selected file", 400

    if file and file.filename.endswith('.pdf'):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Extract text
        extracted_text = extract_text_from_pdf(filepath)

        # Check which option was selected
        option = request.form.get("option")

        if option == "text_dump":
            return render_template('textdump.html', text=extracted_text)

        elif option == "analysis":
            resume_data = analyze_resume(extracted_text)
            return render_template('results.html', data=resume_data)
        
        
        # elif option == "ranking":
        #     ats_score, match_score, keyword_freq, skills_score, experience_score, education_score, formatting_score = rank_resume(extracted_text, job_description)
            
        #     ats_score = max(1, min(100, ats_score))
        #     match_score = max(1, min(100, match_score)) if isinstance(match_score, float) else "N/A"
            
        #     ats_image, keyword_image = generate_graphs(
        #         ats_score, skills_score, experience_score, education_score, formatting_score, keyword_freq
        #     )

        #     return render_template('dashboard.html',
        #                            ats_score=ats_score,
        #                            match_score=match_score,
        #                            keyword_freq=keyword_freq,
        #                            ats_image=ats_image,
        #                            keyword_image=keyword_image,
        #                            job_description=job_description)

        
        elif option == "ranking":
            ats_score, match_score, keyword_freq, skills_score, experience_score, education_score, formatting_score = rank_resume(extracted_text, job_description)

            ats_score = max(1, min(100, ats_score))
            match_score = max(1, min(100, match_score)) if isinstance(match_score, float) else "N/A"

            # ✅ Generate Resume Insights
            insights = generate_resume_insights(extracted_text, job_description)

            ats_image, keyword_image = generate_graphs(
                ats_score, skills_score, experience_score, education_score, formatting_score, keyword_freq
                )
            
            pdf_filename = file.filename.replace(".pdf", "_report")
            pdf_path = generate_pdf_report(pdf_filename, ats_score, match_score, keyword_freq, insights)


            return render_template('dashboard.html',
                           ats_score=ats_score,
                           match_score=match_score,
                           keyword_freq=keyword_freq,
                           ats_image=ats_image,
                           keyword_image=keyword_image,
                           job_description=job_description,
                           insights=insights,
                           pdf_path=pdf_path)



    return "Invalid file format. Please upload a PDF.", 400


import fitz  # PyMuPDF

def extract_text_from_pdf(filepath):
    """Extract text from a PDF using PyMuPDF (Fitz)."""
    text = ""
    with fitz.open(filepath) as pdf:
        for page in pdf:
            text += page.get_text("text") + "\n"

    return text



def analyze_resume(text):
    """Extract key information from resume text."""
    print("\n🔍 Extracted Resume Text:\n", text[:500])  # Print first 500 characters

    data = {
        "name": extract_name(text),
        "email": extract_email(text),
        "phone": extract_phone(text),
        "skills": extract_skills(text),
    }
    print("\n🔍 Extracted Name:", data["name"])  # Debugging name extraction
    return data


def extract_name(text):
    """Extract name by checking the first few words."""
    text = text.strip()
    
    # ✅ Check the first 200 characters (useful for names at the top)
    first_part = text[:200]
    words = first_part.split()

    # Ignore common resume words
    ignore_words = {"resume", "curriculum", "vitae", "profile", "objective"}
    
    if any(word.lower() in ignore_words for word in words[:5]):
        words = words[5:]  # Skip header words

    # ✅ Find first 2+ capitalized words (Full Name)
    name_candidates = []
    for i in range(len(words) - 1):
        if words[i][0].isupper() and words[i + 1][0].isupper():
            name_candidates.append(words[i] + " " + words[i + 1])

    return name_candidates[0] if name_candidates else "Not found"



def extract_email(text):
    """Extract email using regex."""
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    emails = re.findall(email_pattern, text)
    return emails[0] if emails else "Not found"

def extract_phone(text):
    """Extract phone number using regex."""
    phone_pattern = r"\+?\d{1,3}[-.\s]?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}"
    phones = re.findall(phone_pattern, text)
    return phones[0] if phones else "Not found"


def extract_skills(text):
    """Extract skills using NLP-based keyword matching."""
    skills_db = [
        "Python", "Java", "C++", "Machine Learning", "Deep Learning",
        "Data Science", "Flask", "Django", "SQL", "TensorFlow", "Keras",
        "NLP", "Data Visualization", "Git", "AWS", "Docker", "Kubernetes",
        "Pandas", "NumPy", "Matplotlib", "Scikit-learn", "OpenCV"
    ]

    found_skills = []
    doc = nlp(text.lower())  # Convert to lowercase for better matching

    for token in doc:
        if token.text in [skill.lower() for skill in skills_db]:
            found_skills.append(token.text.capitalize())

    return ", ".join(set(found_skills)) if found_skills else "Not found"


class TrieNode:
    """Node for the Trie data structure."""
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    """Trie Data Structure for storing and searching words."""
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        """Insert a word into the Trie."""
        node = self.root
        for char in word.lower():  # Convert to lowercase for case-insensitive search
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def search(self, word):
        """Search for a word in the Trie."""
        node = self.root
        for char in word.lower():
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word


def rank_resume(resume_text, job_description=""):
    """Calculate ATS resume score, job match score, and other key metrics using Trie, HashMap, and Priority Queue."""
    
    resume_text = re.sub(r"[^a-zA-Z0-9\s]", "", resume_text.lower().strip())
    job_description = re.sub(r"[^a-zA-Z0-9\s]", "", job_description.lower().strip())

    resume_tokens = [token.text for token in nlp(resume_text) if not token.is_stop]
    job_tokens = [token.text for token in nlp(job_description) if not token.is_stop] if job_description else []

    # ✅ **Trie for Keyword Search**
    trie = Trie()
    weighted_keywords = {
        "python": 5, "java": 5, "c++": 4, "machine learning": 6, "deep learning": 5,
        "data science": 6, "flask": 4, "django": 4, "tensorflow": 5, "keras": 5,
        "sql": 4, "nlp": 5, "big data": 5, "aws": 4, "docker": 4, "kubernetes": 4
    }

    for word in weighted_keywords.keys():
        trie.insert(word)

    # ✅ **HashMap for Keyword Frequency**
    resume_word_freq = defaultdict(int)
    skills_score = 0

    for word in resume_tokens:
        if trie.search(word):
            resume_word_freq[word] += 1
            skills_score += weighted_keywords.get(word, 0)

    # ✅ **Priority Queue for Ranking Top Keywords**
    top_keywords = heapq.nlargest(5, resume_word_freq.items(), key=lambda x: x[1])
    top_keywords_dict = dict(top_keywords)

    # ✅ **Compute Other Scores**
    experience_score = sum(5 for word in resume_tokens if word in {"engineer", "developer", "analyst", "manager", "intern", "lead"})
    education_score = sum(4 for word in resume_tokens if word in {"bachelor", "master", "phd", "bsc", "msc", "mba", "university", "college"})
    formatting_score = 20 if len(resume_text.split("\n")) > 10 else 10  

    # ✅ **Job Match Score**
    if job_tokens:
        job_word_freq = defaultdict(int)
        for word in job_tokens:
            job_word_freq[word] += 1

        keyword_matches = sum(job_word_freq[word] for word in resume_tokens if word in job_word_freq)
        match_score = (keyword_matches / max(len(job_tokens), 1)) * 100
    else:
        match_score = "N/A"

    # ✅ **Final ATS Score**
    final_score = min((skills_score + experience_score + education_score + formatting_score) / 100 * 100, 100)

    return (
        round(final_score, 2),  
        round(match_score, 2) if isinstance(match_score, float) else match_score,
        top_keywords_dict, 
        skills_score, 
        experience_score, 
        education_score, 
        formatting_score
    )



def generate_graphs(ats_score, skills_score, experience_score, education_score, formatting_score, keyword_freq):
    """Generate ATS Score Breakdown and Keyword Frequency Graphs with predefined colors."""

    # ✅ Predefined Colors
    bar_color = "#2892A0"  # All bars cyan
    pie_colors = ["#add8e6", "#ffcccb", "#d3d3d3", "#ffffe0", "#e6e6fa"]  # Light colors

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns

    # 📊 **Bar Chart for ATS Score Breakdown**
    fig, ax = plt.subplots(figsize=(6, 4))
    components = ["Skills", "Experience", "Education", "Formatting"]
    scores = [skills_score, experience_score, education_score, formatting_score]
    ax.bar(components, scores, color=bar_color)
    ax.set_ylim(0, 100)
    ax.set_title("ATS Score Breakdown")

    ats_buffer = BytesIO()
    plt.savefig(ats_buffer, format="png")
    ats_buffer.seek(0)
    ats_image = base64.b64encode(ats_buffer.getvalue()).decode()
    plt.close()

    # 📊 **Pie Chart for Keyword Frequency**
    fig, ax = plt.subplots(figsize=(6, 4))
    if keyword_freq:
        ax.pie(keyword_freq.values(), labels=keyword_freq.keys(), autopct='%1.1f%%', colors=pie_colors)
        ax.set_title("Top Resume Keywords")

    keyword_buffer = BytesIO()
    plt.savefig(keyword_buffer, format="png")
    keyword_buffer.seek(0)
    keyword_image = base64.b64encode(keyword_buffer.getvalue()).decode()
    plt.close()

    return ats_image, keyword_image

@app.route('/dashboard', methods=['POST'])
def dashboard():
    resume_text = request.form['resume_text']
    job_description = request.form.get('job_description', "")

    ats_score, match_score, keyword_freq, skills_score, experience_score, education_score, formatting_score = rank_resume(resume_text, job_description)

    ats_image, keyword_image = generate_graphs(
        ats_score, match_score, skills_score, experience_score, education_score, formatting_score, job_description, keyword_freq
    )

    return render_template("dashboard.html",
                           ats_score=ats_score,
                           match_score=match_score,
                           ats_image=ats_image,
                           keyword_image=keyword_image,
                           job_description=job_description)


def generate_resume_insights(resume_text, job_description="", match_score=0):
    """Generate AI-powered resume optimization insights using spaCy and data structures."""

    insights = []
    pq = []  # Min-Heap for ranking insights
    
    # ✅ **Step 1: Process Resume Text Using spaCy**
    doc = nlp(resume_text)

    # ✅ **Step 2: Use Trie for Fast Keyword Search**
    trie = Trie()
    job_keywords = set(re.findall(r'\b\w+\b', job_description.lower())) if job_description else set()
    
    for keyword in job_keywords:
        trie.insert(keyword)

    # ✅ **Step 3: Use HashMap for Word Frequency Analysis**
    resume_words = re.findall(r'\b\w+\b', resume_text.lower())
    resume_word_freq = defaultdict(int)

    for word in resume_words:
        resume_word_freq[word] += 1

    # ✅ **Step 4: Identify Missing Job-Related Keywords**
    missing_keywords = [word for word in job_keywords if not trie.search(word)]
    if missing_keywords:
        heapq.heappush(pq, (1, f"Consider adding these important keywords: {', '.join(missing_keywords)}."))

    # ✅ **Step 5: Identify Weak Action Verbs Using POS Tagging**
    action_verbs = {"managed", "led", "developed", "optimized", "implemented", "engineered", "executed", "designed"}
    used_verbs = {token.text.lower() for token in doc if token.pos_ == "VERB" and token.text.lower() in action_verbs}

    if len(used_verbs) < 5:
        heapq.heappush(pq, (2, "Use strong action verbs like 'led', 'optimized', 'executed' to enhance your experience."))

    # ✅ **Step 6: Readability Analysis**
    sentence_lengths = [len(sent.text.split()) for sent in doc.sents]
    avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0

    if avg_sentence_length > 20:
        heapq.heappush(pq, (3, "Consider breaking down long sentences for better readability."))

    # ✅ **Step 7: Bullet Points Check**
    bullet_points = resume_text.count("•") + resume_text.count("- ")
    if bullet_points < 3:
        heapq.heappush(pq, (4, "Use bullet points to structure experience and skills more clearly."))

    # ✅ **Step 8: Passive Voice Detection**
    passive_voice_count = len(re.findall(r'\b(is|was|were|been|being)\s+\w+ed\b', resume_text.lower()))
    if passive_voice_count > 3:
        heapq.heappush(pq, (5, "Reduce passive voice usage. Use direct and confident language."))

    # ✅ **Retrieve Top 10 Insights from Priority Queue**
    while pq and len(insights) < 10:
        _, insight = heapq.heappop(pq)
        insights.append(insight)

    return insights

def generate_pdf_report(filename, ats_score, match_score, keyword_freq, insights):
    """Generate a PDF report summarizing resume analysis results."""
    
    pdf_path = f"reports/{filename}.pdf"  # Save PDF in 'reports/' folder

    # ✅ Create PDF file
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter

    # ✅ Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(200, height - 50, "Resume Analysis Report")

    # ✅ Scores Section
    c.setFont("Helvetica", 12)
    c.drawString(100, height - 100, f"ATS Resume Score: {ats_score}%")
    if match_score != "N/A":
        c.drawString(100, height - 120, f"Job Match Score: {match_score}%")
    else:
        c.drawString(100, height - 120, "No Job Description Provided")

    # ✅ Keyword Frequency Section
    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, height - 160, "Top Keywords Used:")
    c.setFont("Helvetica", 12)
    y = height - 180
    for keyword, freq in keyword_freq.items():
        c.drawString(120, y, f"- {keyword}: {freq} times")
        y -= 20

    # ✅ Resume Insights Section
    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, y - 20, "Resume Optimization Insights:")
    c.setFont("Helvetica", 12)
    y -= 40
    if insights and insights[0] != "No insights required.":
        for insight in insights:
            c.drawString(120, y, f"- {insight}")
            y -= 20
    else:
        c.drawString(120, y, "No insights required.")

    # ✅ Save PDF
    c.save()
    return pdf_path  # Return the file path for downloading

@app.route('/download/<filename>')
def download_report(filename):
    """Serve PDF reports for download."""
    return send_from_directory("reports", filename, as_attachment=True)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
