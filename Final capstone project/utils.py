import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from pdfminer.high_level import extract_text as extract_text_pdf
import docx
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import google.generativeai as genai

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')

# Skills Database (Enhanced for better matching)
ROLE_SKILLS = {
    "Data Scientist": ["python", "machine learning", "statistics", "data visualization", "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "sql", "r", "jupyter", "matplotlib", "seaborn", "data mining"],
    "Software Engineer": ["python", "java", "c++", "c#", "data structures", "algorithms", "git", "docker", "kubernetes", "aws", "rest api", "microservices", "unit testing", "linux", "jenkins", "azure"],
    "Web Developer": ["html", "css", "javascript", "react", "angular", "vue", "node.js", "express", "mongodb", "sql", "bootstrap", "typescript", "sass", "webpack", "jquery", "frontend", "backend", "fullstack", "next.js", "tailwind"],
    "Data Analyst": ["excel", "sql", "tableau", "power bi", "statistics", "data cleaning", "pandas", "python", "data visualization", "bi tools", "data reporting", "analytical skills"],
    "Business Analyst": ["requirement analysis", "business process", "sql", "excel", "agile", "scrum", "jira", "communication", "stakeholder management", "use cases", "user stories", "documentation"],
    "Product Manager": ["product strategy", "roadmap", "agile", "scrum", "market research", "user experience", "analytics", "leadership", "product lifecycle", "prioritization", "wireframing"],
    "Machine Learning Engineer": ["python", "tensorflow", "pytorch", "deep learning", "nlp", "computer vision", "scikit-learn", "mlops", "data engineering", "keras", "neural networks", "model deployment"],
    "DevOps Engineer": ["docker", "kubernetes", "jenkins", "terraform", "ansible", "aws", "azure", "gcp", "ci/cd", "monitoring", "linux", "bash", "prometheus", "grafana", "gitops"],
    "UI/UX Designer": ["figma", "adobe xd", "sketch", "user research", "wireframing", "prototyping", "user interface", "user experience", "design system", "visual design", "usability testing"],
    "Cybersecurity Analyst": ["network security", "penetration testing", "siem", "firewall", "vulnerability assessment", "cryptography", "incident response", "ethical hacking", "soc", "compliance"],
    "Cloud Architect": ["aws", "azure", "gcp", "cloud infrastructure", "serverless", "iam", "networking", "cloud migration", "microservices", "disaster recovery", "scalability"],
    "Mobile Developer": ["swift", "kotlin", "java", "react native", "flutter", "objective-c", "ios", "android", "mobile app", "xcode", "android studio", "firebase"]
}

def extract_text_from_pdf(file_path):
    return extract_text_pdf(file_path)

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

def get_extracted_skills(text):
    all_skills = [skill for skills in ROLE_SKILLS.values() for skill in skills]
    all_skills = list(set(all_skills))
    found_skills = []
    text_lower = text.lower()
    for skill in all_skills:
        if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
            found_skills.append(skill)
    return list(set(found_skills))

def detect_sections(text):
    sections = {
        "Education": ["education", "academic", "university", "college", "degree"],
        "Experience": ["experience", "work history", "employment", "professional", "internship"],
        "Skills": ["skills", "technical skills", "competencies", "technologies"],
        "Projects": ["projects", "personal projects", "key projects", "academic projects"],
        "Certifications": ["certifications", "certification", "credentials", "licenses", "certificates"]
    }
    found_sections = []
    text_lower = text.lower()
    for section, keywords in sections.items():
        for keyword in keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                found_sections.append(section)
                break
    return found_sections

def calculate_ats_score(text, role_skills):
    found_skills = get_extracted_skills(text)
    
    # 1. Skills Match (30%)
    matching_skills = [s for s in found_skills if s in role_skills]
    skill_score = (len(matching_skills) / max(len(role_skills), 1)) * 30
    
    # 2. Projects (25%)
    found_sections = detect_sections(text)
    has_project_section = "Projects" in found_sections
    project_keywords = ['project', 'projects', 'github', 'github.com', 'git', 'deployed', 'built', 'implemented', 'developer', 'application', 'link']
    project_keyword_matches = sum(1 for k in project_keywords if k in text.lower())
    projects_score = 0
    if has_project_section:
        projects_score += 15
    projects_score += min(project_keyword_matches * 2.0, 10.0)
    
    # 3. Experience (20%)
    has_exp_section = "Experience" in found_sections
    exp_keywords = ['experience', 'work', 'job', 'intern', 'internship', 'developer', 'engineer', 'lead', 'manager', 'years', 'duration', 'employer']
    exp_keyword_matches = sum(1 for k in exp_keywords if k in text.lower())
    experience_score = 0
    if has_exp_section:
        experience_score += 10
    experience_score += min(exp_keyword_matches * 1.5, 10.0)
    
    # 4. Certifications (15%)
    has_cert_section = "Certifications" in found_sections
    cert_keywords = ['certification', 'certifications', 'certified', 'credential', 'udemy', 'coursera', 'certificate', 'training', 'nptel', 'aws certified', 'microsoft certified', 'google certified']
    cert_keyword_matches = sum(1 for k in cert_keywords if k in text.lower())
    certifications_score = 0
    if has_cert_section:
        certifications_score += 7
    certifications_score += min(cert_keyword_matches * 2.0, 8.0)
    
    # 5. Structure (10%)
    all_possible_sections = ["Education", "Experience", "Skills", "Projects", "Certifications"]
    structure_score = (len(found_sections) / len(all_possible_sections)) * 10
    
    total_score = skill_score + projects_score + experience_score + certifications_score + structure_score
    
    # Keyword density and stats
    vectorizer = TfidfVectorizer(stop_words='english', min_df=1, ngram_range=(1, 1), 
                                 token_pattern=r'\b[a-zA-Z]{3,}\b')
    try:
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]
        
        top_indices = scores.argsort()[::-1]
        top_keywords = []
        top_scores = []
        for i in top_indices:
            word = feature_names[i]
            score = float(scores[i])
            if len(word) > 2 and word not in ['and', 'com', 'the', 'for', 'with', 'from']:
                top_keywords.append(word)
                top_scores.append(round(score, 4))
            if len(top_keywords) >= 10:
                break
    except:
        top_keywords = []
        top_scores = []
        
    return {
        "total": round(total_score, 2),
        "skills_match": round(skill_score, 2),
        "projects": round(projects_score, 2),
        "experience": round(experience_score, 2),
        "certifications": round(certifications_score, 2),
        "structure": round(structure_score, 2),
        "found_skills": found_skills,
        "matching_skills": matching_skills,
        "missing_skills": [s for s in role_skills if s not in found_skills],
        "top_keywords": top_keywords,
        "keyword_scores": top_scores,
        "found_sections": found_sections,
        "missing_sections": [s for s in all_possible_sections if s not in found_sections]
    }

def get_company_readiness(found_skills):
    COMPANY_SKILLS = {
        "TCS": ["python", "java", "sql", "communication", "excel", "git"],
        "Infosys": ["javascript", "python", "sql", "agile", "documentation", "html"],
        "Accenture": ["cloud infrastructure", "agile", "scrum", "sql", "communication", "rest api"],
        "Microsoft": ["c++", "c#", "data structures", "algorithms", "azure", "docker", "kubernetes"],
        "Google": ["python", "java", "c++", "algorithms", "data structures", "system design", "git", "linux"]
    }
    
    found_skills_lower = [s.lower() for s in found_skills]
    readiness = {}
    for company, reqs in COMPANY_SKILLS.items():
        matched = [r for r in reqs if r.lower() in found_skills_lower]
        missing = [r for r in reqs if r.lower() not in found_skills_lower]
        score = round((len(matched) / len(reqs)) * 100)
        readiness[company] = {
            "score": score,
            "matched": matched,
            "missing": missing
        }
    return readiness

def get_career_roadmap(predicted_role):
    if "Data Scientist" in predicted_role or "Machine Learning" in predicted_role:
        return [
            {"month": "Month 1", "topic": "Python & Statistics", "details": "Learn Python basics, OOP, linear algebra, probability, and hypothesis testing."},
            {"month": "Month 2", "topic": "Data Preprocessing & Visualization", "details": "Master Pandas, NumPy, Matplotlib, Seaborn, and SQL databases."},
            {"month": "Month 3", "topic": "Classical Machine Learning", "details": "Study supervised and unsupervised learning algorithms using Scikit-Learn."},
            {"month": "Month 4", "topic": "Advanced ML & Power BI", "details": "Dive into feature engineering, model tuning, and building dashboards in Power BI/Tableau."},
            {"month": "Month 5", "topic": "End-to-End Capstone Projects", "details": "Build and deploy 2 ML projects on GitHub or cloud platforms (AWS/Heroku)."},
            {"month": "Month 6", "topic": "Interview Prep & Portfolio", "details": "Solve SQL/Python coding challenges, mock interviews, and optimize your portfolio/LinkedIn."}
        ]
    elif "Web Developer" in predicted_role or "Frontend" in predicted_role or "Backend" in predicted_role:
        return [
            {"month": "Month 1", "topic": "HTML, CSS & Tailwind CSS", "details": "Learn semantic HTML, modern layout techniques (Flexbox, Grid), and CSS preprocessors."},
            {"month": "Month 2", "topic": "JavaScript & DOM Manipulation", "details": "Master ES6+ syntax, asynchronous JS, APIs fetching, and JSON manipulation."},
            {"month": "Month 3", "topic": "Frontend Framework (React/Vue/Next.js)", "details": "Build single page applications, state management, and component architectural patterns."},
            {"month": "Month 4", "topic": "Backend Development (Node.js/SQL)", "details": "Create REST APIs, handle routing, databases connection, authentication and authorization."},
            {"month": "Month 5", "topic": "Full-Stack Project & Deployment", "details": "Develop a full-stack project, connect frontend/backend, deploy on Vercel/Render, and use Git."},
            {"month": "Month 6", "topic": "System Design & Interview Prep", "details": "Study web security, caching, load balancing, performance optimization, and practice coding."}
        ]
    else:
        return [
            {"month": "Month 1", "topic": "Core Fundamentals", "details": "Learn core programming languages, command line, version control (Git/GitHub)."},
            {"month": "Month 2", "topic": "Data Structures & Algorithms", "details": "Understand basic arrays, lists, queues, stacks, trees, and searching/sorting algorithms."},
            {"month": "Month 3", "topic": "Frameworks & Databases", "details": "Pick a primary framework for your target role and master database querying (SQL/NoSQL)."},
            {"month": "Month 4", "topic": "API Design & Cloud Basics", "details": "Learn how components communicate (REST/GraphQL), and understand AWS/Azure cloud basics."},
            {"month": "Month 5", "topic": "Portfolio Projects Development", "details": "Build 2 distinct portfolio projects highlighting key architectural decisions."},
            {"month": "Month 6", "topic": "Mock Interviews & Applications", "details": "Practice behavioral & coding questions, polish resume/LinkedIn, and start applying."}
        ]

def get_section_analysis(text, found_skills):
    found_sections = detect_sections(text)
    analysis = {}
    
    # 1. Skills
    skills_count = len(found_skills)
    if skills_count >= 8:
        analysis["Skills"] = {"status": "Excellent", "text": f"Found {skills_count} matching technical skills, showing strong competence."}
    elif skills_count >= 4:
        analysis["Skills"] = {"status": "Good", "text": f"Found {skills_count} skills. Adding more role-specific technologies can boost visibility."}
    else:
        analysis["Skills"] = {"status": "Needs Improvement", "text": "Low skill diversity. List more relevant technical libraries and frameworks."}
        
    # 2. Projects
    has_project_section = "Projects" in found_sections
    project_keywords = ['project', 'projects', 'github', 'github.com', 'git', 'deployed', 'built', 'implemented']
    project_keyword_matches = sum(1 for k in project_keywords if k in text.lower())
    if has_project_section and project_keyword_matches >= 4:
        analysis["Projects"] = {"status": "Excellent", "text": "Detailed project description found with standard technical verbs and tools."}
    elif has_project_section or project_keyword_matches >= 2:
        analysis["Projects"] = {"status": "Good", "text": "Projects section is present but details could be richer. Try adding links and metrics."}
    else:
        analysis["Projects"] = {"status": "Needs Improvement", "text": "No projects section or references found. Add hands-on projects to demonstrate your skills."}
        
    # 3. Certifications
    has_cert_section = "Certifications" in found_sections
    cert_keywords = ['certification', 'certifications', 'certified', 'credential', 'udemy', 'coursera', 'certificate']
    cert_keyword_matches = sum(1 for k in cert_keywords if k in text.lower())
    if has_cert_section and cert_keyword_matches >= 2:
        analysis["Certifications"] = {"status": "Excellent", "text": "Multiple certifications and training credentials detected."}
    elif has_cert_section or cert_keyword_matches >= 1:
        analysis["Certifications"] = {"status": "Good", "text": "Certification keywords found. Standardizing them in a dedicated section is recommended."}
    else:
        analysis["Certifications"] = {"status": "Needs Improvement", "text": "No certifications detected. Adding industry certifications (e.g. AWS, Coursera) will validate your skills."}
        
    # 4. Education
    has_edu_section = "Education" in found_sections
    edu_keywords = ["degree", "bachelor", "master", "phd", "university", "college", "btech", "mtech"]
    edu_keyword_matches = sum(1 for k in edu_keywords if k in text.lower())
    if has_edu_section and edu_keyword_matches >= 2:
        analysis["Education"] = {"status": "Excellent", "text": "Education section present with university details and academic degree description."}
    elif has_edu_section:
        analysis["Education"] = {"status": "Good", "text": "Education section present but could benefit from listing relevant coursework."}
    else:
        analysis["Education"] = {"status": "Needs Improvement", "text": "Education details are missing or unrecognized. Ensure degree and university are listed."}
        
    # 5. Achievements
    ach_keywords = ["award", "achieved", "won", "scholarship", "hackathon", "placed", "rank", "selected", "achievement", "achievements"]
    ach_matches = sum(1 for k in ach_keywords if k in text.lower())
    if ach_matches >= 3:
        analysis["Achievements"] = {"status": "Excellent", "text": "Strong track record of accomplishments and achievements identified."}
    elif ach_matches >= 1:
        analysis["Achievements"] = {"status": "Good", "text": "Some accomplishments noted, but could be highlighted in a dedicated section."}
    else:
        analysis["Achievements"] = {"status": "Needs Improvement", "text": "No major achievement keywords detected. Add awards, competitions, or honors."}
        
    return analysis

def get_resume_improvements(ats_score, ats_breakdown):
    suggestions = []
    potential_score = ats_score
    
    if ats_breakdown["skills_match"] < 25:
        points_gain = 8
        suggestions.append({
            "text": "Add missing key technical skills for the target role",
            "points": f"+{points_gain}"
        })
        potential_score += points_gain
        
    if ats_breakdown["projects"] < 20:
        points_gain = 7
        suggestions.append({
            "text": "Add a detailed project with GitHub repository link",
            "points": f"+{points_gain}"
        })
        potential_score += points_gain
        
    if ats_breakdown["experience"] < 15:
        points_gain = 6
        suggestions.append({
            "text": "Detail your professional experience or internships using action verbs",
            "points": f"+{points_gain}"
        })
        potential_score += points_gain
        
    if ats_breakdown["certifications"] < 10:
        points_gain = 5
        suggestions.append({
            "text": "Add relevant professional certifications (AWS, GCP, Coursera, etc.)",
            "points": f"+{points_gain}"
        })
        potential_score += points_gain
        
    if ats_breakdown["structure"] < 9:
        points_gain = 4
        suggestions.append({
            "text": "Format your resume to clearly label all major sections (Education, Experience, Skills, Projects, Certifications)",
            "points": f"+{points_gain}"
        })
        potential_score += points_gain
        
    return {
        "suggestions": suggestions,
        "potential_score": min(round(potential_score, 2), 98)
    }

def get_job_matching(resume_text, job_desc, role_skills):
    if not job_desc:
        return {
            "score": 0,
            "label": "N/A",
            "matching_skills": [],
            "missing_keywords": [],
            "compatibility": "N/A"
        }
    
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([resume_text, job_desc])
    similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    percentage = round(similarity * 100, 2)
    
    if percentage > 75: 
        label = "Good"
        compatibility = "Compatible"
    elif percentage > 50: 
        label = "Moderate"
        compatibility = "Semi-Compatible"
    else: 
        label = "Low"
        compatibility = "Incompatible"
        
    found_skills = get_extracted_skills(resume_text)
    job_skills = get_extracted_skills(job_desc)
    
    if not job_skills:
        job_skills = [s for s in role_skills if s in job_desc.lower()]
    if not job_skills:
        job_skills = role_skills[:6]
        
    matching_skills = [s for s in found_skills if s in job_skills]
    missing_keywords = [s for s in job_skills if s not in found_skills]
    
    return {
        "score": percentage,
        "label": label,
        "matching_skills": list(set(matching_skills)),
        "missing_keywords": list(set(missing_keywords)),
        "compatibility": compatibility
    }

def get_gemini_suggestions(text, api_key):
    genai.configure(api_key=api_key)
    # Using the requested Gemini 3 version
    model = genai.GenerativeModel('gemini-3-flash-preview')
    
    prompt = f"""
    Analyze the following resume text and provide:
    1. Strengths (3 points)
    2. Weaknesses (3 points)
    3. Improvements (3 points)
    4. Recommended Projects (2 ideas)
    5. Skills to improve (3 skills)
    
    Resume Text:
    {text[:2000]}
    
    Format the response as a JSON with keys: strengths, weaknesses, improvements, recommended_projects, skills_to_improve.
    Each value should be a list of strings.
    """
    
    try:
        response = model.generate_content(prompt)
        # Ensure we got a valid response
        if not response or not response.text:
            raise Exception("Empty response from Gemini")
            
        json_text = re.search(r'\{.*\}', response.text, re.DOTALL).group()
        import json
        return json.loads(json_text)
    except Exception as e:
        print(f"Gemini Error: {e}")
        error_msg = str(e)
        if "429" in error_msg:
            return {"error": "API Quota Exceeded. Please try again in a few minutes or check your Gemini API billing."}
        return {"error": error_msg}

def get_market_intelligence(role, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-3-flash-preview')
    prompt = f"Provide job market intelligence for the role: {role}. Include: 1. Current demand (High/Medium/Low), 2. Top hiring companies, 3. Key emerging trends, 4. Remote work availability. Format as JSON with keys: demand, companies, trends, remote_availability."
    try:
        response = model.generate_content(prompt)
        json_text = re.search(r'\{.*\}', response.text, re.DOTALL).group()
        import json
        return json.loads(json_text)
    except Exception as e:
        return {"error": str(e)}

def get_course_recommendations(skills, role, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-3-flash-preview')
    prompt = f"Based on these missing skills: {', '.join(skills)} for the role {role}, recommend 3-4 specific online courses or learning paths. Format as JSON with key 'courses' containing a list of objects with 'title' and 'platform'."
    try:
        response = model.generate_content(prompt)
        json_text = re.search(r'\{.*\}', response.text, re.DOTALL).group()
        import json
        return json.loads(json_text)
    except Exception as e:
        return {"error": str(e)}

def get_salary_prediction(role, text, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-3-flash-preview')
    prompt = f"Predict the salary range in Indian Rupees (INR) for a {role} in the Indian job market based on this resume snippet: {text[:1000]}. Provide: 1. Entry level, 2. Mid level, 3. Senior level. Format as JSON with keys: entry, mid, senior. Ensure values are in Lakhs per Annum (LPA) or standard Indian format."
    try:
        response = model.generate_content(prompt)
        json_text = re.search(r'\{.*\}', response.text, re.DOTALL).group()
        import json
        return json.loads(json_text)
    except Exception as e:
        return {"error": str(e)}

def get_interview_prep(role, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-3-flash-preview')
    prompt = f"Provide interview preparation for {role}. Include: 1. Top 5 technical questions, 2. Top 3 behavioral questions, 3. A key tip for success. Format as JSON with keys: technical_questions, behavioral_questions, tip."
    try:
        response = model.generate_content(prompt)
        json_text = re.search(r'\{.*\}', response.text, re.DOTALL).group()
        import json
        return json.loads(json_text)
    except Exception as e:
        return {"error": str(e)}

def get_job_matching(resume_text, job_desc, role_skills):
    if not job_desc:
        return {
            "score": 0,
            "label": "N/A",
            "matching_skills": [],
            "missing_keywords": [],
            "compatibility": "N/A"
        }
    
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([resume_text, job_desc])
    similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    percentage = round(similarity * 100, 2)
    
    if percentage > 75: 
        label = "Good"
        compatibility = "Compatible"
    elif percentage > 50: 
        label = "Moderate"
        compatibility = "Semi-Compatible"
    else: 
        label = "Low"
        compatibility = "Incompatible"
        
    found_skills = get_extracted_skills(resume_text)
    job_skills = get_extracted_skills(job_desc)
    
    if not job_skills:
        job_skills = [s for s in role_skills if s in job_desc.lower()]
    if not job_skills:
        job_skills = role_skills[:6]
        
    matching_skills = [s for s in found_skills if s in job_skills]
    missing_keywords = [s for s in job_skills if s not in found_skills]
    
    return {
        "score": percentage,
        "label": label,
        "matching_skills": list(set(matching_skills)),
        "missing_keywords": list(set(missing_keywords)),
        "compatibility": compatibility
    }

def get_role_based_fallback(role, text):
    role_lower = role.lower()
    
    # 1. Salary prediction fallbacks based on role (INR in LPA)
    salary = {"entry": "₹5L - ₹8L", "mid": "₹10L - ₹18L", "senior": "₹22L - ₹40L"}
    if "data scientist" in role_lower or "machine learning" in role_lower:
        salary = {"entry": "₹6L - ₹9L", "mid": "₹12L - ₹22L", "senior": "₹25L - ₹45L"}
    elif "data analyst" in role_lower or "analytics" in role_lower or "business analyst" in role_lower:
        salary = {"entry": "₹4.5L - ₹7L", "mid": "₹9L - ₹16L", "senior": "₹18L - ₹30L"}
    elif "ui/ux designer" in role_lower or "designer" in role_lower:
        salary = {"entry": "₹4L - ₹7L", "mid": "₹8L - ₹15L", "senior": "₹18L - ₹32L"}
    elif "software engineer" in role_lower or "cloud" in role_lower or "data engineer" in role_lower or "devops" in role_lower:
        salary = {"entry": "₹5L - ₹8L", "mid": "₹11L - ₹20L", "senior": "₹24L - ₹42L"}
    elif "web developer" in role_lower:
        salary = {"entry": "₹4L - ₹6.5L", "mid": "₹8L - ₹15L", "senior": "₹18L - ₹30L"}
        
    # 2. Market Intelligence fallbacks
    market = {
        "demand": "High",
        "companies": ["TCS", "Infosys", "Wipro", "Cognizant", "Accenture"],
        "trends": ["Digital Transformation", "Cloud Migration", "AI-Assisted Dev"],
        "remote_availability": "Medium"
    }
    if "data scientist" in role_lower or "machine learning" in role_lower:
        market = {
            "demand": "Very High",
            "companies": ["Google", "Microsoft", "Meta", "Amazon", "OpenAI"],
            "trends": ["Generative AI Integration", "MLOps Automation", "LLM Fine-tuning"],
            "remote_availability": "High"
        }
    elif "data analyst" in role_lower or "analytics" in role_lower or "business analyst" in role_lower:
        market = {
            "demand": "High",
            "companies": ["TCS", "Accenture", "Google", "Amazon", "Mu Sigma"],
            "trends": ["Power BI/Tableau dashboarding", "SQL query performance optimization", "Big Data cloud analytics"],
            "remote_availability": "Medium"
        }
    elif "ui/ux designer" in role_lower or "designer" in role_lower:
        market = {
            "demand": "High",
            "companies": ["Figma", "Canva", "Airbnb", "Shopify", "Accenture"],
            "trends": ["Design systems scaling", "Micro-interactions animation", "Accessibility guidelines (WCAG)"],
            "remote_availability": "Very High"
        }
    elif "software engineer" in role_lower or "cloud" in role_lower or "data engineer" in role_lower or "devops" in role_lower:
        market = {
            "demand": "High",
            "companies": ["Microsoft", "AWS", "Google", "Salesforce", "Accenture"],
            "trends": ["Serverless Architecture", "Kubernetes Scaling", "Infrastructure as Code"],
            "remote_availability": "High"
        }
    elif "web developer" in role_lower:
        market = {
            "demand": "High",
            "companies": ["Shopify", "Vercel", "Wix", "TCS", "Accenture"],
            "trends": ["Next.js & Server Components", "Headless CMS", "Micro-frontends"],
            "remote_availability": "Very High"
        }
        
    # 3. Recommended Courses
    courses = [
        {"title": f"Professional Certificate in {role}", "platform": "Coursera"},
        {"title": f"Become a {role} Nanodegree", "platform": "Udacity"},
        {"title": f"Master {role} Bootcamp", "platform": "Udemy"}
    ]
    if "data scientist" in role_lower or "machine learning" in role_lower:
        courses = [
            {"title": "Machine Learning Specialization by Andrew Ng", "platform": "DeepLearning.AI / Coursera"},
            {"title": "Applied Data Science with Python Specialization", "platform": "University of Michigan / Coursera"},
            {"title": "Data Science MicroMasters Program", "platform": "MIT / edX"}
        ]
    elif "data analyst" in role_lower or "analytics" in role_lower or "business analyst" in role_lower:
        courses = [
            {"title": "Google Data Analytics Professional Certificate", "platform": "Google / Coursera"},
            {"title": "Data Analysis with Python Career Path", "platform": "freeCodeCamp"},
            {"title": "SQL for Data Science Course", "platform": "UC Davis / Coursera"}
        ]
    elif "ui/ux designer" in role_lower or "designer" in role_lower:
        courses = [
            {"title": "Google UX Design Professional Certificate", "platform": "Google / Coursera"},
            {"title": "User Experience Design Essentials", "platform": "Udemy"},
            {"title": "UI/UX Design Career Path Program", "platform": "Interaction Design Foundation"}
        ]
    elif "software engineer" in role_lower or "cloud" in role_lower or "data engineer" in role_lower or "devops" in role_lower:
        courses = [
            {"title": "Software Engineering Masterclass", "platform": "Udemy"},
            {"title": "AWS Certified Solutions Architect Course", "platform": "Udemy / AWS"},
            {"title": "Google Cloud Professional Cloud Architect certification", "platform": "Google Cloud"}
        ]
    elif "web developer" in role_lower:
        courses = [
            {"title": "The Complete 2026 Web Development Bootcamp", "platform": "Udemy"},
            {"title": "React - The Complete Guide (incl Hooks, React Router, Redux)", "platform": "Udemy"},
            {"title": "Full-Stack Web Developer Career Path", "platform": "Scrimba"}
        ]
        
    # 4. Interview Preparation
    interview = {
        "technical_questions": [
            "Explain the difference between SQL and NoSQL databases.",
            "How do you handle API rate limiting in your code?",
            "What is version control, and how does git merge work?"
        ],
        "behavioral_questions": [
            "Tell me about a challenging project you worked on.",
            "How do you keep up with new technology trends?"
        ],
        "tip": "Practice whiteboarding and explain your thought process clearly."
    }
    if "data scientist" in role_lower or "machine learning" in role_lower:
        interview = {
            "technical_questions": [
                "What is the difference between L1 and L2 regularization?",
                "How do you handle imbalanced datasets in machine learning?",
                "Explain the architecture of a Transformer model."
            ],
            "behavioral_questions": [
                "Tell me about a time you had to explain a complex ML model to a non-technical stakeholder.",
                "How do you evaluate model performance in production?"
            ],
            "tip": "Be prepared to write code on a whiteboard and explain the bias-variance tradeoff."
        }
    elif "data analyst" in role_lower or "analytics" in role_lower or "business analyst" in role_lower:
        interview = {
            "technical_questions": [
                "What is a window function in SQL and when would you use it?",
                "How do you handle missing values or outliers in a dataset?",
                "What is the difference between a left join and an inner join in SQL?"
            ],
            "behavioral_questions": [
                "Describe a time you found a surprising pattern in data and how you communicated it to managers.",
                "How do you deal with conflicting metrics when analyzing performance?"
            ],
            "tip": "Be ready to explain SQL window functions and basic statistics (mean, median, standard deviation)."
        }
    elif "ui/ux designer" in role_lower or "designer" in role_lower:
        interview = {
            "technical_questions": [
                "What is your design process and how do you conduct user research?",
                "What is the difference between a wireframe, a prototype, and a mockup?",
                "How do you ensure accessibility (WCAG compliance) in your mobile designs?"
            ],
            "behavioral_questions": [
                "How do you handle negative feedback from client stakeholders on your designs?",
                "Tell me about a time you made a design decision based on user feedback."
            ],
            "tip": "Prepare to present a portfolio case study explaining the user problem, research, user flows, and final visual mocks."
        }
    elif "web developer" in role_lower:
        interview = {
            "technical_questions": [
                "What is event bubbling and event capturing in JavaScript?",
                "Explain the virtual DOM and how React renders changes.",
                "What are web components and server-side rendering (SSR)?"
            ],
            "behavioral_questions": [
                "Describe a time you had a disagreement with a UI designer and how you resolved it.",
                "How do you optimize page loading performance?"
            ],
            "tip": "Brush up on CSS Grid/Flexbox, ES6+ syntax, and web performance optimization techniques."
        }
        
    # 5. Suggestions
    suggestions = {
        "strengths": [
            f"Shows solid understanding of {role} fundamentals.",
            "Good presentation of technical skills.",
            "Clear educational background matches target industry."
        ],
        "weaknesses": [
            "Could include more quantifiable metrics and achievements.",
            "Missing some emerging technology stack items.",
            "Needs more links to live portfolios or code bases."
        ],
        "improvements": [
            "Add key metrics and outcomes to project descriptions.",
            "Include GitHub links for personal repositories.",
            "Standardize sections in a clean, single-page format."
        ]
    }
    
    return {
        "salary": salary,
        "market": market,
        "courses": courses,
        "interview": interview,
        "suggestions": suggestions
    }
