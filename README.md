[README.md](https://github.com/user-attachments/files/32465683/README.md)
[README (1) (1).md](https://github.com/user-attachments/files/32464937/README.1.1.md)
<div align="center" style="border: 2px solid #ccc; padding: 20px; border-radius: 12px; width: 80%; margin: auto; box-shadow: 0 0 10px rgba(0,0,0,0.15);">
    <img
        width="180"
        height="220"
        alt="Logo - SURE ProEd"
        src="https://github.com/user-attachments/assets/88fa5098-24b1-4ece-87df-95eb920ea721"
        style="border-radius: 10px;"
    />

  <h1 align="center" style="font-family: Arial; font-weight: 600; margin-top: 15px;">SURE ProEd (formerly SURE Trust) 
      </h1>
<h2 style="color: #2b6cb0; font-family: Arial;">Skill Upgradation for Rural youth Empowerment Trust</h2>
</div>

<hr style="border: 0; border-top: 1px solid #ccc; width: 80%;" />

<div style="padding: 20px; border: 2px solid #ddd; border-radius: 12px; width: 90%; margin: auto; background: #fafafa; font-family: Arial;">

<h2 style = "color:#333;"> Student Details </h2>
<div align = "left" style ="margin: 20px; font-size: 16px;">
    <p><strong>Name:</strong> Siddhi Gandhi </p>
    <p><strong>Email ID:</strong> siddhigandhi.g37python@gmail.com </p>
    <p><strong>College Name:</strong> D.Y.Patil College of Engineering and technology, Kolhapur </p>
    <p><strong>Branch/Specialization :</strong> B.Tech in Computer Science and Engineering (AI & ML) </p>
    <p><strong>College ID:</strong> 5880923 </p>
</div>

<hr style="border: 0; border-top: 1px solid #ccc; width: 80%;" />

<h2 style="color:#333;"> Course Details </h2>
<div align="left" style="margin: 20px; font-size: 16px;">
    <p><strong>Course Opted:</strong> 6-Month Project-Based Internship in Artificial Intelligence & Machine Learning </p>
    <p><strong>Instructor Name:</strong> Gaurav Sir </p>
</div>

<div align="left" style="margin: 20px; font-size: 16px;">
    <p><strong>Duration:</strong> 6 Months </p>

<hr style="border: 0; border-top: 1px solid #ccc; width: 80%;" />

<h2 style="color:#333;"> Trainer Details </h2>

<div align="left" style="margin: 20px; font-size: 16px;">

<p><strong>Trainer Name:</strong> Gaurav Patel </p>
<p><strong>Trainer Email ID:</strong>gaurav.patel.gpp@gmail.com </p>
<p><strong>Trainer Designation:</strong>Data Science Instructor</p>

<hr style="border: 0; border-top: 1px solid #ccc; width: 80%;" />

<h2 style="color:#333;"> Projects Completed </h2>



# ResumeAI

Developed ResumeAI, an AI-powered Resume Analyzer and Career Guidance System using Python, NLP, TF-IDF, and Machine Learning to analyze PDF/DOCX resumes and predict suitable career roles across 12 profiles. Trained and compared Random Forest, Logistic Regression, Naive Bayes, and AdaBoost models, with Random Forest achieving 92% classification accuracy. Implemented an ATS scoring engine, cosine similarity-based job matching, resume strength prediction, and Explainable AI (XAI) for interpretable results. Integrated Google Gemini AI to generate personalized career roadmaps, skill-gap analysis, salary insights, course recommendations, and interview preparation. Developed an interactive Flask web application with Plotly visualizations and automated PDF report generation using ReportLab.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Data Collection & Preprocessing](#data-collection--preprocessing)
3. [Feature Engineering & Embeddings](#feature-engineering--embeddings)
4. [Modeling Approaches](#modeling-approaches)
   - [Traditional Machine Learning Models](#traditional-machine-learning-models)
   - [Deep Learning Models](#deep-learning-models)
5. [Experiment Tracking with MLflow](#experiment-tracking-with-mlflow)
6. [Deployment](#deployment)
7. [Docker Image](#docker-image)
8. [Conclusion](#conclusion)

---

<hr style="border: 0; border-top: 1px solid #ccc; width: 80%;" />


## Project Overview

**ResumeAI** bridges the gap between traditional Applicant Tracking Systems (ATS) and modern AI career guidance platforms. Standard ATS systems rely on strict keyword matching, often missing qualified candidates due to minor phrasing differences. ResumeAI solves this by deploying a hybrid ML + GenAI architecture:
- **Automated Resume Parsing**: Extracts structured text from multi-page `.pdf` and `.docx` documents using `pdfminer.six` and `python-docx`.
- **Career Role Classification**: Predicts the candidate's best-fitting role across 12 industry categories (Data Scientist, Software Engineer, Web Developer, Data Analyst, Business Analyst, Product Manager, Machine Learning Engineer, DevOps Engineer, UI/UX Designer, Cybersecurity Analyst, Cloud Architect, Mobile Developer) using a Random Forest Classifier trained on 3,001 resumes.
- **Weighted ATS Scoring Engine**: Evaluates resume quality across 5 weighted categories: Skills Match (30%), Projects (25%), Experience (20%), Certifications (15%), and Document Structure (10%).
- **Explainable AI (XAI)**: Demystifies black-box ML predictions by plotting local feature-attribution weights using TF-IDF feature importance.
- **Generative AI Insights**: Integrates Google Gemini AI (`gemini-3-flash-preview`) to dynamically generate personalized 6-month career roadmaps, skill gap analysis, market intelligence, Indian market salary estimations (LPA), and role-specific interview preparation.
- **Production-Grade Architecture**: Includes an automated fallback engine for API rate limits and full multi-stage Docker deployment support.

---

## Data Collection & Preprocessing

### Dataset Summary
The machine learning pipeline is trained on `resume_dataset.csv`, a corpus containing **3,001 resume text samples** balanced across 12 tech and management job profiles.
### NLP Preprocessing Pipeline
Text extracted from resumes undergoes a rigorous NLP pipeline implemented in `utils.py` and `model_trainer.py`:
### Local NLTK Storage
To ensure full portability and prevent missing data crashes during server startup, all NLTK data (`punkt`, `stopwords`, `wordnet`, `omw-1.4`, `punkt_tab`) is cached locally in an `nltk_data/` folder inside the project root.

---

## Feature Engineering & Embeddings

### 1. TF-IDF Vectorization
The preprocessed text is transformed into high-dimensional numerical feature vectors using **Term Frequency-Inverse Document Frequency (TF-IDF)**:
- **Vocabulary Size**: 5,000 top features (`max_features=5000`)
- **N-gram Range**: Unigrams (`ngram_range=(1,1)`)
- **Sublinear TF Scaling**: Reduces the impact of very high-frequency terms.
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{N}{|\{d \in D : t \in d\}|}\right)$$
### 2. Job Description Match via Cosine Similarity
To compare a resume against a specific target Job Description (JD), the system computes the cosine angle between their TF-IDF vector embeddings:
$$\text{Cosine Similarity} = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}$$
- **Similarity > 75%**: Compatible (Good)
- **50% - 75%**: Semi-Compatible (Moderate)
- **< 50%**: Incompatible (Low)

## Modeling Approaches

### Traditional Machine Learning Models
Four multi-class classification algorithms were trained and evaluated on an 80/20 train-test split (2,400 training samples, 601 test samples):
1. **Random Forest Classifier (Chosen Best Model)**: An ensemble of 100 decision trees. Handles sparse TF-IDF text vectors exceptionally well without overfitting.
2. **Logistic Regression**: Linear classifier with $L_2$ regularization (`max_iter=1000`).
3. **Multinomial Naive Bayes**: Probabilistic baseline classifier for document classification.
4. **AdaBoost Classifier**: Boosting ensemble classifier using decision stumps.
In addition, a **Random Forest Regressor** is trained to predict a continuous **Resume Strength Rating** (0 to 100), paired with a Classifier that labels resumes into **Strong**, **Average**, or **Weak**.
### Generative AI & Hybrid Fallback Engine
ResumeAI combines deterministic Machine Learning with Generative AI via the **Google Gemini API**:

                 ┌────────────────────────┐
                   │  Resume Upload & NLP   │
                   └───────────┬────────────┘
                               │
                ┌──────────────┴──────────────┐
                ▼                             ▼
   ┌────────────────────────┐    ┌────────────────────────┐
   │ Random Forest Pipeline │    │   Google Gemini API    │
   │ (Role & ATS Scoring)   │    │  (GenAI Insights)      │
   └────────────┬───────────┘    └────────────┬───────────┘
                │                             │
                │                    Quota Exceeded (429)?
                │                       ├── Yes ──► Local Rule-Based Engine
                │                       └── No  ──► Dynamic Gemini JSON
                ▼                             ▼
   ┌──────────────────────────────────────────────────────┐
   │     Unified Interactive Dashboard & PDF Export      │
   └──────────────────────────────────────────────────────┘

### Deep Learning Models

To capture complex patterns, we built a deep learning model using TensorFlow Keras:

- **Two-Layer LSTM:** The model includes an embedding layer, two LSTM layers, and dense layers with dropout for regularization.
- **Text Tokenization & Padding:** We convert raw text into sequences using Keras’ Tokenizer and pad them to a uniform length.
- **Evaluation:** Model performance is evaluated on standard metrics and confusion matrices are logged.

---

## Experiment Tracking with MLflow

Model evaluation metrics were recorded during training and exported to `metrics.csv`:
| Model Name | Accuracy | Precision | Recall | F1 Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **92.0%** | **91.0%** | **93.0%** | **92.0%** | **BEST MODEL (Selected)** |
| **Logistic Regression** | 87.0% | 86.0% | 88.0% | 87.0% | Evaluated |
| **Naive Bayes** | 81.0% | 80.0% | 83.0% | 81.0% | Evaluated |
| **AdaBoost** | 74.0% | 73.0% | 75.0% | 74.0% | Evaluated |

## 6. Deployment & Interactive Dashboard
The application is deployed using **Flask 3.0** with a responsive frontend built on **Tailwind CSS**, **FontAwesome**, and **Plotly.js**.
### Dashboard Modules
1. **Overall ATS Score Card**: Weighted percentage breakdown with visual progress bar.
2. **Top 3 Predicted Roles**: Ensemble probabilities for top matching career categories.
3. **Skill Gap Analysis**: Found vs missing skill tags compared against target role requirements.
4. **Indian Salary Prediction (LPA)**: Entry-level, Mid-level, and Senior-level salary expectations.
5. **Market Intelligence**: Current demand status, top hiring companies, and emerging technology trends.
6. **Recommended Courses**: Curated learning paths from Coursera, Udemy, edX, etc.
7. **Interview Preparation**: Top role-specific technical questions, behavioral prompts, and expert tips.
8. **Company Readiness Score**: Circular progress indicators matching candidate skills against tech giant stacks (Google, Microsoft, Amazon, TCS, Accenture).
9. **Personalized 6-Month Career Learning Roadmap**: Step-by-step monthly milestones tailored to the selected target role.
10. **Resume Section Analyzer**: Structural review of Education, Projects, Experience, Certifications, and Achievements.
11. **Explainable AI (XAI)**: Feature-influence bar charts showing exact keyword weights driving model predictions.
12. **PDF Report Downloader**: Exportable dynamic multi-page PDF generated on-the-fly using `reportlab`.


### Overfitting & Anomaly Checks
- **Train Accuracy**: 94.1%
- **Test Accuracy**: 92.0%
- **Variance**: 2.1% (Minimal variance proves excellent generalization on unseen data).




---
### Local Setup & Run Commands
```bash
# 1. Clone repository
git clone https://github.com/Siddhigandhi854/ResumeAI.git
cd Resume1
# 2. Install dependencies
pip install -r requirements.txt
# 3. Train models
python model_trainer.py
# 4. Start Flask application
python app.py

## Conclusion

ResumeAI demonstrates a complete, industry-ready Machine Learning and GenAI software lifecycle. By combining deterministic NLP tokenization, TF-IDF vectorization, ensemble Random Forest classification, Explainable AI (XAI), and Google Gemini Generative AI with a resilient fallback mechanism, the platform offers an accurate, transparent, and scalable solution for resume parsing and career development.



<hr style="height:1px; border-top:1px solid #ccc; width:80%;" />

<h2 id="project-report" style="color:#333;"> Project Report </h2>

<p>
  <a href="https://github.com/Siddhigandhi854/ResumeAI1/blob/main/ResumeAI_Project_FinalReport.pdf" target="_blank">
    <strong>→ View Full Project Report</strong>
  </a>
</p>

<hr style="height:1px; border-top:1px solid #ccc; width:80%;" />

## **References**
[![Python](https://img.shields.io/badge/Python-3.8-orange)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-2.0.2-blue)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.2.3-blue)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5.2-yellow)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-yellow)](https://www.tensorflow.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.20.3-lightblue)](https://mlflow.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7.6-green)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.43.0-red)](https://streamlit.io/)


## **Learnings from LST and SST**

LST and SST sessions played an important role in improving both my technical understanding and professional development throughout the internship period. These sessions provided valuable exposure to industry-oriented practices, communication techniques, and structured learning methodologies which helped me become more confident and disciplined in my approach towards project development and teamwork.

Through these sessions, I improved my communication skills, presentation abilities, and interpersonal interaction skills. Participating in discussions, collaborative activities, and learning exercises helped me develop confidence while expressing ideas, explaining technical concepts, and working effectively in team-oriented environments.

The sessions also helped me understand the importance of problem-solving, analytical thinking, time management, and proper documentation practices in real-world project development. I learned how structured planning, consistency, and systematic execution contribute significantly towards successful completion of technical projects and assignments.

In addition, LST and SST sessions provided guidance regarding professional ethics, workplace expectations, collaborative learning, and continuous self-improvement. These learnings helped me maintain a more organized and professional workflow during the development of the IntelliLearn-AI project and strengthened my understanding of practical implementation strategies.

Overall, the LST and SST sessions contributed significantly to my personal growth, professional readiness, and understanding of real-world project environments. The experience gained from these sessions enhanced both my technical confidence and my ability to work responsibly and effectively in academic as well as professional settings.

---

## **Community Services**

During my internship period, I actively participated in community-oriented activities focused on social responsibility and public welfare. These activities helped me understand the importance of contributing positively to society and working collaboratively for community development.

### **Activities Involved**

- **Tree Plantation Drive** – Participated in tree plantation activities near Ratnagiri, Maharashtra and contributed towards promoting environmental awareness and greener surroundings.

- **Food Distribution Activity** – Participated in food distribution activities near Ratnagiri, Maharastra as part of social service and community support initiatives for needy people.
- **Blood Donation – Participated in a blood donation activity and voluntarily donated blood to support patients in need and contribute towards community healthcare.

### **Impact / Contribution**

- Contributed towards environmental improvement through plantation activities.
- Supported community welfare initiatives by helping in food distribution activities.
- Improved communication, coordination, teamwork, and social responsibility skills.
- Gained practical exposure to community service and collaborative volunteering activities.
--- 
### **Photos**

<div align="center">

<img src="https://github.com/sure-trust/SIDDHI-SHASHIKANT-GANDHI-g37-ai-ml/blob/main/Tree%20Plantation.jpeg" alt="Community Service Photo 1" width="41%">

<img src="https://github.com/sure-trust/SIDDHI-SHASHIKANT-GANDHI-g37-ai-ml/blob/main/Blood%20Donation.jpeg" alt="Community Service Photo 2" width="41%">

<img src="https://github.com/sure-trust/SIDDHI-SHASHIKANT-GANDHI-g37-ai-ml/blob/main/Food%20Distribution.jpeg" alt="Community Service Photo 3" width="41%">

</div>
---

## **Certificate**

The internship certificate serves as an official acknowledgment of the successful completion of my training period. It will be issued by the organization upon fulfilling all required tasks and meeting the performance expectations of the program. The certificate validates the skills, experience, and contributions made during the internship.

<!-- add your certificate image url below (inside src='')-->

<p align="center">
<img src="https://github.com/Lord-Rahul/Practice-Programs/blob/main/react/1/public/Gemini_Generated_Image_a6w8rda6w8rda6w8.png?raw=true" alt="Internship Certificate" width="80%">
</p>

---

## **Acknowledgments**

<!-- you can add Acknowledgments over here in same syntax as below . eg trainer name , company name , role etc -->

- [Prof. Radhakumari Challa](https://www.linkedin.com/in/prof-radhakumari-challa-a3850219b) , Executive Director and Founder - [SURE Trust](https://www.suretrustforruralyouth.com/)



