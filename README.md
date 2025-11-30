# NeonHire — Smart Resume Screener

## Overview
NeonHire is an AI-powered resume screening agent that extracts skills from job descriptions and resumes, computes an ATS-like score, highlights matched/missing/extra skills, evaluates experience with seniority multipliers, and provides an explainable breakdown.

## Features
- Robust PDF/DOCX/TXT extraction (pdfplumber, python-docx)
- Universal skill extractor: technical + soft + business + retail + multi-word phrase skills + synonyms
- Improved scoring with extras bonus and seniority multiplier
- Explain Score: quick summary, natural-language analysis, suggestions
- Matched / Missing / Extra skill badges on each result card
- Resume snippet preview
- CSV export (download button)

## Run locally
git clone <your-repo-url>
cd neonhire-smart-resume-screener

python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py

## Architecture
User → Upload JD + Resumes
          ↓
   Text Extraction Layer
 (PDF / DOCX / TXT parser)
          ↓
   Universal Skill Extractor
  (technical + soft + synonyms)
          ↓
    Experience Analyzer
 (years + seniority detection)
          ↓
   ATS-style Scoring Engine
 (skills match + seniority + extras)
          ↓
 Explainable Score Generator
 (analysis + suggestions)
          ↓
   Streamlit UI Renderer
 (cards + badges + scorebars)

## Tested On
-Technical roles (Python/Java/Full-stack)
-Non-technical roles (Customer Service, Sales, Retail)
-Manager/Lead/Director positions
-Freshers and senior profiles

## Future Enhancements
-AI embedding similarity scoring
-JD-based automatic skill extraction
-Keyword highlighting inside resumes
-Multi-candidate comparison mode
-Role classification

## Acknowledgements
Built using:
-Streamlit
-pdfplumber
-python-docx
-pandas
