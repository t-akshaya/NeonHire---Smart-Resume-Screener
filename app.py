# app.py — FINAL CLEAN C3 VERSION (Universal JD/resume skill extraction + improved scoring & explain)
# Copy-paste this entire file (single file). UI/CSS unchanged, scoring/extraction improved for ANY JD.

import streamlit as st
import pdfplumber, docx, io, os, re, base64
import pandas as pd
from datetime import datetime

# ---------------- Page Config ----------------
st.set_page_config(page_title="Resume Screener", layout="wide")

# ---------------- CSS ----------------
CSS = r"""
<style>
:root {
  --bg:#05060d;
  --muted:#9fb0d6;
  --a1:#00f0ff;
  --a2:#9b5cff;
}

body, .main {
  background: var(--bg) !important;
  color: #e6f6ff !important;
  font-family: Inter, Arial, sans-serif;
}

.main .block-container {
  max-width: 1200px;
  margin-left: auto;
  margin-right: auto;
  padding: 12px;
}

/* Title - Premium Neon Hero Style */

h1 {
  text-align: center;
  font-size: 62px !important;       /* large, premium size */
  font-weight: 900 !important;
  letter-spacing: 1px;
  margin-top: 0px !important;
  margin-bottom: 6px !important;

  background: linear-gradient(90deg, #4cc9ff, #6a5cff, #b443ff);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;

  /* REMOVE blur — keep text SUPER SHARP */
  text-shadow: none !important;
}




.resumes {
  margin-top: 18px;
  background: rgba(255,255,255,0.03);
  padding: 16px;
  border-radius: 12px;
  border: 1px solid rgba(255,255,255,0.06);
}

.stButton>button {
  background: linear-gradient(90deg, var(--a1), var(--a2)) !important;
  color: #02101a !important;
  font-weight: 800;
  border-radius: 10px !important;
  padding: 9px 16px !important;
}

.card {
  background: rgba(10,16,30,0.96);
  padding: 18px;
  border-radius: 12px;
  border: 1px solid rgba(255,255,255,0.05);
  font-size: 16px;
}

.grid-cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
  gap: 18px;
  margin-top: 16px;
}

.score-badge {
  padding: 10px 18px;
  border-radius: 999px;
  font-weight: 900;
  font-size: 24px;
  color: #02101a;
}

.score-bar {
  height: 16px;
  border-radius: 999px;
  background: rgba(255,255,255,0.06);
  overflow: hidden;
  margin-top: 8px;
}

.score-fill {
  height: 100%;
  border-radius: 999px;
  transition: width .6s ease;
}

.preview {
  background: rgba(0,0,0,0.45);
  padding: 10px;
  border-radius: 10px;
  font-family: monospace;
  font-size: 14px;
  white-space: pre-wrap;
}

.small-muted {
  color: var(--muted);
  font-size: 14px;
}

/* inline tag helpers if displayed via HTML */
.badge {
  display:inline-block;
  padding:6px 10px;
  border-radius:999px;
  font-size:12px;
  margin-right:6px;
  margin-bottom:6px;
}
.badge-matched { background: rgba(0,255,204,0.08); color:#00ffcc; border:1px solid rgba(0,255,204,0.12); }
.badge-missing { background: rgba(255,92,124,0.06); color:#ff5c7c; border:1px solid rgba(255,92,124,0.12); }
.badge-extra { background: rgba(255,255,255,0.03); color:var(--muted); border:1px solid rgba(255,255,255,0.04); }

.left, .right {
  background: rgba(255,255,255,0.03);
  border-radius: 12px;
  padding: 16px;
  border: 1px solid rgba(255,255,255,0.06);
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------------- Sample Data ----------------
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
SAMPLES = {"alice.txt": "Alice sample resume", "bob.txt": "Bob sample resume"}
for fn, txt in SAMPLES.items():
    p = os.path.join(DATA_DIR, fn)
    if not os.path.exists(p):
        with open(p, "w", encoding="utf8") as f: f.write(txt)

# ---------------- Helper Functions & Skill Lists ----------------

# Hard technical skills (existing)
TECHNICAL_SKILLS = [
    "python","java","c++","javascript","react","node","django","flask",
    "rest","api","sql","postgresql","mysql","docker","kubernetes","aws","azure","gcp",
    "git","testing","pytest","pandas","html","css","typescript","spring","spring boot",
    "machine learning","ml","deep learning","scikit-learn","tensorflow","keras"
]

# Soft / domain skills (retail / sales / management / general)
SOFT_SKILLS = [
    "customer service", "customer support", "communication", "leadership", "teamwork",
    "problem solving", "conflict resolution", "scheduling", "time management",
    "sales", "merchandising", "returns", "point of sale", "pos", "transaction",
    "cash handling", "inventory", "store maintenance", "staff management",
    "coaching", "training", "people management", "supervising", "shift scheduling",
    "reporting", "metrics", "kpIs", "customer satisfaction", "csat", "upselling"
]

# Managerial / business skills
BUSINESS_SKILLS = [
    "project management", "stakeholder management", "budgeting", "planning",
    "analysis", "strategic planning", "operations", "process improvement",
    "vendor management", "contract negotiation", "presentations"
]

# Phrases to catch multiword skills
PHRASE_SKILLS = [
    "machine learning", "deep learning", "data analysis", "data science",
    "object oriented programming", "rest api", "api development",
    "web development", "unit testing", "test automation", "microservices",
    "customer service", "conflict resolution", "point of sale", "cash handling",
    "team management", "staff management", "project management", "store maintenance"
]

# Synonym mapping (domain-independent)
SYNONYMS = {
    "js": "javascript",
    "py": "python",
    "postgres": "postgresql",
    "sql server": "sql",
    "mariadb": "mysql",
    "frontend": ["html","css","javascript"],
    "backend": ["python","java","node"],
    "cloud": ["aws","azure","gcp"],
    "pos": "point of sale",
    "customer-support": "customer support",
    "csat": "customer satisfaction",
    "ux": "user experience",
    "ui": "user interface",
    "mgr": "manager",
    "supervisor": "supervising",
    "people management": "staff management"
}

# Seniority keywords for multiplier
SENIORITY_KEYWORDS = {
    "senior": 1.08, "sr.": 1.06, "lead": 1.07, "principal": 1.10,
    "junior": 0.95, "jr": 0.95, "intern": 0.85, "internship": 0.85,
    "expert": 1.10, "experienced": 1.04, "manager": 1.06, "director": 1.12
}

# Utility: normalize text
def normalize_text(t):
    if not t:
        return ""
    return re.sub(r'\s+', ' ', t).strip().lower()

# Robust PDF/docx/text extraction
def extract_text_from_pdf(b):
    txt = ""
    try:
        with pdfplumber.open(io.BytesIO(b)) as pdf:
            for p in pdf.pages:
                txt += (p.extract_text() or "") + "\n"
    except Exception:
        try:
            txt = b.decode("utf-8", "ignore")
        except:
            txt = ""
    return txt

def extract_text_from_docx(b):
    try:
        doc = docx.Document(io.BytesIO(b))
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception:
        try:
            return b.decode("utf-8", "ignore")
        except:
            return ""

def extract_text(uploaded):
    name = getattr(uploaded, "name", "").lower()
    b = uploaded.read()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(b)
    if name.endswith(".docx"):
        return extract_text_from_docx(b)
    # fallback to decode text
    try:
        return b.decode("utf-8", "ignore")
    except:
        return ""

# Years extraction - robust and safe
def find_years(txt):
    txt = txt or ""
    # look for "YYYY" ranges first
    yrs = re.findall(r"(\d{4})", txt)
    clean = []
    for y in yrs:
        try:
            yi = int(y)
            if 1900 < yi <= datetime.now().year:
                clean.append(yi)
        except:
            continue
    if len(clean) >= 2:
        return max(clean) - min(clean)
    # fallback: "X years" pattern
    m = re.search(r'(\d{1,2})\+?\s+years', txt.lower())
    if m:
        try:
            return int(m.group(1))
        except:
            pass
    # fallback: seniority words imply experience but not quantified -> return 0
    return 0

# anonymize PII
def anonymize(t):
    if not t:
        return t
    t = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[EMAIL]", t)
    t = re.sub(r"\+?\d[\d\s\-]{7,}", "[PHONE]", t)
    return t

# ---------------- Universal Skill Extraction (JD + Resume) ----------------
ALL_SKILLS_CANONICAL = sorted(set(TECHNICAL_SKILLS + SOFT_SKILLS + BUSINESS_SKILLS + PHRASE_SKILLS))

def expand_synonyms_in_text_set(text):
    matches = set()
    t = text.lower()
    for syn, mapped in SYNONYMS.items():
        if syn in t:
            if isinstance(mapped, list):
                for m in mapped:
                    matches.add(m)
            else:
                matches.add(mapped)
    return matches

def extract_skills_universal(text):
    """
    Extract skills from free text (JD or resume). Works for technical and non-technical roles.
    Returns sorted list of detected skill strings (canonical where possible).
    """
    t = normalize_text(text)
    detected = set()

    # 1) phrase skills (multiword) - check first to avoid splitting
    for phrase in PHRASE_SKILLS:
        if phrase in t:
            detected.add(phrase)

    # 2) canonical single-word skills & technical skills
    for s in ALL_SKILLS_CANONICAL:
        # use word boundary so "react" doesn't match "reactive" unintentionally
        if re.search(r'\b' + re.escape(s.lower()) + r'\b', t):
            detected.add(s.lower())

    # 3) synonyms mapping
    syn_matches = expand_synonyms_in_text_set(t)
    detected.update(syn_matches)

    # 4) heuristics: common verbs/nouns indicating function
    heuristics = {
        "manage": ["management","manage","manager","managing"],
        "lead": ["lead","led","leading","leadership"],
        "train": ["train","training","coaching","coach"],
        "sell": ["sell","selling","sales","upsell","merchandising","retail"]
    }
    for key, variants in heuristics.items():
        for v in variants:
            if re.search(r'\b' + re.escape(v) + r'\b', t):
                # map to friendly canonical tokens if appropriate
                if key == "sell":
                    detected.add("sales")
                elif key == "lead":
                    detected.add("leadership")
                elif key == "train":
                    detected.add("training")
                elif key == "manage":
                    detected.add("management")
                break

    # 5) cleanup: normalize near-duplicates, produce readable tokens
    final = set()
    for s in detected:
        s_norm = s.lower().strip()
        # map short forms to full phrase if in synonyms (reverse map)
        final.add(s_norm)

    return sorted(final)

# ---------------- Seniority detection multiplier ----------------
def detect_seniority_multiplier(text):
    t = normalize_text(text)
    mult = 1.0
    for kw, m in SENIORITY_KEYWORDS.items():
        if re.search(r'\b' + re.escape(kw) + r'\b', t):
            if m > mult:
                mult = m
    return mult

# ---------------- Scoring (improved, domain-agnostic) ----------------
def compute_score_improved(jd_skills, cand_skills, years, full_text):
    """
    Domain-agnostic scoring:
      - If JD skills present: skill_pct = matched/required
      - extras give small bonus
      - exp_pct computed relative to baseline (3 years) with seniority multiplier
      - final weighted: 0.7 skills_effective + 0.3 exp_pct
    Returns: final (0-100), skill_pct, exp_pct, extras_list
    """
    jd = [s.lower() for s in (jd_skills or [])]
    cand = [s.lower() for s in (cand_skills or [])]

    # experience handling
    seniority = detect_seniority_multiplier(full_text)
    exp_pct = min(100.0, round((years / 3.0) * 100.0 * seniority, 2)) if isinstance(years, (int,float)) and years >= 0 else 0.0

    if not jd:
        # no JD skills: base final primarily on experience (and seniority)
        final = round(exp_pct, 2)
        return final, 0.0, exp_pct, []

    matched = set(jd) & set(cand)
    extras = [s for s in cand if s not in jd]

    skill_pct = round(len(matched) / max(1, len(jd)) * 100.0, 2)

    # extras bonus (small): 1.5% per extra up to 8%
    extra_bonus = min(8.0, len(extras) * 1.5)

    skill_effective = min(100.0, skill_pct + extra_bonus)

    final = round((0.7 * (skill_effective / 100.0) + 0.3 * (exp_pct / 100.0)) * 100.0, 2)
    return final, skill_pct, exp_pct, extras

# ---------------- Build human-friendly explain output ----------------
def build_explain_improved(jd_skills, cand_skills, years, final, skill_pct, exp_pct, extras):
    jd = [s.lower() for s in (jd_skills or [])]
    cand = [s.lower() for s in (cand_skills or [])]

    matched = sorted(list(set(jd) & set(cand)))
    missing = sorted(list(set(jd) - set(cand)))
    extras = extras or []

    analysis_paragraphs = []
    if jd:
        analysis_paragraphs.append(
            f"You matched **{len(matched)} out of {len(jd)} required skills ({skill_pct}%)** — this is the main driver of the score."
        )
    else:
        analysis_paragraphs.append("No explicit required skills detected in the JD; scoring primarily uses experience, seniority clues, and detected role-related skills.")

    if isinstance(years, (int,float)) and years > 0:
        analysis_paragraphs.append(f"Estimated experience span: **{years} years** — contributing approximately **{exp_pct}%** to the score.")
    else:
        analysis_paragraphs.append("Experience could not be reliably determined from the resume text; consider adding explicit dates or duration.")

    if missing:
        analysis_paragraphs.append(f"The resume is missing these JD-specified skills: **{', '.join(missing)}**.")
    else:
        if jd:
            analysis_paragraphs.append("All required JD skills were found in the resume.")
        else:
            analysis_paragraphs.append("No specific JD skills were provided to compare against.")

    if extras:
        analysis_paragraphs.append(f"Additional detected skills not listed in the JD (small positive effect): {', '.join(extras)}.")

    # suggestions
    suggestions = []
    if jd:
        if skill_pct < 50:
            suggestions.append("Add concise bullets demonstrating the missing skills in project or role sections; mention the technologies or methods used.")
            suggestions.append("Add a short 'Key Projects' or 'Relevant Experience' area listing specific tasks that used the missing skills.")
        elif skill_pct < 80:
            suggestions.append("Good skills overlap — strengthen by quantifying achievements (metrics, scale, impact).")
        else:
            suggestions.append("Strong skills alignment — consider highlighting leadership or cross-functional impact.")

    if exp_pct < 50:
        suggestions.append("If years are low, emphasise project-level outcomes and demonstrated impact to compensate.")
    elif exp_pct < 80:
        suggestions.append("Consider adding short context sentences on responsibilities to indicate depth of experience.")

    analysis_text = "\n\n".join(analysis_paragraphs)
    return {
        "matched": matched,
        "missing": missing,
        "match_pct": round(skill_pct, 2) if skill_pct is not None else None,
        "exp_pct": round(exp_pct, 2) if exp_pct is not None else 0.0,
        "final": round(final, 2),
        "analysis": analysis_text,
        "suggestions": suggestions,
        "extras": extras
    }

# ---------------- Session State ----------------
if "samples" not in st.session_state:
    st.session_state.samples = []

# ---------------- UI (unchanged look) ----------------
st.markdown('<h1 class="main-title">NeonHire — Smart Resume Screener</h1>', unsafe_allow_html=True)

col1, col2 = st.columns([2,1])

with col1:
    st.markdown('<div class="left">', unsafe_allow_html=True)
    st.subheader("Job Description")
    jd_file = st.file_uploader("Upload JD", type=["txt", "pdf", "docx"])
    jd_text = st.text_area("Or paste JD", height=220)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="right">', unsafe_allow_html=True)
    st.subheader("Required Skills")
    skills_override = st.text_input("Comma-separated skills (optional)")
    st.subheader("Sample Data")
    if st.button("Load sample resumes"):
        loaded=[]
        for fn in os.listdir(DATA_DIR):
            with open(os.path.join(DATA_DIR,fn),"rb") as f:
                loaded.append({"name":fn,"bytes":f.read()})
        st.session_state.samples=loaded
        st.success("Sample resumes loaded.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="resumes">', unsafe_allow_html=True)
st.subheader("Upload Resumes")
uploaded = st.file_uploader("Upload resumes", type=["pdf","docx","txt"], accept_multiple_files=True)
st.markdown("</div>", unsafe_allow_html=True)

# Always disable anonymization (hidden)
anonym = False
export = True


# Toggle between old/improved scoring (backwards-compatible)
use_improved_toggle = True

run = st.button("Run Screening")

# ---------------- Processing ----------------
def get_jd_raw():
    if jd_file:
        try:
            return extract_text(jd_file)
        except:
            return ""
    return (jd_text or "").strip()

if run:
    jd_raw = get_jd_raw()
    if not jd_raw and not skills_override:
        st.error("Please provide JD text or comma-separated required skills.")
        st.stop()

    # Determine JD skills: prefer explicit override, else extract universally
    if skills_override and skills_override.strip():
        jd_skills = [s.strip().lower() for s in skills_override.split(",") if s.strip()]
    else:
        jd_skills = extract_skills_universal(jd_raw)

    # collect resumes
    resumes = []
    if uploaded:
        for u in uploaded:
            try:
                resumes.append({"name":u.name, "bytes": u.read()})
            except:
                pass
    if st.session_state.samples:
        resumes.extend(st.session_state.samples)

    if not resumes:
        st.error("No resumes uploaded or loaded.")
        st.stop()

    results = []
    for r in resumes:
        name = r.get("name") or "unknown"
        raw = r.get("bytes") or b""

        # extract text robustly
        try:
            low = name.lower()
            if low.endswith(".pdf"):
                text = extract_text_from_pdf(raw)
            elif low.endswith(".docx"):
                text = extract_text_from_docx(raw)
            else:
                try:
                    text = raw.decode("utf-8", "ignore")
                except:
                    text = ""
        except Exception:
            try:
                text = raw.decode("utf-8", "ignore")
            except:
                text = ""

        shown = anonymize(text) if anonym else text
        years = find_years(text)
        cand_skills = extract_skills_universal(text)

        # compute score (toggle)
        if use_improved_toggle:
            final, skill_pct, exp_pct, extras = compute_score_improved(jd_skills, cand_skills, years, text)
        else:
            # fallback to a simpler calculation (older style)
            # compute match_ratio defensively
            if jd_skills:
                match_ratio = len(set(jd_skills) & set(cand_skills)) / max(1, len(jd_skills))
            else:
                match_ratio = 0.0
            exp_ratio = min(1.0, years / 3.0) if isinstance(years, (int,float)) else 0.0
            final = round((0.7 * match_ratio + 0.3 * exp_ratio) * 100, 2)
            # provide skill_pct and exp_pct for explain builder
            skill_pct = round(match_ratio * 100.0, 2)
            exp_pct = round(exp_ratio * 100.0, 2)
            extras = [s for s in cand_skills if s not in jd_skills]

        # build explanation object
        explain_struct = build_explain_improved(jd_skills, cand_skills, years, final, skill_pct, exp_pct, extras)

        results.append({
            "file": name,
            "years": int(years) if isinstance(years, (int,float)) else 0,
            "skills": cand_skills,
            "final": explain_struct["final"],
            "match_pct": explain_struct["match_pct"],
            "exp_pct": explain_struct["exp_pct"],
            "matched": explain_struct["matched"],
            "missing": explain_struct["missing"],
            "suggestions": explain_struct["suggestions"],
            "analysis": explain_struct["analysis"],
            "extras": explain_struct["extras"],
            "snippet": shown[:1200]
        })

    df = pd.DataFrame(results).sort_values("final", ascending=False).reset_index(drop=True)

    st.subheader("Results")
    st.dataframe(df[["file","years","skills","final"]])

    # Candidate cards grid
    st.markdown('<div class="grid-cards">', unsafe_allow_html=True)
    for r in results:
        score = r["final"]
        tint = (
            "linear-gradient(90deg,#00ffcc,#00f0ff)" if score>=75 else
            "linear-gradient(90deg,#ffd166,#ff7ab6)" if score>=45 else
            "linear-gradient(90deg,#ff5c7c,#ff3d3d)"
        )

        st.markdown(f"""
        <div class="card">
            <div style="display:flex;justify-content:space-between;align-items:center">
                <div>
                    <strong>{r['file']}</strong>
                    <div class="small-muted">Years: {r['years']} • Skills: {', '.join(r['skills']) if r['skills'] else '—'}</div>
                </div>
                <span class="score-badge" style="background:{tint}">{score}</span>
            </div>
            <div class="score-bar"><div class="score-fill" style="width:{score}%;background:{tint}"></div></div>
        """, unsafe_allow_html=True)

        # badges: matched / missing / extras
        badges_html = ""
        for s in r["matched"]:
            badges_html += f'<span class="badge badge-matched">{s}</span>'
        for s in r["missing"]:
            badges_html += f'<span class="badge badge-missing">{s}</span>'
        for s in r["extras"]:
            badges_html += f'<span class="badge badge-extra">{s}</span>'

        if badges_html:
            st.markdown(f"<div style='margin-top:10px'>{badges_html}</div>", unsafe_allow_html=True)

        # Expander: Mixed style (quick summary + natural-language analysis + suggestions)
        with st.expander("Explain score ▼"):
            # Quick Summary
            st.markdown("### 🔹 Quick summary")
            if r["match_pct"] is None:
                st.info("No JD skills were provided — match percentage unavailable.")
            else:
                st.markdown(f"- **Skill Match:** `{r['match_pct']}%`")
                st.markdown(f"- **Experience:** `{r['exp_pct']}%`")
                st.markdown(f"- **Total Score:** `{r['final']}%`")

            # Natural language analysis
            st.markdown("### 🧠 Natural-language analysis")
            if r.get("analysis"):
                st.write(r["analysis"])
            else:
                st.write("No analysis available.")

            # Suggestions
            st.markdown("### 💡 Suggestions")
            if r["suggestions"]:
                for s in r["suggestions"]:
                    st.write(f"- {s}")
            else:
                st.write("Strong profile — no immediate suggestions.")

        # Snippet preview
        st.markdown(f"""
            <div style="margin-top:10px"><strong>Snippet:</strong>
            <div class="preview">{r['snippet']}</div></div></div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    if export:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv_bytes, file_name="results.csv", mime="text/csv")
