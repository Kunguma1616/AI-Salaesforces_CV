# app.py - COMPLETE AND CORRECTED VERSION
import streamlit as st
from pdfminer.high_level import extract_text
import re  # Using regex for keyword matching
from dotenv import load_dotenv, find_dotenv
import os
from pathlib import Path
import base64
from typing import List
from pydantic import BaseModel, Field
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# LangChain imports
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# Azure
from azure.storage.blob import BlobServiceClient
import uuid

# PDF creation
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, HRFlowable, Table, TableStyle, PageBreak
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from io import BytesIO

# Salesforce imports
from simple_salesforce import Salesforce
from simple_salesforce.exceptions import SalesforceAuthenticationFailed, SalesforceGeneralError

# ========= LOAD ENVIRONMENT ==========
env_path = find_dotenv()
if env_path:
    load_dotenv(env_path)
else:
    script_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
    env_file = script_dir / ".env"
    if env_file.exists():
        load_dotenv(env_file)

api_key = os.getenv("GROQ_API_KEY")
MODEL_ID = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


# ========= STRUCTURED OUTPUT SCHEMA ==========
class CriterionScore(BaseModel):
    criterion_name: str = Field(description="Name of the evaluation criterion")
    score: float = Field(description="Score out of 5", ge=1.0, le=5.0)
    explanation: str = Field(description="Detailed explanation for the score")


class ResumeAnalysisOutput(BaseModel):
    job_category: str = Field(description="Classified job category (e.g., 'AI/Machine Learning', 'Marketing', 'Sales', 'Engineering', etc.)")
    job_subcategory: str = Field(description="More specific role classification")
    criteria_scores: List[CriterionScore] = Field(description="List of 5-7 key criteria")
    pros: List[str] = Field(description="Strengths of the resume")
    cons: List[str] = Field(description="Weaknesses or missing areas")
    overall_assessment: str = Field(description="Overall match summary")
    key_requirements_met: List[str] = Field(description="Requirements satisfied")
    key_requirements_missing: List[str] = Field(description="Requirements missing")
    skills_identified: List[str] = Field(description="All skills extracted from resume")
    skills_relevant: List[str] = Field(description="Skills related to job description")
    skills_missing: List[str] = Field(description="Required job skills missing in resume")


# ========= PAGE CONFIG ==========
st.set_page_config(page_title="Aspect AI Resume Analyzer", page_icon="✅", layout="wide")

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()

LOGO_FILE = str(SCRIPT_DIR / "C:images.png")


# ========= CSS STYLING ==========
st.markdown("""
<style>
    :root {
        --color-primary: #27549D;
        --color-dark-blue: #0f1e33;
        --color-secondary: #7099DB;
        --color-accent: #F1FF24;
        --text-light: #0f172a; /* CHANGED from white to dark */
        --text-dark: #0f172a;
        --card-bg: rgba(255, 255, 255, 0.98);
        --glass-bg: rgba(255, 255, 255, 0.12);
    }
    html, body, [data-testid="stAppViewContainer"] {
        /* CHANGED from blue gradient to yellow gradient */
        background: linear-gradient(135deg, #FFFDE7 0%, #FFF59D 100%) !important;
    }
    [data-testid="stAppViewContainer"] > .main, .block-container {
        background: transparent !important;
    }
    .stApp {
        color: var(--text-light) !important; /* This now uses the dark text color */
        font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, 'Helvetica Neue', Arial, 'Noto Sans';
    }
    .hero-header {
        background: rgba(255, 255, 255, 0.6); /* CHANGED from glass-bg to be more opaque */
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2.5rem;
        margin: 2rem 0;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.15); /* Softened shadow */
        display: flex; flex-direction: column; align-items: center; gap: 1rem;
    }
    .hero-header img { max-width: 150px; margin-bottom: 1rem; }
    .hero-header h1 { 
        color: var(--color-primary); /* CHANGED from accent (yellow) to primary (blue) */
        font-size: 3rem; 
        font-weight: 900; 
        margin: 0; 
    }
    .hero-header p { 
        color: #333; /* CHANGED from light text */
        font-size: 1.15rem; 
        margin-top: 0.5rem; 
    }

    .stForm, .score-container, .report-container, .white-card {
        background: var(--card-bg) !important; color: var(--text-dark) !important;
        border-radius: 20px; padding: 2rem; box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    }
    .report-container h2, .white-card h2, .stForm h2,
    .report-container h3, .white-card h3, .stForm h3 {
        color: var(--color-primary) !important;
    }

    .score-card {
        background: linear-gradient(135deg, var(--color-primary) 0%, var(--color-secondary) 100%) !important;
        color: #FFFFFF !important; padding: 2rem !important; border-radius: 15px !important;
        text-align: center !important; margin: 1rem 0 !important; box-shadow: 0 10px 25px rgba(39,84,157,0.4) !important;
    }
    .score-card h3 { margin: 0; font-size: 1.05rem; opacity: 0.9; }
    .score-card h2 { margin: 0.5rem 0 0 0; font-size: 3rem; font-weight: 900; color: var(--color-accent); }
    .score-card p { margin: 0.5rem 0 0 0; font-size: 0.95rem; opacity: 0.85; }

    .stDownloadButton > button {
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%) !important; color: white !important;
        border: none !important; border-radius: 50px !important; padding: 0.85rem 2.2rem !important;
        font-weight: 800 !important; box-shadow: 0 8px 20px rgba(34, 197, 94, 0.45) !important;
        font-size: 1.1rem !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, var(--color-primary) 0%, var(--color-secondary) 100%) !important;
        color: white !important; border: none !important; border-radius: 50px !important;
        padding: 0.85rem 2.2rem !important; font-weight: 800 !important;
    }
</style>
""", unsafe_allow_html=True)


# ========= HELPERS ==========
def get_image_as_base64(file):
    try:
        with open(file, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None


def extract_pdf_text(uploaded_file):

    try:
        return extract_text(uploaded_file)
    except:
        st.error("Error reading PDF")
        return None


# --- ATS SCORE FIX: NEW FUNCTION ---
def calculate_keyword_ats_score(resume_text, job_desc):
    """
    Calculates a simple keyword-based ATS score.
    It checks what percentage of keywords from the job description
    are present in the resume.
    """
    try:
        # Extract unique words (lowercase, alphanumeric) from job description
        jd_keywords = set(re.findall(r'\b\w+\b', job_desc.lower()))
        
        if not jd_keywords:
            return 0.0  # No keywords in job description to match against

        # Extract unique words from resume
        resume_keywords = set(re.findall(r'\b\w+\b', resume_text.lower()))
        
        # Find common keywords
        matching_keywords = jd_keywords.intersection(resume_keywords)
        
        # Calculate match percentage
        score = len(matching_keywords) / len(jd_keywords)
        return score
    except Exception as e:
        st.error(f"Error in keyword calculation: {e}")
        return 0.0 # Return 0 on any error


# ========= STRUCTURED ANALYSIS WITH CLASSIFICATION ==========
def get_structured_report(resume: str, job_desc: str) -> ResumeAnalysisOutput:
    system_prompt = """
You are an elite HR analyst and resume evaluator with expertise in recruitment, 
competency-based evaluation, and job-market benchmarking.

You MUST always return structured JSON that adheres exactly to the provided schema.

Your responsibilities:

1. **Job Classification**
   - Identify the correct job category (AI/ML, Data Science, Engineering, IT, Product, Marketing, Sales, Finance, HR, etc.)
   - Identify a specific job subcategory (e.g., “Machine Learning Engineer”, “Full-Stack Developer”, “FP&A Analyst”, 
     “Digital Marketing Specialist”).

2. **Evaluation Criteria (7–10 criteria)**
   Choose criteria that are *most relevant* to the classified job category.  
   Examples:  
   - Technical Skills Depth  
   - Domain Knowledge  
   - Experience Relevance  
   - Communication Skills  
   - Leadership & Ownership  
   - Problem-Solving Ability  
   - Project Delivery / Impact  
   - Tooling / Tech Stack Fit  

   For each criterion:
   - Assign a score from 1–5  
   - Provide a minimum **6-sentence justification** explaining:
         • What the resume demonstrates  
         • What the job description requires  
         • Gaps or differences  
         • Strengths  
         • Impact of missing items  
         • Real-world interpretation  
   Explanations must be detailed and role-specific.

3. **Pros & Cons**
   - Provide 6–10 strengths backed by resume evidence.
   - Provide 6–10 weaknesses tied to job requirements.

4. **Requirements Met & Missing**
   - Compare the resume directly against the job description.  
   - Use short, clear bullet points (no long paragraphs).

5. **Skills Extraction**
   - Extract **ALL skills** from the resume  
   - Identify skills relevant to the job description  
   - Identify missing skills that the job requires  

6. **Overall Assessment**
   - Provide a detailed, HR-style summary (8–12 sentences)  
   - Reflect overall fit, risks, readiness, and hiring recommendation language  
   - Explain the meaning of the ATS vs AI score  

THE MOST IMPORTANT RULE:  
Always return strictly valid JSON matching the schema.
"""

    llm = ChatGroq(api_key=api_key, model=MODEL_ID, temperature=0.2)

    structured_llm = llm.with_structured_output(
        ResumeAnalysisOutput,
        method="function_calling",
        include_raw=False
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"""
Analyze the following resume against the job description.

Ensure:
- role classification
- deep multi-criteria evaluation
- strong HR justification
- bullet-pointed strengths/weaknesses
- requirements matched and missing
- skills extracted
- full JSON output

JOB DESCRIPTION:
{job_desc}

RESUME:
{resume}

Return valid JSON ONLY.
""")
    ]

    return structured_llm.invoke(messages)


# ========= SIMPLE SCORE VISUALIZATION ==========
def create_simple_score_chart(ats_score, ai_score):
    """Create a clean, professional bar chart for scores"""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    categories = ['ATS Keyword\nMatch', 'AI Evaluation\nScore'] # Updated label
    scores = [ats_score * 100, ai_score * 100]
    colors_list = ['#27549D', '#7099DB']
    
    bars = ax.bar(categories, scores, color=colors_list, width=0.5)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_ylabel('Match Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_title('Resume Match Analysis', fontsize=16, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf


def create_skills_table_chart(relevant, missing):
    """Create a simple table visualization for skills"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    max_rows = max(len(relevant), len(missing))
    table_data = [['Skills Matched', 'Skills Missing']]
    
    for i in range(max_rows):
        row = [
            relevant[i] if i < len(relevant) else '',
            missing[i] if i < len(missing) else ''
        ]
        table_data.append(row)
    
    table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                       colWidths=[0.5, 0.5])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    for i in range(2):
        cell = table[(0, i)]
        cell.set_facecolor('#27549D')
        cell.set_text_props(weight='bold', color='white')
    
    for i in range(1, len(table_data)):
        table[(i, 0)].set_facecolor('#e8f5e9')
        table[(i, 1)].set_facecolor('#ffebee')
    
    plt.title('Skills Analysis', fontsize=16, fontweight='bold', pad=20)
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf


# ========= CLEAN TEXT FOR PDF ==========
def clean_text_for_pdf(text):
    """Remove all markdown symbols and clean text for PDF"""
    # Remove markdown bold
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    # Remove single asterisks
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    # Remove bullet points
    text = re.sub(r'^[\s]*[-•*]\s+', '', text, flags=re.MULTILINE)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ========= PERFECT PDF BUILDER ==========
def build_professional_pdf(*, logo_path: str | None, candidate_name: str, 
                           job_category: str, job_subcategory: str,
                           ats_score: float, ai_score: float, 
                           analysis: ResumeAnalysisOutput,
                           chart_buffer: BytesIO, skills_chart_buffer: BytesIO) -> bytes:
    """Build a professional, clean PDF report for HR teams"""
    
    buff = BytesIO()
    doc = SimpleDocTemplate(buff, pagesize=A4, leftMargin=50, rightMargin=50, 
                             topMargin=50, bottomMargin=50)
    
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle(
        "Title",
        parent=styles["Title"],
        fontSize=26,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#27549D"),
        spaceAfter=20,
        fontName='Helvetica-Bold'
    )
    
    section_header = ParagraphStyle(
        "SectionHeader",
        parent=styles["Heading1"],
        fontSize=16,
        textColor=colors.HexColor("#27549D"),
        spaceAfter=12,
        spaceBefore=20,
        fontName='Helvetica-Bold',
        borderWidth=0,
        borderColor=colors.HexColor("#27549D"),
        borderPadding=5
    )
    
    subsection_header = ParagraphStyle(
        "SubsectionHeader",
        parent=styles["Heading2"],
        fontSize=13,
        textColor=colors.HexColor("#0f1e33"),
        spaceAfter=8,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    body_text = ParagraphStyle(
        "BodyText",
        parent=styles["BodyText"],
        fontSize=10,
        leading=16,
        alignment=TA_JUSTIFY,
        spaceAfter=10,
        textColor=colors.HexColor("#1a1a1a"),
        fontName='Helvetica'
    )
    
    bullet_style = ParagraphStyle(
        "BulletStyle",
        parent=styles["BodyText"],
        fontSize=10,
        leading=16,
        leftIndent=20,
        spaceAfter=6,
        textColor=colors.HexColor("#1a1a1a"),
        fontName='Helvetica'
    )
    
    story = []
    
    # HEADER WITH LOGO
    if logo_path and Path(logo_path).exists():
        try:
            story.append(Image(logo_path, width=100, height=100))
            story.append(Spacer(1, 10))
        except:
            pass
    
    story.append(Paragraph("Resume Analysis Report", title_style))
    story.append(Spacer(1, 10))
    
    # Candidate Info Box
    info_data = [
        ['Candidate Name:', candidate_name],
        ['Job Category:', job_category],
        ['Specific Role:', job_subcategory],
    ]
    
    info_table = Table(info_data, colWidths=[2*inch, 4*inch])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#f0f4f8")),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor("#1a1a1a")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db"))
    ]))
    story.append(info_table)
    story.append(Spacer(1, 20))
    
    # DIVIDER
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#27549D")))
    story.append(Spacer(1, 20))
    
    # SCORE SUMMARY SECTION
    story.append(Paragraph("Score Summary", section_header))
    
    score_data = [
        ['Metric', 'Score', 'Percentage'],
        ['ATS Keyword Match', f'{ats_score:.2f}', f'{ats_score*100:.1f}%'], # Updated label
        ['AI Evaluation Score', f'{ai_score:.2f}', f'{ai_score*100:.1f}%']
    ]
    
    score_table = Table(score_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
    score_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#27549D")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#f9fafb")),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor("#d1d5db")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
    ]))
    story.append(score_table)
    story.append(Spacer(1, 20))
    
    # SCORE CHART
    story.append(Image(chart_buffer, width=5.5*inch, height=2.8*inch))
    story.append(Spacer(1, 20))
    
    # PAGE BREAK
    story.append(PageBreak())
    
    # EVALUATION CRITERIA
    story.append(Paragraph("Evaluation Criteria Analysis", section_header))
    story.append(Spacer(1, 10))
    
    if analysis.criteria_scores:
        for criterion in analysis.criteria_scores:
            story.append(Paragraph(f"{criterion.criterion_name}", subsection_header))
            story.append(Paragraph(f"<b>Score:</b> {criterion.score}/5.0", body_text))
            clean_explanation = clean_text_for_pdf(criterion.explanation)
            story.append(Paragraph(clean_explanation, body_text))
            story.append(Spacer(1, 10))
    
    # PROS AND CONS
    story.append(Spacer(1, 10))
    story.append(Paragraph("Strengths (Pros)", section_header))
    for i, pro in enumerate(analysis.pros, 1):
        clean_pro = clean_text_for_pdf(pro)
        story.append(Paragraph(f"{i}. {clean_pro}", bullet_style))
    
    story.append(Spacer(1, 15))
    story.append(Paragraph("Areas for Improvement (Cons)", section_header))
    for i, con in enumerate(analysis.cons, 1):
        clean_con = clean_text_for_pdf(con)
        story.append(Paragraph(f"{i}. {clean_con}", bullet_style))
    
    # PAGE BREAK
    story.append(PageBreak())
    
    # REQUIREMENTS ANALYSIS
    story.append(Paragraph("Requirements Analysis", section_header))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("Requirements Met", subsection_header))
    for i, req in enumerate(analysis.key_requirements_met, 1):
        clean_req = clean_text_for_pdf(req)
        story.append(Paragraph(f"{i}. {clean_req}", bullet_style))
    
    story.append(Spacer(1, 15))
    story.append(Paragraph("Requirements Missing", subsection_header))
    for i, req in enumerate(analysis.key_requirements_missing, 1):
        clean_req = clean_text_for_pdf(req)
        story.append(Paragraph(f"{i}. {clean_req}", bullet_style))
    
    # SKILLS ANALYSIS
    story.append(Spacer(1, 20))
    story.append(Paragraph("Skills Analysis", section_header))
    story.append(Spacer(1, 10))
    
    # Skills Summary Table
    skills_summary_data = [
        ['Metric', 'Count'],
        ['Total Skills Identified', str(len(analysis.skills_identified))],
        ['Relevant Skills Matched', str(len(analysis.skills_relevant))],
        ['Required Skills Missing', str(len(analysis.skills_missing))]
    ]
    
    skills_summary_table = Table(skills_summary_data, colWidths=[3*inch, 2*inch])
    skills_summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#27549D")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#f9fafb")),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor("#d1d5db"))
    ]))
    story.append(skills_summary_table)
    story.append(Spacer(1, 20))
    
    # Skills Chart
    story.append(Image(skills_chart_buffer, width=6*inch, height=3.5*inch))
    story.append(Spacer(1, 20))
    
    
    # OVERALL ASSESSMENT
    story.append(Paragraph("Overall Summary", section_header))
    clean_assessment = clean_text_for_pdf(analysis.overall_assessment)
    story.append(Paragraph(clean_assessment, body_text))
    
    # FOOTER
    story.append(Spacer(1, 30))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#d1d5db")))
    footer_style = ParagraphStyle(
        "Footer",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.HexColor("#6b7280"),
        alignment=TA_CENTER
    )
    story.append(Spacer(1, 10))
    story.append(Paragraph("Generated by Aspect AI Resume Analyzer | Confidential Document", footer_style))
    
    # BUILD PDF
    doc.build(story)
    buff.seek(0)
    return buff.read()


# ========= MARKDOWN CONVERTER ==========
def convert_structured_to_markdown(analysis: ResumeAnalysisOutput) -> str:
    md = []

    md.append("## Job Classification\n")
    md.append(f"**Category:** {analysis.job_category}")
    md.append(f"**Specific Role:** {analysis.job_subcategory}\n")

    md.append("## Evaluation Criteria Scores\n")
    if analysis.criteria_scores:
        for c in analysis.criteria_scores:
            md.append(f"### {c.criterion_name}")
            md.append(f"**Score:** {c.score}/5.0")
            md.append(f"{c.explanation}\n")

    md.append("## Pros (Strengths)")
    for p in analysis.pros:
        md.append(f"- {p}")

    md.append("\n## Cons (Areas for Improvement)")
    for c in analysis.cons:
        md.append(f"- {c}")

    md.append("\n## Requirements Met")
    for x in analysis.key_requirements_met:
        md.append(f"- {x}")

    md.append("\n## Requirements Missing")
    for x in analysis.key_requirements_missing:
        md.append(f"- {x}")

    md.append("\n## Skills Analysis")
    md.append(f"\n**Total Skills Identified:** {len(analysis.skills_identified)}")
    md.append(f"**Relevant Skills:** {len(analysis.skills_relevant)}")
    md.append(f"**Missing Required Skills:** {len(analysis.skills_missing)}")

    md.append("\n### Skills Matched:")
    for s in analysis.skills_relevant[:10]:
        md.append(f"- {s}")
    if len(analysis.skills_relevant) > 10:
        md.append(f"- ...and {len(analysis.skills_relevant) - 10} more")

    md.append("\n### Missing Required Skills:")
    for s in analysis.skills_missing:
        md.append(f"- {s}")

    md.append("\n## Overall Summary")
    md.append(analysis.overall_assessment)

    return "\n".join(md)


# ========= SESSION STATE ==========
if "form_submitted" not in st.session_state:
    st.session_state.form_submitted = False
if "resume" not in st.session_state:
    st.session_state.resume = ""
if "job_desc" not in st.session_state:
    st.session_state.job_desc = ""
if "candidate_name" not in st.session_state:
    st.session_state.candidate_name = ""
if "selected_trade" not in st.session_state:
    st.session_state.selected_trade = ""


# ========= UI HEADER ==========
logo_b64 = get_image_as_base64(LOGO_FILE)
if logo_b64:
    st.markdown(f"""
    <div class="hero-header">
        <img src="data:image/png;base64,{logo_b64}" alt="Aspect AI Logo">
        <h1>Aspect AI Resume Analyzer</h1>
        <p>Advanced AI-powered resume analysis with job classification</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="hero-header">
        <h1>Aspect AI Resume Analyzer</h1>
        <p>Advanced AI-powered resume analysis with job classification</p>
    </div>
    """, unsafe_allow_html=True)


# ========= MAIN FORM ==========
if not api_key:
    st.error("❌ GROQ_API_KEY missing from environment variables.")
    st.stop()

# Regex functions for validation
def is_valid_name(name):
    # Allows letters, spaces, hyphens, and apostrophes
    return bool(re.match(r"^[A-Za-z\s'-]+$", name.strip()))

def is_valid_email(email):
    return bool(re.match(r"^[\w\.-]+@[\w\.-]+\.\w+$", email.strip()))

# =============== SALESFORCE CONNECTION ==================
def get_salesforce_connection():
    """
    Connect to Salesforce using credentials from .env
    """
    try:
        sf = Salesforce(
            username=os.getenv("SF_USERNAME"),
            password=os.getenv("SF_PASSWORD"),
            security_token=os.getenv("SF_SECURITY_TOKEN"),
            domain=os.getenv("SF_DOMAIN", "login")
        )
        return sf
    except Exception as e:
        st.error(f"❌ Salesforce connection failed: {e}")
        return None


# =============== CREATE SALESFORCE RECORD ==================
def create_engineer_application_record(first_name, last_name, email, azure_url, trade):
    """
    Inserts a record into engineer_appication__c with First Name, Last Name, Email, Resume URL.
    """

    sf = get_salesforce_connection()
    if sf is None:
        return {"success": False, "error": "Salesforce connection missing"}

    try:
        # <--- This sends the exact trade value from the map
        result = sf.Engineer_Application__c.create({
            "First_Name__c": first_name,
            "Last_Name__c": last_name,
            "Email_Address__c": email,
            "Your_CV__c": azure_url,  # <--- !!! THIS IS THE FIX (Capital 'Y') !!!
            "Primary_Trade__c": trade
        })

        return {"success": True, "result": result}

    except Exception as e:
        return {"success": False, "error": str(e)}


# =================================================================
# <--- SALESFORCE FIX: THIS MAP IS NOW CORRECT ---
# =================================================================
#
# This map is built directly from your 'bala.csv' file.
# The Label (left) is what users see in the dropdown.
# The API Name (right) is the exact value sent to Salesforce.
#
TRADE_MAP = {
    'Access': 'Access',
    'Brickwork & Paving': 'Brickwork & Paving',
    'Carpentry': 'Carpentry',
    'Drainage': 'Drainage',
    'Drainage Survey': 'Drainage Survey',
    'Drains & Blockages (Soil Water)': 'Drains & Blockages (Soil Water)',
    'Drains & Blockages (Waste Water)': 'Drains & Blockages (Waste Water)',
    'Electrical': 'Electrical',
    'Electrical Testing': 'Electrical Testing',
    'Fencing, Decking & Cladding': 'Fencing, Decking & Cladding',
    'Gardening': 'Gardening',
    'Gas': 'Gas',
    'Gas Commercial': 'Gas Commercial',
    'Glazing': 'Glazing',
    'Heating, Ventilation, & Air Conditioning': 'Heating, Ventilation, & Air Conditioning',
    'Leak Detection - Drainage': 'Leak Detection - Drainage',
    'Leak Detection - Heating/Hot Water': 'Leak Detection - Heating/Hot Water',
    'Leak Detection - Multi': 'Leak Detection - Multi',
    'Leak Detection - Plumbing': 'Leak Detection - Plumbing',
    'Leak Detection - Roofing': 'Leak Detection - Roofing',
    'Locksmith': 'Locksmith',
    'Multi Skilled': 'Multi Skilled',
    'Painting & Decorating': 'Painting & Decorating',
    'Pest Control': 'Pest Control',
    'Plastering': 'Plastering',
    'Plumbing': 'Plumbing',
    'Project Manager': 'Project Manager',
    'Roofing': 'Roofing',
    'Tiling': 'Tiling',
    'Ventilation': 'Ventilation',
    'Waste Clearance': 'Waste Clearance',
    'Windows & Doors': 'Windows & Doors',
}
# =================================================================


# Only show the form if not submitted
if not st.session_state.form_submitted:

    st.title("Candidate Details Form")

    # --- ERROR FIX: All inputs are now inside the single form ---
    with st.form("input_form"):
        st.markdown("###  Candidate Information")
        first_name = st.text_input("First Name", placeholder="Enter first name")
        last_name = st.text_input("Last Name", placeholder="Enter last name")
        email = st.text_input("Email Address", placeholder="Enter email address")
        
        st.markdown("###  Upload Resume")
        resume_file = st.file_uploader("Upload Resume (PDF)", type="pdf")

        st.markdown("###  Job Description")
        job_desc_input = st.text_area("Paste job description here:", height=250)

        st.markdown("###  Traders")
        
        # <--- SALESFORCE FIX: Use the TRADE_MAP keys (labels) for the options
        selected_trade_label = st.selectbox(
            "Select Trade:", 
            options=list(TRADE_MAP.keys()), # <-- Show labels from map
            index=None, 
            placeholder="Select a trade"
        )
        
        # --- Submit Button ---
        submit_btn = st.form_submit_button("🚀 Analyze Resume", use_container_width=True)

    # Logic now runs *after* the form block, when the button is pressed
    if submit_btn:
        errors = []

        # Validation rules
        if not is_valid_name(first_name):
            errors.append("❌ Please enter a valid **First Name**.")
        if not is_valid_name(last_name):
            errors.append("❌ Please enter a valid **Last Name**.")
        if not is_valid_email(email):
            errors.append("❌ Please enter a valid **Email Address**.")
        if not job_desc_input:
            errors.append("⚠️ Please provide a **Job Description**.")
        if not resume_file:
            errors.append("⚠️ Please upload a **Resume PDF**.")
            
        # <--- SALESFORCE FIX: Validate using the label from the selectbox
        if not selected_trade_label:
             errors.append("⚠️ Please select a **Trade**.")
        

        # Display errors or run analysis
        if errors:
            for error in errors:
                st.error(error)
        else:
            # All inputs are valid, proceed with analysis
            st.session_state.first_name = first_name
            st.session_state.last_name = last_name
            st.session_state.email = email
            
            # <--- SALESFORCE FIX: Get the API Name from the map using the selected label
            selected_trade_api_name = TRADE_MAP.get(selected_trade_label)
            st.session_state.selected_trade = selected_trade_api_name
            
            text = extract_pdf_text(resume_file)
            
            if text:
                st.session_state.resume = text
                st.session_state.job_desc = job_desc_input
                st.session_state.candidate_name = f"{first_name} {last_name}".strip()
                st.session_state.form_submitted = True
                st.rerun()
            else:
                st.error("❌ Failed to extract text from PDF.")

else:
    # ========= ANALYSIS ==========
    with st.spinner("⚡ Analyzing resume... This may take a moment."):
        
        ats_score = calculate_keyword_ats_score(
            st.session_state.resume,
            st.session_state.job_desc
        )

        try:
            analysis = get_structured_report(
                st.session_state.resume,
                st.session_state.job_desc
            )
            print(analysis)
            # This score is the average from the AI's detailed criteria
            if analysis.criteria_scores:
                ai_score = sum(c.score for c in analysis.criteria_scores) / (len(analysis.criteria_scores) * 5)
            else:
                ai_score = 0.0 # Handle case with no criteria
                
            report_md = convert_structured_to_markdown(analysis)
            
        except Exception as e:
            st.error(f"❌ Analysis error: {str(e)}")
            st.exception(e) # Print full traceback
            if st.button("🔄 Try Again"):
                st.session_state.form_submitted = False
                st.rerun()
            st.stop()

    # Generate charts
    chart_buffer = create_simple_score_chart(ats_score, ai_score)
    skills_chart_buffer = create_skills_table_chart(
        analysis.skills_relevant[:15],
        analysis.skills_missing
    )

    # ========= JOB CLASSIFICATION DISPLAY ==========
    st.markdown('<div class="white-card">', unsafe_allow_html=True)
    st.markdown("## 🎯 Job Classification")
    col_cat1, col_cat2 = st.columns(2)
    with col_cat1:
        st.markdown(f"**Category:** {analysis.job_category}")
    with col_cat2:
        st.markdown(f"**Specific Role:** {analysis.job_subcategory}")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ========= SCORE CARDS ==========
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="score-card">
            <h3>📊 ATS Keyword Match</h3>
            <h2>{ats_score:.1%}</h2>
            <p>Resume-Job Description Match</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="score-card">
            <h3> AI Evaluation Score</h3>
            <h2>{ai_score:.1%}</h2>
            <p>Criteria-Based Assessment</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ========= FULL REPORT ==========
    st.markdown('<div class="report-container">', unsafe_allow_html=True)
    st.markdown("##  Detailed Analysis Report")
    st.markdown(report_md, unsafe_allow_html=True) # Allow HTML in markdown for safety
    st.markdown('</div>', unsafe_allow_html=True)

    # ========= PDF, UPLOAD, AND RESET LOGIC ==========

    chart_buffer_copy = BytesIO(chart_buffer.getvalue())
    skills_buffer_copy = BytesIO(skills_chart_buffer.getvalue())
    
    try:
        # --- STEP 1: Generate PDF ---
        pdf_bytes = build_professional_pdf(
            logo_path=LOGO_FILE,
            candidate_name=st.session_state.candidate_name,
            job_category=analysis.job_category,
            job_subcategory=analysis.job_subcategory,
            ats_score=ats_score,
            ai_score=ai_score,
            analysis=analysis,
            chart_buffer=chart_buffer_copy,
            skills_chart_buffer=skills_buffer_copy
        )

    except Exception as e:
        st.error(f"❌ Failed to generate PDF: {e}")
        st.exception(e) # Print full traceback for debugging
        st.stop() # <-- STOPS SCRIPT IF PDF FAILS, PREVENTING UPLOAD ERROR

    # --- STEP 2: Show Download Button ---
    st.download_button(
        "📄 Download PDF Report",
        data=pdf_bytes,
        file_name=f"Resume_Analysis_{st.session_state.candidate_name.replace(' ', '_')}_{analysis.job_category.replace('/', '_')}.pdf",
        mime="application/pdf",
        use_container_width=True
    )
    
    # --- STEP 3: Azure Upload + Salesforce (The single, combined block) ---
    connect_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container_name = os.getenv("AZURE_CONTAINER_NAME")
    sas_token = os.getenv("AZURE_SAS_TOKEN")  # <-- Your SAS token from .env

    if connect_str and container_name and sas_token:
        try:
            # Step 1 — Connect to Azure Storage
            blob_service_client = BlobServiceClient.from_connection_string(connect_str)
            container_client = blob_service_client.get_container_client(container_name)

            # Step 2 — Generate unique filename
            filename = f"{st.session_state.candidate_name.replace(' ', '_')}_{uuid.uuid4().hex[:6]}.pdf"
            blob_client = container_client.get_blob_client(filename)

            # Step 3 — Upload PDF to Azure
            with st.spinner("☁️ Uploading PDF to Azure..."):
                pdf_bytes_io = BytesIO(pdf_bytes)
                pdf_bytes_io.seek(0)
                blob_client.upload_blob(pdf_bytes_io, overwrite=True)

            # Step 4 — Build FULL public SAS URL
            azure_url = f"{blob_client.url}?{sas_token}"

            st.success(" PDF uploaded to Azure successfully!")
            st.info(f"Public Azure File URL:\n{azure_url}")

            # Step 5 — Push to Salesforce
            with st.spinner("Saving record to Salesforce..."):
                sf_result = create_engineer_application_record(
                    first_name=st.session_state.first_name,
                    last_name=st.session_state.last_name,
                    email=st.session_state.email,
                    azure_url=azure_url,  # <-- Public SAS URL
                    trade=st.session_state.selected_trade
                )

            if sf_result["success"]:
                st.success(f" Salesforce Record Created! ID: {sf_result['result']['id']}")
            else:
                st.error(f" Salesforce Error: {sf_result['error']}")

        except Exception as err:
            st.error(f" Azure Upload or Salesforce Error: {err}")
            st.exception(err)
    else:
        st.error(" Missing Azure configuration. Check .env file.")

    # --- STEP 4: Reset Button ---
    st.markdown("---")
    if st.button("🔄 Analyze Another Resume", use_container_width=True):
        # Clear all session state keys
        for key in list(st.session_state.keys()):
            st.session_state.pop(key)
        
        st.rerun()
        
