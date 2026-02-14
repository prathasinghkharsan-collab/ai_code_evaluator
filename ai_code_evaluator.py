import streamlit as st
import subprocess
import tempfile
import os
import json
import matplotlib.pyplot as plt
import ast
import time
import tokenize
from io import StringIO
from difflib import SequenceMatcher
from radon.complexity import cc_visit
from radon.metrics import mi_visit
from openai import OpenAI
from dotenv import load_dotenv
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# ---------------------------
# PAGE CONFIG
# ---------------------------

st.set_page_config(
    page_title="AI Coding Evaluator",
    page_icon="🚀",
    layout="wide"
)

# ---------------------------
# LOAD ENV
# ---------------------------

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

# ---------------------------
# CONFIG
# ---------------------------

WEIGHTS = {
    "correctness": 30,
    "time_efficiency": 20,
    "space_efficiency": 10,
    "code_quality": 15,
    "readability": 10,
    "edge_cases": 10,
    "best_practices": 5
}

BASE_TEST_CASES = [
    {"input": "abcabcbb", "expected": "3"},
    {"input": "", "expected": "0"},
    {"input": "bbbbb", "expected": "1"},
    {"input": "pwwkew", "expected": "3"},
]

def generate_edge_cases():
    return [
        {"input": "", "expected": "0"},
        {"input": "a"*1000, "expected": "1"},
        {"input": "abcdefghijklmnopqrstuvwxyz", "expected": "26"}
    ]

TEST_CASES = BASE_TEST_CASES + generate_edge_cases()

# ---------------------------
# EXECUTION
# ---------------------------

def run_python(code, input_data):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as tmp:
        tmp.write(code.encode())
        filename = tmp.name

    result = subprocess.run(
        ["python", filename],
        input=input_data,
        text=True,
        capture_output=True,
        timeout=3
    )

    os.unlink(filename)
    return result.stdout.strip()

def run_test_cases(code):
    passed = 0
    feedback = []

    for case in TEST_CASES:
        try:
            output = run_python(code, case["input"])
            if output == case["expected"]:
                passed += 1
            else:
                feedback.append(f"Failed for input '{case['input']}'")
        except:
            feedback.append(f"Runtime error for input '{case['input']}'")

    return passed, len(TEST_CASES), feedback

# ---------------------------
# ANALYSIS
# ---------------------------

def advanced_time_complexity(code):
    try:
        tree = ast.parse(code)

        class LoopVisitor(ast.NodeVisitor):
            def __init__(self):
                self.depth = 0
                self.max_depth = 0

            def visit_For(self, node):
                self.depth += 1
                self.max_depth = max(self.max_depth, self.depth)
                self.generic_visit(node)
                self.depth -= 1

            def visit_While(self, node):
                self.depth += 1
                self.max_depth = max(self.max_depth, self.depth)
                self.generic_visit(node)
                self.depth -= 1

        visitor = LoopVisitor()
        visitor.visit(tree)

        if visitor.max_depth >= 2:
            return 60, f"Nested loops detected (depth {visitor.max_depth})"
        elif visitor.max_depth == 1:
            return 85, "Single loop detected"
        else:
            return 95, "No loops detected"

    except:
        return 70, "Complexity analysis unavailable"

def benchmark_runtime(code):
    try:
        start = time.time()
        run_python(code, "abcabcbb")
        runtime = time.time() - start

        if runtime < 0.1:
            return 95, f"Fast execution ({runtime:.4f}s)"
        elif runtime < 0.5:
            return 80, f"Moderate execution ({runtime:.4f}s)"
        else:
            return 60, f"Slow execution ({runtime:.4f}s)"
    except:
        return 70, "Benchmark unavailable"

def estimate_space_complexity(code):
    if any(ds in code for ds in ["dict(", "{}", "[]", "set("]):
        return 80
    return 95

def evaluate_code_quality(code):
    try:
        complexity_blocks = cc_visit(code)
        max_complexity = max(block.complexity for block in complexity_blocks)
        maintainability = mi_visit(code, True)

        score = 100
        if max_complexity > 10:
            score -= 30
        if maintainability < 65:
            score -= 20

        return max(score, 0)
    except:
        return 60

def evaluate_readability(code):
    score = 100
    if '"""' not in code:
        score -= 20
    if "#" not in code:
        score -= 10
    return max(score, 0)

def evaluate_best_practices(code):
    if "if __name__" not in code:
        return 80
    return 100

def evaluate_edge_cases(code):
    if "if not" not in code:
        return 70
    return 100

# ---------------------------
# PLAGIARISM
# ---------------------------

def tokenize_code(code):
    tokens = []
    try:
        for tok in tokenize.generate_tokens(StringIO(code).readline):
            if tok.type in (tokenize.NAME, tokenize.OP):
                tokens.append(tok.string)
        return tokens
    except:
        return []

def advanced_plagiarism(code1, code2):
    return round(
        SequenceMatcher(None, tokenize_code(code1), tokenize_code(code2)).ratio() * 100,
        2
    )

# ---------------------------
# SCORING
# ---------------------------

def calculate_overall(scores):
    return round(sum(scores[k] * WEIGHTS[k] / 100 for k in scores))

def calculate_confidence(correctness, runtime, complexity):
    return round((correctness * 0.6) + (runtime * 0.2) + (complexity * 0.2))

# ---------------------------
# PDF REPORT
# ---------------------------

def generate_pdf_report(scores, overall, confidence):
    file_path = "evaluation_report.pdf"
    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("<b>AI Coding Evaluation Report</b>", styles['Title']))
    elements.append(Spacer(1, 0.4 * inch))
    elements.append(Paragraph(f"Overall Score: {overall}/100", styles['Normal']))
    elements.append(Paragraph(f"Confidence: {confidence}%", styles['Normal']))
    elements.append(Spacer(1, 0.3 * inch))

    for k, v in scores.items():
        elements.append(Paragraph(f"{k}: {v}/100", styles['Normal']))

    doc.build(elements)
    return file_path

# ---------------------------
# LEADERBOARD
# ---------------------------

def update_leaderboard(score):
    file = "leaderboard.json"

    if os.path.exists(file):
        with open(file, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(score)

    with open(file, "w") as f:
        json.dump(data, f)

    return sorted(data, reverse=True)[:5]

# ---------------------------
# UI
# ---------------------------

st.markdown("""
# 🚀 AI Coding Assignment Evaluator
### Hybrid Static + Runtime + AI-Powered Code Assessment Engine
""")

uploaded_file = st.file_uploader("Upload Python File", type=["py"])

if uploaded_file:
    code = uploaded_file.read().decode()

    with st.spinner("Running evaluation engine..."):

        passed, total, feedback = run_test_cases(code)
        correctness = int((passed / total) * 100)

        complexity_score, complexity_msg = advanced_time_complexity(code)
        runtime_score, runtime_msg = benchmark_runtime(code)

        scores = {
            "correctness": correctness,
            "time_efficiency": runtime_score,
            "space_efficiency": estimate_space_complexity(code),
            "code_quality": evaluate_code_quality(code),
            "readability": evaluate_readability(code),
            "edge_cases": evaluate_edge_cases(code),
            "best_practices": evaluate_best_practices(code)
        }

        overall = calculate_overall(scores)
        confidence = calculate_confidence(correctness, runtime_score, complexity_score)

    # Gradient Header
    st.markdown(f"""
    <h1 style='text-align:center;
    background: linear-gradient(90deg,#00ffcc,#0066ff);
    -webkit-background-clip:text;
    color:transparent;'>
    🎯 Overall Score: {overall}/100
    </h1>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("🔬 Confidence", f"{confidence}%")
    col2.metric("✅ Tests Passed", f"{passed}/{total}")
    col3.metric("⚡ Runtime Score", f"{runtime_score}/100")

    st.divider()

    st.subheader("📊 Performance Breakdown")

    for k, v in scores.items():
        st.progress(v / 100)
        st.write(f"**{k.replace('_',' ').title()}** — {v}/100")

    st.divider()

    with st.expander("🧠 Complexity & Runtime Details"):
        st.info(complexity_msg)
        st.info(runtime_msg)
        if feedback:
            st.warning("Failed Tests:")
            for f in feedback:
                st.write("-", f)

    st.subheader("📈 Radar View")
    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    labels = list(scores.keys())
    values = list(scores.values())
    values += values[:1]
    angles = [n / float(len(labels)) * 2 * 3.14159 for n in range(len(labels))]
    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.3)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])
    st.pyplot(fig)

    st.divider()

    # Leaderboard
    top_scores = update_leaderboard(overall)
    st.subheader("🏆 Leaderboard (Top 5)")
    for i, s in enumerate(top_scores):
        st.write(f"{i+1}. {s}/100")

    # PDF Download
    if st.button("📄 Download PDF Report"):
        pdf = generate_pdf_report(scores, overall, confidence)
        with open(pdf, "rb") as f:
            st.download_button("Download Report", f, file_name="evaluation_report.pdf")
