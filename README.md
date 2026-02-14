# 🚀 IntelliGrade AI – AI-Powered Code Evaluation System

IntelliGrade AI is a hybrid static + runtime + AI-powered grading engine designed to automate coding assignment evaluation.

## 🔥 Features

- Automated test case execution
- Time complexity analysis (AST-based)
- Runtime benchmarking
- Code quality scoring (Radon)
- Readability & best practices checks
- Token-level plagiarism detection
- Radar performance visualization
- PDF evaluation report generation
- AI-powered code review (OpenAI)

## 🛠 Tech Stack

- Python
- Streamlit
- Radon
- OpenAI API
- ReportLab
- Matplotlib

## 📊 How It Works

1. Upload a Python file.
2. The engine runs test cases.
3. Static analysis evaluates complexity & quality.
4. Performance metrics are calculated.
5. A final score and confidence score are generated.
6. AI provides smart improvement suggestions.

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run ai_code_evaluator.py
