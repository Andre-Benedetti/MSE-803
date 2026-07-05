import os
import tkinter as tk
from tkinter import filedialog
from groq import Groq
from docx import Document
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from dotenv import load_dotenv

load_dotenv()

class GroqCVClient:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key) if api_key else None

    def get_analysis_and_revised_cv(self, cv_text):
        """Requests both advice and the full revised CV text."""
        prompt = f"""
        You are a NZ tech career expert. Analyze this CV and return two sections:
        
        SECTION 1: 3 actionable recommendations to improve it for a Software Engineering role.
        SECTION 2: A complete, improved version of the CV incorporating those changes.
        
        CV Content:
        {cv_text}
        """
        
        completion = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        return completion.choices[0].message.content

class CVAnalyzer:
    def __init__(self, client):
        self.client = client

    def save_to_pdf(self, text, output_path):
        """Saves text to PDF with proper wrapping and line breaks."""
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = [Paragraph("Career Advice & Revised CV", styles['Title']), Spacer(1, 12)]
        
        # Split by double newline to create paragraphs
        for line in text.split('\n'):
            if line.strip():
                story.append(Paragraph(line, styles['Normal']))
                story.append(Spacer(1, 6))
        
        doc.build(story)

    def save_to_docx(self, text, output_path):
        """Saves the revised CV text into a new .docx file."""
        doc = Document()
        doc.add_heading('Revised CV', 0)
        doc.add_paragraph(text)
        doc.save(output_path)

if __name__ == "__main__":
    API_KEY = "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"
    analyzer = CVAnalyzer(GroqCVClient(API_KEY))

    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select your CV", filetypes=[("Docx/Txt", "*.docx *.txt")])

    if file_path:
        # Extract existing content
        doc = Document(file_path) if file_path.endswith('.docx') else open(file_path, 'r').read()
        content = "\n".join([p.text for p in doc.paragraphs]) if file_path.endswith('.docx') else doc
        
        # Get response
        print("Analyzing and rewriting... please wait.")
        result = analyzer.client.get_analysis_and_revised_cv(content)
        
        # Save files
        base_dir = os.path.dirname(file_path)
        analyzer.save_to_pdf(result, os.path.join(base_dir, "CV_Feedback.pdf"))
        analyzer.save_to_docx(result, os.path.join(base_dir, "Revised_CV.docx"))
        
        print(f"Done! Files saved in: {base_dir}")