from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def generate_pdf_report():

    filename = "AI_Agent_Report.pdf"

    c = canvas.Canvas(filename, pagesize=letter)

    c.drawString(100, 750, "AI Agent Data Science Report")

    c.drawString(100, 700, "Model training completed successfully")

    c.save()

    return filename
