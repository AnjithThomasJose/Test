#!/usr/bin/env python3
"""
Convert Markdown report to PDF using markdown and reportlab.
"""
import markdown
import re
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Preformatted
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from html.parser import HTMLParser
import sys
import os

class HTMLStripper(HTMLParser):
    """Strip HTML tags from text."""
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text = []
    
    def handle_data(self, d):
        self.text.append(d)
    
    def get_text(self):
        return ''.join(self.text)

def strip_html(html):
    """Remove HTML tags from text."""
    s = HTMLStripper()
    s.feed(html)
    return s.get_text()

def convert_md_to_pdf(md_file, pdf_file):
    """Convert markdown file to PDF."""
    # Read markdown file
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert markdown to HTML
    html_content = markdown.markdown(
        md_content,
        extensions=['extra', 'codehilite', 'tables', 'toc']
    )
    
    # Create PDF document
    doc = SimpleDocTemplate(
        pdf_file,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18
    )
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=30
    )
    
    h1_style = ParagraphStyle(
        'CustomH1',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        spaceBefore=20
    )
    
    h2_style = ParagraphStyle(
        'CustomH2',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#34495e'),
        spaceAfter=10,
        spaceBefore=15
    )
    
    h3_style = ParagraphStyle(
        'CustomH3',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=colors.HexColor('#7f8c8d'),
        spaceAfter=8,
        spaceBefore=12
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        spaceAfter=6
    )
    
    code_style = ParagraphStyle(
        'CustomCode',
        parent=styles['Code'],
        fontSize=9,
        fontName='Courier',
        backColor=colors.HexColor('#f4f4f4'),
        leftIndent=10,
        rightIndent=10,
        spaceAfter=10,
        spaceBefore=10
    )
    
    # Parse HTML and convert to PDF elements
    lines = html_content.split('\n')
    in_code_block = False
    code_block_lines = []
    in_table = False
    table_rows = []
    
    for line in lines:
        line = line.strip()
        
        if not line:
            if not in_code_block and not in_table:
                elements.append(Spacer(1, 6))
            continue
        
        # Handle code blocks
        if '<pre>' in line or '<code class=' in line:
            in_code_block = True
            code_block_lines = []
            continue
        
        if '</pre>' in line or '</code>' in line:
            if code_block_lines:
                code_text = '\n'.join(code_block_lines)
                code_text = strip_html(code_text)
                elements.append(Preformatted(code_text, code_style))
                elements.append(Spacer(1, 10))
            in_code_block = False
            code_block_lines = []
            continue
        
        if in_code_block:
            code_block_lines.append(strip_html(line))
            continue
        
        # Handle tables
        if '<table>' in line:
            in_table = True
            table_rows = []
            continue
        
        if '</table>' in line:
            if table_rows:
                # Create table
                table = Table(table_rows)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                    ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#ddd')),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9f9f9')]),
                ]))
                elements.append(table)
                elements.append(Spacer(1, 12))
            in_table = False
            table_rows = []
            continue
        
        if in_table:
            if '<tr>' in line:
                current_row = []
            elif '</tr>' in line:
                if current_row:
                    table_rows.append(current_row)
            elif '<th>' in line or '<td>' in line:
                cell_text = strip_html(line)
                if cell_text:
                    current_row.append(cell_text)
            continue
        
        # Handle headings
        if line.startswith('<h1>'):
            text = strip_html(line)
            if 'RAG-Based Course Recommendation System' in text:
                elements.append(Paragraph(text, title_style))
            else:
                elements.append(Paragraph(text, h1_style))
            elements.append(Spacer(1, 12))
            continue
        
        if line.startswith('<h2>'):
            text = strip_html(line)
            elements.append(Paragraph(text, h2_style))
            elements.append(Spacer(1, 10))
            continue
        
        if line.startswith('<h3>'):
            text = strip_html(line)
            elements.append(Paragraph(text, h3_style))
            elements.append(Spacer(1, 8))
            continue
        
        # Handle horizontal rules
        if '<hr' in line:
            elements.append(Spacer(1, 20))
            continue
        
        # Handle lists
        if '<li>' in line:
            text = strip_html(line)
            text = '• ' + text if text else text
            elements.append(Paragraph(text, normal_style))
            continue
        
        # Handle blockquotes
        if '<blockquote>' in line:
            text = strip_html(line)
            quote_style = ParagraphStyle(
                'Quote',
                parent=normal_style,
                leftIndent=20,
                borderColor=colors.HexColor('#3498db'),
                borderWidth=0,
                borderPadding=(0, 0, 0, 4)
            )
            elements.append(Paragraph(text, quote_style))
            continue
        
        # Handle regular paragraphs
        text = strip_html(line)
        if text and not text.startswith('<'):
            # Clean up common markdown artifacts
            text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)  # Bold
            text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)  # Italic
            text = re.sub(r'`(.*?)`', r'<font name="Courier" size="9" backColor="#f4f4f4">\1</font>', text)  # Inline code
            
            elements.append(Paragraph(text, normal_style))
    
    # Build PDF
    print(f"📄 Converting {md_file} to PDF...")
    doc.build(elements)
    print(f"✅ PDF created successfully: {pdf_file}")

if __name__ == "__main__":
    md_file = "agents/scripts/RAG_Course_Recommendation_System_Report.md"
    pdf_file = "agents/scripts/RAG_Course_Recommendation_System_Report.pdf"
    
    if not os.path.exists(md_file):
        print(f"❌ Error: {md_file} not found")
        sys.exit(1)
    
    try:
        convert_md_to_pdf(md_file, pdf_file)
        print(f"\n📊 File size: {os.path.getsize(pdf_file) / 1024:.1f} KB")
    except Exception as e:
        print(f"❌ Error converting to PDF: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

Convert Markdown report to PDF using markdown and reportlab.
"""
import markdown
import re
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Preformatted
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from html.parser import HTMLParser
import sys
import os

class HTMLStripper(HTMLParser):
    """Strip HTML tags from text."""
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text = []
    
    def handle_data(self, d):
        self.text.append(d)
    
    def get_text(self):
        return ''.join(self.text)

def strip_html(html):
    """Remove HTML tags from text."""
    s = HTMLStripper()
    s.feed(html)
    return s.get_text()

def convert_md_to_pdf(md_file, pdf_file):
    """Convert markdown file to PDF."""
    # Read markdown file
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert markdown to HTML
    html_content = markdown.markdown(
        md_content,
        extensions=['extra', 'codehilite', 'tables', 'toc']
    )
    
    # Create PDF document
    doc = SimpleDocTemplate(
        pdf_file,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18
    )
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=30
    )
    
    h1_style = ParagraphStyle(
        'CustomH1',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        spaceBefore=20
    )
    
    h2_style = ParagraphStyle(
        'CustomH2',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#34495e'),
        spaceAfter=10,
        spaceBefore=15
    )
    
    h3_style = ParagraphStyle(
        'CustomH3',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=colors.HexColor('#7f8c8d'),
        spaceAfter=8,
        spaceBefore=12
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        spaceAfter=6
    )
    
    code_style = ParagraphStyle(
        'CustomCode',
        parent=styles['Code'],
        fontSize=9,
        fontName='Courier',
        backColor=colors.HexColor('#f4f4f4'),
        leftIndent=10,
        rightIndent=10,
        spaceAfter=10,
        spaceBefore=10
    )
    
    # Parse HTML and convert to PDF elements
    lines = html_content.split('\n')
    in_code_block = False
    code_block_lines = []
    in_table = False
    table_rows = []
    
    for line in lines:
        line = line.strip()
        
        if not line:
            if not in_code_block and not in_table:
                elements.append(Spacer(1, 6))
            continue
        
        # Handle code blocks
        if '<pre>' in line or '<code class=' in line:
            in_code_block = True
            code_block_lines = []
            continue
        
        if '</pre>' in line or '</code>' in line:
            if code_block_lines:
                code_text = '\n'.join(code_block_lines)
                code_text = strip_html(code_text)
                elements.append(Preformatted(code_text, code_style))
                elements.append(Spacer(1, 10))
            in_code_block = False
            code_block_lines = []
            continue
        
        if in_code_block:
            code_block_lines.append(strip_html(line))
            continue
        
        # Handle tables
        if '<table>' in line:
            in_table = True
            table_rows = []
            continue
        
        if '</table>' in line:
            if table_rows:
                # Create table
                table = Table(table_rows)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                    ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#ddd')),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9f9f9')]),
                ]))
                elements.append(table)
                elements.append(Spacer(1, 12))
            in_table = False
            table_rows = []
            continue
        
        if in_table:
            if '<tr>' in line:
                current_row = []
            elif '</tr>' in line:
                if current_row:
                    table_rows.append(current_row)
            elif '<th>' in line or '<td>' in line:
                cell_text = strip_html(line)
                if cell_text:
                    current_row.append(cell_text)
            continue
        
        # Handle headings
        if line.startswith('<h1>'):
            text = strip_html(line)
            if 'RAG-Based Course Recommendation System' in text:
                elements.append(Paragraph(text, title_style))
            else:
                elements.append(Paragraph(text, h1_style))
            elements.append(Spacer(1, 12))
            continue
        
        if line.startswith('<h2>'):
            text = strip_html(line)
            elements.append(Paragraph(text, h2_style))
            elements.append(Spacer(1, 10))
            continue
        
        if line.startswith('<h3>'):
            text = strip_html(line)
            elements.append(Paragraph(text, h3_style))
            elements.append(Spacer(1, 8))
            continue
        
        # Handle horizontal rules
        if '<hr' in line:
            elements.append(Spacer(1, 20))
            continue
        
        # Handle lists
        if '<li>' in line:
            text = strip_html(line)
            text = '• ' + text if text else text
            elements.append(Paragraph(text, normal_style))
            continue
        
        # Handle blockquotes
        if '<blockquote>' in line:
            text = strip_html(line)
            quote_style = ParagraphStyle(
                'Quote',
                parent=normal_style,
                leftIndent=20,
                borderColor=colors.HexColor('#3498db'),
                borderWidth=0,
                borderPadding=(0, 0, 0, 4)
            )
            elements.append(Paragraph(text, quote_style))
            continue
        
        # Handle regular paragraphs
        text = strip_html(line)
        if text and not text.startswith('<'):
            # Clean up common markdown artifacts
            text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)  # Bold
            text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)  # Italic
            text = re.sub(r'`(.*?)`', r'<font name="Courier" size="9" backColor="#f4f4f4">\1</font>', text)  # Inline code
            
            elements.append(Paragraph(text, normal_style))
    
    # Build PDF
    print(f"📄 Converting {md_file} to PDF...")
    doc.build(elements)
    print(f"✅ PDF created successfully: {pdf_file}")

if __name__ == "__main__":
    md_file = "agents/scripts/RAG_Course_Recommendation_System_Report.md"
    pdf_file = "agents/scripts/RAG_Course_Recommendation_System_Report.pdf"
    
    if not os.path.exists(md_file):
        print(f"❌ Error: {md_file} not found")
        sys.exit(1)
    
    try:
        convert_md_to_pdf(md_file, pdf_file)
        print(f"\n📊 File size: {os.path.getsize(pdf_file) / 1024:.1f} KB")
    except Exception as e:
        print(f"❌ Error converting to PDF: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

Convert Markdown report to PDF using markdown and reportlab.
"""
import markdown
import re
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Preformatted
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from html.parser import HTMLParser
import sys
import os

class HTMLStripper(HTMLParser):
    """Strip HTML tags from text."""
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text = []
    
    def handle_data(self, d):
        self.text.append(d)
    
    def get_text(self):
        return ''.join(self.text)

def strip_html(html):
    """Remove HTML tags from text."""
    s = HTMLStripper()
    s.feed(html)
    return s.get_text()

def convert_md_to_pdf(md_file, pdf_file):
    """Convert markdown file to PDF."""
    # Read markdown file
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert markdown to HTML
    html_content = markdown.markdown(
        md_content,
        extensions=['extra', 'codehilite', 'tables', 'toc']
    )
    
    # Create PDF document
    doc = SimpleDocTemplate(
        pdf_file,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18
    )
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=30
    )
    
    h1_style = ParagraphStyle(
        'CustomH1',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        spaceBefore=20
    )
    
    h2_style = ParagraphStyle(
        'CustomH2',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#34495e'),
        spaceAfter=10,
        spaceBefore=15
    )
    
    h3_style = ParagraphStyle(
        'CustomH3',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=colors.HexColor('#7f8c8d'),
        spaceAfter=8,
        spaceBefore=12
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        spaceAfter=6
    )
    
    code_style = ParagraphStyle(
        'CustomCode',
        parent=styles['Code'],
        fontSize=9,
        fontName='Courier',
        backColor=colors.HexColor('#f4f4f4'),
        leftIndent=10,
        rightIndent=10,
        spaceAfter=10,
        spaceBefore=10
    )
    
    # Parse HTML and convert to PDF elements
    lines = html_content.split('\n')
    in_code_block = False
    code_block_lines = []
    in_table = False
    table_rows = []
    
    for line in lines:
        line = line.strip()
        
        if not line:
            if not in_code_block and not in_table:
                elements.append(Spacer(1, 6))
            continue
        
        # Handle code blocks
        if '<pre>' in line or '<code class=' in line:
            in_code_block = True
            code_block_lines = []
            continue
        
        if '</pre>' in line or '</code>' in line:
            if code_block_lines:
                code_text = '\n'.join(code_block_lines)
                code_text = strip_html(code_text)
                elements.append(Preformatted(code_text, code_style))
                elements.append(Spacer(1, 10))
            in_code_block = False
            code_block_lines = []
            continue
        
        if in_code_block:
            code_block_lines.append(strip_html(line))
            continue
        
        # Handle tables
        if '<table>' in line:
            in_table = True
            table_rows = []
            continue
        
        if '</table>' in line:
            if table_rows:
                # Create table
                table = Table(table_rows)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                    ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#ddd')),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9f9f9')]),
                ]))
                elements.append(table)
                elements.append(Spacer(1, 12))
            in_table = False
            table_rows = []
            continue
        
        if in_table:
            if '<tr>' in line:
                current_row = []
            elif '</tr>' in line:
                if current_row:
                    table_rows.append(current_row)
            elif '<th>' in line or '<td>' in line:
                cell_text = strip_html(line)
                if cell_text:
                    current_row.append(cell_text)
            continue
        
        # Handle headings
        if line.startswith('<h1>'):
            text = strip_html(line)
            if 'RAG-Based Course Recommendation System' in text:
                elements.append(Paragraph(text, title_style))
            else:
                elements.append(Paragraph(text, h1_style))
            elements.append(Spacer(1, 12))
            continue
        
        if line.startswith('<h2>'):
            text = strip_html(line)
            elements.append(Paragraph(text, h2_style))
            elements.append(Spacer(1, 10))
            continue
        
        if line.startswith('<h3>'):
            text = strip_html(line)
            elements.append(Paragraph(text, h3_style))
            elements.append(Spacer(1, 8))
            continue
        
        # Handle horizontal rules
        if '<hr' in line:
            elements.append(Spacer(1, 20))
            continue
        
        # Handle lists
        if '<li>' in line:
            text = strip_html(line)
            text = '• ' + text if text else text
            elements.append(Paragraph(text, normal_style))
            continue
        
        # Handle blockquotes
        if '<blockquote>' in line:
            text = strip_html(line)
            quote_style = ParagraphStyle(
                'Quote',
                parent=normal_style,
                leftIndent=20,
                borderColor=colors.HexColor('#3498db'),
                borderWidth=0,
                borderPadding=(0, 0, 0, 4)
            )
            elements.append(Paragraph(text, quote_style))
            continue
        
        # Handle regular paragraphs
        text = strip_html(line)
        if text and not text.startswith('<'):
            # Clean up common markdown artifacts
            text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)  # Bold
            text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)  # Italic
            text = re.sub(r'`(.*?)`', r'<font name="Courier" size="9" backColor="#f4f4f4">\1</font>', text)  # Inline code
            
            elements.append(Paragraph(text, normal_style))
    
    # Build PDF
    print(f"📄 Converting {md_file} to PDF...")
    doc.build(elements)
    print(f"✅ PDF created successfully: {pdf_file}")

if __name__ == "__main__":
    md_file = "agents/scripts/RAG_Course_Recommendation_System_Report.md"
    pdf_file = "agents/scripts/RAG_Course_Recommendation_System_Report.pdf"
    
    if not os.path.exists(md_file):
        print(f"❌ Error: {md_file} not found")
        sys.exit(1)
    
    try:
        convert_md_to_pdf(md_file, pdf_file)
        print(f"\n📊 File size: {os.path.getsize(pdf_file) / 1024:.1f} KB")
    except Exception as e:
        print(f"❌ Error converting to PDF: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

Convert Markdown report to PDF using markdown and reportlab.
"""
import markdown
import re
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Preformatted
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from html.parser import HTMLParser
import sys
import os

class HTMLStripper(HTMLParser):
    """Strip HTML tags from text."""
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text = []
    
    def handle_data(self, d):
        self.text.append(d)
    
    def get_text(self):
        return ''.join(self.text)

def strip_html(html):
    """Remove HTML tags from text."""
    s = HTMLStripper()
    s.feed(html)
    return s.get_text()

def convert_md_to_pdf(md_file, pdf_file):
    """Convert markdown file to PDF."""
    # Read markdown file
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert markdown to HTML
    html_content = markdown.markdown(
        md_content,
        extensions=['extra', 'codehilite', 'tables', 'toc']
    )
    
    # Create PDF document
    doc = SimpleDocTemplate(
        pdf_file,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18
    )
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=30
    )
    
    h1_style = ParagraphStyle(
        'CustomH1',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        spaceBefore=20
    )
    
    h2_style = ParagraphStyle(
        'CustomH2',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#34495e'),
        spaceAfter=10,
        spaceBefore=15
    )
    
    h3_style = ParagraphStyle(
        'CustomH3',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=colors.HexColor('#7f8c8d'),
        spaceAfter=8,
        spaceBefore=12
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        spaceAfter=6
    )
    
    code_style = ParagraphStyle(
        'CustomCode',
        parent=styles['Code'],
        fontSize=9,
        fontName='Courier',
        backColor=colors.HexColor('#f4f4f4'),
        leftIndent=10,
        rightIndent=10,
        spaceAfter=10,
        spaceBefore=10
    )
    
    # Parse HTML and convert to PDF elements
    lines = html_content.split('\n')
    in_code_block = False
    code_block_lines = []
    in_table = False
    table_rows = []
    
    for line in lines:
        line = line.strip()
        
        if not line:
            if not in_code_block and not in_table:
                elements.append(Spacer(1, 6))
            continue
        
        # Handle code blocks
        if '<pre>' in line or '<code class=' in line:
            in_code_block = True
            code_block_lines = []
            continue
        
        if '</pre>' in line or '</code>' in line:
            if code_block_lines:
                code_text = '\n'.join(code_block_lines)
                code_text = strip_html(code_text)
                elements.append(Preformatted(code_text, code_style))
                elements.append(Spacer(1, 10))
            in_code_block = False
            code_block_lines = []
            continue
        
        if in_code_block:
            code_block_lines.append(strip_html(line))
            continue
        
        # Handle tables
        if '<table>' in line:
            in_table = True
            table_rows = []
            continue
        
        if '</table>' in line:
            if table_rows:
                # Create table
                table = Table(table_rows)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                    ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#ddd')),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9f9f9')]),
                ]))
                elements.append(table)
                elements.append(Spacer(1, 12))
            in_table = False
            table_rows = []
            continue
        
        if in_table:
            if '<tr>' in line:
                current_row = []
            elif '</tr>' in line:
                if current_row:
                    table_rows.append(current_row)
            elif '<th>' in line or '<td>' in line:
                cell_text = strip_html(line)
                if cell_text:
                    current_row.append(cell_text)
            continue
        
        # Handle headings
        if line.startswith('<h1>'):
            text = strip_html(line)
            if 'RAG-Based Course Recommendation System' in text:
                elements.append(Paragraph(text, title_style))
            else:
                elements.append(Paragraph(text, h1_style))
            elements.append(Spacer(1, 12))
            continue
        
        if line.startswith('<h2>'):
            text = strip_html(line)
            elements.append(Paragraph(text, h2_style))
            elements.append(Spacer(1, 10))
            continue
        
        if line.startswith('<h3>'):
            text = strip_html(line)
            elements.append(Paragraph(text, h3_style))
            elements.append(Spacer(1, 8))
            continue
        
        # Handle horizontal rules
        if '<hr' in line:
            elements.append(Spacer(1, 20))
            continue
        
        # Handle lists
        if '<li>' in line:
            text = strip_html(line)
            text = '• ' + text if text else text
            elements.append(Paragraph(text, normal_style))
            continue
        
        # Handle blockquotes
        if '<blockquote>' in line:
            text = strip_html(line)
            quote_style = ParagraphStyle(
                'Quote',
                parent=normal_style,
                leftIndent=20,
                borderColor=colors.HexColor('#3498db'),
                borderWidth=0,
                borderPadding=(0, 0, 0, 4)
            )
            elements.append(Paragraph(text, quote_style))
            continue
        
        # Handle regular paragraphs
        text = strip_html(line)
        if text and not text.startswith('<'):
            # Clean up common markdown artifacts
            text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)  # Bold
            text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)  # Italic
            text = re.sub(r'`(.*?)`', r'<font name="Courier" size="9" backColor="#f4f4f4">\1</font>', text)  # Inline code
            
            elements.append(Paragraph(text, normal_style))
    
    # Build PDF
    print(f"📄 Converting {md_file} to PDF...")
    doc.build(elements)
    print(f"✅ PDF created successfully: {pdf_file}")

if __name__ == "__main__":
    md_file = "agents/scripts/RAG_Course_Recommendation_System_Report.md"
    pdf_file = "agents/scripts/RAG_Course_Recommendation_System_Report.pdf"
    
    if not os.path.exists(md_file):
        print(f"❌ Error: {md_file} not found")
        sys.exit(1)
    
    try:
        convert_md_to_pdf(md_file, pdf_file)
        print(f"\n📊 File size: {os.path.getsize(pdf_file) / 1024:.1f} KB")
    except Exception as e:
        print(f"❌ Error converting to PDF: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
