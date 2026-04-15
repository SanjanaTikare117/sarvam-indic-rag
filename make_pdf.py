from fpdf import FPDF

pdf = FPDF()
pdf.add_page()
pdf.set_font("Helvetica", size=12)

lines = [
    "Karnataka is a state in South India.",
    "Its capital is Bengaluru.",
    "Kannada is the official language of Karnataka.",
    "Mysuru Dasara is a famous festival.",
    "Hampi is a historic site in Karnataka.",
]
for line in lines:
    pdf.cell(200, 10, txt=line, ln=True)

pdf.output("docs/kannada_clean.pdf")