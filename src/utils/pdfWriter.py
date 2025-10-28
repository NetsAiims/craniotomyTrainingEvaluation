import jinja2
from datetime import datetime
import email, smtplib, ssl
import os
import base64 

from PyQt5.QtWidgets import QMessageBox
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from credentials import *

from playwright.sync_api import sync_playwright

def image_to_base64(filepath):
    """
    Reads an image file and converts it into a Base64 encoded data URI.
    This makes the image self-contained within the HTML.
    """
    try:

        file_ext = os.path.splitext(filepath)[1].lower()
        mime_types = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.gif': 'image/gif'}
        mime_type = mime_types.get(file_ext, 'application/octet-stream')

        with open(filepath, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:{mime_type};base64,{encoded_string}"
    except Exception as e:
        print(f"Error converting image to base64: {filepath}, Error: {e}")
        return "" 

class PdfWrite():
    def __init__(self, inputs):
        self.inputs = inputs

    def showdialog(self, text1, text2):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText(text1)
        msg.setInformativeText(text2)
        msg.setStandardButtons(QMessageBox.Ok)
        retval = msg.exec_()

    def write(self):
        try:
            today_date = datetime.today().strftime("%d %b, %Y")
            
            context = {
                        'name': self.inputs["name"], 
                        'email': self.inputs["email"],
                        'final': self.inputs["item1"],
                        'program': self.inputs["program"],
                        'iteration': self.inputs["iteration"],
                        'image': image_to_base64(self.inputs["image"]), 
                        'today_date': today_date,
                        'time': self.inputs["time"],
                        'logo1': image_to_base64("src/images/aiims.png"), 
                        'logo2': image_to_base64("src/images/iitd.png"), 
                    }

            if "gradImage" in self.inputs and self.inputs["gradImage"] is not None:
                gradcam_path = self.inputs["gradImage"]
                if os.path.exists(gradcam_path):
                    context['gradImage'] = image_to_base64(gradcam_path) 

            template_loader = jinja2.FileSystemLoader('./')
            template_env = jinja2.Environment(loader=template_loader)

            html_template = 'src/utils/basicTemplate.html'
            template = template_env.get_template(html_template)
            output_text = template.render(context)
            
            with open('src/utils/style.css', 'r', encoding='utf-8') as f:
                css_string = f.read()
            style_injection_point = output_text.find('</head>')
            final_html_content = output_text[:style_injection_point] + f'<style>{css_string}</style>' + output_text[style_injection_point:]

            if not os.path.exists("data/Reports"):
                os.mkdir("data/Reports")
            pdf_name = f'data/Reports/{self.inputs["name"]}_{self.inputs["iteration"]}.pdf'
            output_pdf = 'data/Reports/pdf_generated.pdf'

            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.set_content(final_html_content, wait_until='load')
                
                pdf_options = {
                    "format": 'A4',
                    "print_background": True,
                    "margin": {'top': '0px', 'bottom': '0px', 'left': '0px', 'right': '0px'}
                }
                
                page.pdf(path=output_pdf, **pdf_options)
                page.pdf(path=pdf_name, **pdf_options)
                browser.close()
                
            print("PDF generation complete.")
            
            subject = "Drilling Evaluation Report"
            body = "Dear Sir/Madam,\n\nThank you for your interest in the drilling task. Please find attached the evaluation report for the same. Please note that the scores presented herein have been acquired through the utilization of an automated, AI-driven tool. In light of the ongoing nature of the associated research, this document does not possess the legal authority of a formal certificate.\n\nThanks and Regards,\nNETS Lab, AIIMS, New Delhi"
            sender_email = report_sender_email
            receiver_email = self.inputs["email"]
            password = report_sender_password

            message = MIMEMultipart()
            message["From"] = sender_email
            message["To"] = receiver_email
            message["Subject"] = subject
            
            message.attach(MIMEText(body, "plain"))
            
            filename = output_pdf

            with open(filename, "rb") as attachment:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())

            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename=Drilling_Report.pdf")
            message.attach(part)
            text = message.as_string()

            context = ssl.create_default_context()
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
                server.login(sender_email, password)
                server.sendmail(sender_email, receiver_email, text)
            
            self.showdialog(text1="Report Sent", text2="Please check your email for the generated report.")
        
        except Exception as e:
            print(f"An error occurred: {e}")
            import traceback
            traceback.print_exc()
            self.showdialog(text1="An error occurred", text2="Please check the console for details and try again.")
            return
