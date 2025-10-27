import jinja2
import pdfkit
from datetime import datetime
import email, smtplib, ssl
import os
from PyQt5.QtWidgets import QMessageBox

from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from credentials import *

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
                        'image': "file://" + os.getcwd() + "/" + self.inputs["image"],
                        #'gradImage' : "file://" + os.getcwd() + "/" + self.inputs["gradImage"],
                        'today_date': today_date,
                        'time': self.inputs["time"],
                        'logo1': "file://" + os.getcwd() + "/" + "src/images/aiims.png",
                        'logo2': "file://" + os.getcwd() + "/" + "src/images/iitd.png",
                    }

            if "gradImage" in self.inputs and self.inputs["gradImage"] is not None:
                gradcam_path = self.inputs["gradImage"]
                
                if os.path.exists(gradcam_path):
                    print(f"DEBUG: Gradcam file exists at: {gradcam_path}")
                    context['gradImage'] = "file://" + os.getcwd() + "/" + gradcam_path
                else:
                    print(f"DEBUG: Gradcam file doesn't exist at: {gradcam_path}")
            else:
                print("DEBUG: No gradImage key found in inputs")

            print(f"DEBUG: Final context keys: {list(context.keys())}")
            print(f"DEBUG: Context has gradImage: {'gradImage' in context}")

            template_loader = jinja2.FileSystemLoader('./')
            template_env = jinja2.Environment(loader=template_loader)

            html_template = 'src/utils/basic-template.html'
            template = template_env.get_template(html_template)
            output_text = template.render(context)

            config = pdfkit.configuration(wkhtmltopdf='/usr/local/bin/wkhtmltopdf')
            # config = pdfkit.configuration(wkhtmltopdf='/usr/bin/wkhtmltopdf')           
            if not os.path.exists("data/Reports"):
                os.mkdir("data/Reports")
            pdf_name = f'data/Reports/{self.inputs["name"]}_{self.inputs["iteration"]}.pdf'
            output_pdf = 'data/Reports/pdf_generated.pdf'
            pdfkit.from_string(output_text, output_pdf, configuration=config, css=['src/utils/style.css'], options={"enable-local-file-access": ""})
            pdfkit.from_string(output_text, pdf_name, configuration=config, css=['src/utils/style.css'], options={"enable-local-file-access": ""})

            subject = "Drilling Evaluation Report"
            body = "Dear Sir/Madam,\n\nThank you for your interest in the drilling task. Please find attached the evaluation report for the same. Please note that the scores presented herein have been acquired through the utilization of an automated, AI-driven tool. In light of the ongoing nature of the associated research, this document does not possess the legal authority of a formal certificate.\n\nThanks and Regards,\nNETS Lab, AIIMS, New Delhi"
            sender_email = report_sender_email
            receiver_email = self.inputs["email"]
            password = report_sender_password

            message = MIMEMultipart()
            message["From"] = sender_email
            message["To"] = receiver_email
            message["Subject"] = subject
            message["Bcc"] = receiver_email  

            message.attach(MIMEText(body, "plain"))

            filename = "data/Reports/pdf_generated.pdf"  # In same directory as script

            with open(filename, "rb") as attachment:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())

            encoders.encode_base64(part)

            part.add_header(
                "Content-Disposition",
                f"attachment; filename= {filename}",
            )

            message.attach(part)
            text = message.as_string()

            context = ssl.create_default_context()
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
                server.login(sender_email, password)
                server.sendmail(sender_email, receiver_email, text)
            self.showdialog(text1="Report Sent", text2="Please check your email for the generated report.")
        except Exception as e:
            print(f"Error occurred: {e}")
            self.showdialog(text1="Some error occured", text2="Please try again later")
            return




