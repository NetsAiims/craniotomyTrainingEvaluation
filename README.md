# Craniotomy Training Evaluation 

This repository contains the code for craniotomy training evaluation, where a deep learning model is deployed and used to assess the surgical training performance of neurosurgeons. The model analyzes images captured after the training procedure to infer the quality and accuracy of the performed craniotomy.

The codebase in this repository successfully deployes a deep learning model for real time inference through a stand alone application built in python, The application also has features like gradient visualization and report generation with intuitive user interface and options to record and capture camera feed. 

For more details regarding the model architecture and training methodology, please refer to the following resources:

GitHub Repository - https://github.com/ramank1137/Microscopic-Neuro-Drilling;
Scientific Article - https://www.sciencedirect.com/science/article/abs/pii/S0010482525010017
  

## Installation & Setup

### Using Conda

```conda env create -f environment.yml```

```conda activate drillingEvaluation```

### Using Python Environment

```python -m venv venv```

For Windows
```venv\Scripts\activate``` 

For Linux or Mac
```venv/bin/activate```

```pip install -r requirements.txt```

## Run App

Before running application, declare environment variables "report_sender_email" and "report_sender_password" or create a file called credentials.py in root directory and assign your email and app passwords to variables.

Example:
```
credentials.py

        report_sender_email = your email
        report_sender_password = app password for the above email

```


To run the application, run program called GUI.py. Make sure that all mentioned dependencies are installed.

```python GUI.py```

### Miscelleneous

There are 3 additional files in src/utils/

1.legecyMacpdfWriter.py
2.legecyMacBasicTemplate.html
3.legecyMacStyle.css

These files contain the code for report generation using a tool wkhtmltopdf which did not support portability across different operationg systems.
