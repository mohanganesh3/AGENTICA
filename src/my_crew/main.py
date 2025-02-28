#!/usr/bin/env python
import os
from dotenv import load_dotenv
from my_crew.crew import HealthcareDocProcessorCrew

# Load environment variables
load_dotenv()

def run():
    # Replace with your inputs for document processing
    inputs = {
        'document_path': '/Users/mohanganesh/Desktop/Medical_Report.pdf',
        'document_type': 'pdf',
        'output_format': 'json',
        'compliance_level': 'strict'
    }
    
    # Kickoff the crew with the specified inputs
    HealthcareDocProcessorCrew().crew().kickoff(inputs=inputs)

if __name__ == "__main__":
    run()