from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field
import boto3
import pytesseract
from pdf2image import convert_from_path
import os
from PIL import Image
import io
import base64

class OCRInput(BaseModel):
    """Input for OCR processing."""
    document_path: str = Field(..., description="Path or URL to the document to be processed")
    document_type: str = Field(..., description="Type of document: 'pdf', 'image', 'scan'")
    use_aws: bool = Field(False, description="Whether to use AWS Textract for OCR")
    
class OCRTool(BaseTool):
    name = "ocr_tool"
    description = "Extract text from documents using OCR technology"
    args_schema: Type[BaseModel] = OCRInput
    
    def _run(self, document_path: str, document_type: str, use_aws: bool = False):
        """Run OCR on the document."""
        if use_aws:
            return self._aws_textract(document_path)
        else:
            return self._tesseract_ocr(document_path, document_type)
    
    def _aws_textract(self, document_path: str) -> str:
        """Use AWS Textract for OCR."""
        try:
            client = boto3.client('textract')
            
            if document_path.startswith('http'):
                import requests
                response = requests.get(document_path)
                document_bytes = response.content
            else:
                with open(document_path, 'rb') as document:
                    document_bytes = document.read()
            
            response = client.detect_document_text(Document={'Bytes': document_bytes})
            
            text = ""
            for item in response['Blocks']:
                if item['BlockType'] == 'LINE':
                    text += item['Text'] + '\n'
            
            return text
        except Exception as e:
            return f"Error using AWS Textract: {str(e)}"
    
    def _tesseract_ocr(self, document_path: str, document_type: str) -> str:
        """Use Tesseract for OCR."""
        try:
            if document_type.lower() == 'pdf':
                # Convert PDF to images
                images = convert_from_path(document_path)
                text = ""
                for img in images:
                    text += pytesseract.image_to_string(img) + "\n"
                return text
            
            elif document_type.lower() in ['image', 'scan', 'jpg', 'jpeg', 'png']:
                # Process image directly
                image = Image.open(document_path)
                text = pytesseract.image_to_string(image)
                return text
            
            else:
                return f"Unsupported document type: {document_type}"
        
        except Exception as e:
            return f"Error using Tesseract OCR: {str(e)}"


