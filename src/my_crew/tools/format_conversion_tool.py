from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field
import docx
import io
import base64
import zipfile
import xml.etree.ElementTree as ET
import json
import re
import csv
import datetime
import os

class FormatConversionInput(BaseModel):
    """Input for format conversion."""
    content: str = Field(..., description="Content or file path to convert")
    source_format: str = Field(..., description="Original format: 'docx', 'text', 'html', 'pdf_text'")
    target_format: str = Field(..., description="Target format: 'text', 'json', 'html', 'markdown'")
    metadata: dict = Field({}, description="Additional metadata to include")

class FormatConversionTool(BaseTool):
    name = "format_conversion_tool"
    description = "Convert documents between different formats"
    args_schema: Type[BaseModel] = FormatConversionInput
    
    def _run(self, content: str, source_format: str, target_format: str, metadata: dict = {}):
        """Convert document format."""
        # First extract content if it's a file path
        extracted_content = self._extract_content(content, source_format)
        
        # Then convert to target format
        if target_format.lower() == 'text':
            return extracted_content
        
        elif target_format.lower() == 'json':
            return self._convert_to_json(extracted_content, metadata)
        
        elif target_format.lower() == 'html':
            return self._convert_to_html(extracted_content, source_format, metadata)
        
        elif target_format.lower() == 'markdown':
            return self._convert_to_markdown(extracted_content, source_format, metadata)
        
        else:
            return f"Unsupported target format: {target_format}"
    
    def _extract_content(self, content: str, source_format: str) -> str:
        """Extract text content from various formats."""
        try:
            # If content is a file path
            if os.path.exists(content):
                if source_format.lower() == 'docx':
                    doc = docx.Document(content)
                    return "\n".join([para.text for para in doc.paragraphs])
                
                elif source_format.lower() in ['text', 'txt']:
                    with open(content, 'r', encoding='utf-8') as file:
                        return file.read()
                
                elif source_format.lower() == 'html':
                    with open(content, 'r', encoding='utf-8') as file:
                        html_content = file.read()
                    return self._extract_text_from_html(html_content)
                
                else:
                    return f"Unsupported source format for file: {source_format}"
            
            # If content is the actual content as string
            else:
                if source_format.lower() == 'html':
                    return self._extract_text_from_html(content)
                elif source_format.lower() in ['text', 'txt', 'pdf_text', 'docx_text']:
                    return content
                else:
                    return content
                    
        except Exception as e:
            return f"Error extracting content: {str(e)}"
    
    def _extract_text_from_html(self, html_content: str) -> str:
        """Extract text from HTML content."""
        # Simple regex-based HTML tag removal
        clean_text = re.sub(r'<[^>]*>', ' ', html_content)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        return clean_text
    
    def _convert_to_json(self, content: str, metadata: dict) -> str:
        """Convert content to JSON format."""
        result = {
            "content": content,
            "metadata": metadata,
            "length": len(content),
            "timestamp": datetime.datetime.now().isoformat()
        }
        return json.dumps(result, indent=2)
    
    def _convert_to_html(self, content: str, source_format: str, metadata: dict) -> str:
        """Convert content to HTML format."""
        # Create basic HTML structure
        paragraphs = content.split('\n')
        html_paragraphs = [f"<p>{p}</p>" for p in paragraphs if p.strip()]
        
        # Add metadata as comments
        metadata_str = "<!-- Document Metadata:\n"
        for key, value in metadata.items():
            metadata_str += f"{key}: {value}\n"
        metadata_str += "-->"
        
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>{metadata.get('title', 'Converted Document')}</title>
    <meta charset="UTF-8">
    {metadata_str}
</head>
<body>
    <div class="document-content">
        {"".join(html_paragraphs)}
    </div>
</body>
</html>"""
        
        return html_content
    
    def _convert_to_markdown(self, content: str, source_format: str, metadata: dict) -> str:
        """Convert content to Markdown format."""
        # Add metadata as YAML front matter
        front_matter = "---\n"
        for key, value in metadata.items():
            front_matter += f"{key}: {value}\n"
        front_matter += "---\n\n"
        
        # Basic conversion of paragraphs
        paragraphs = content.split('\n')
        markdown_content = front_matter
        
        # Add title if available
        if 'title' in metadata:
            markdown_content += f"# {metadata['title']}\n\n"
        
        # Add paragraphs
        for p in paragraphs:
            if p.strip():
                markdown_content += f"{p}\n\n"
        
        return markdown_content