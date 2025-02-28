from langchain.tools import BaseTool
from typing import Optional, Type, List
from pydantic import BaseModel, Field
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import os
import json
from sklearn.metrics.pairwise import cosine_similarity

class NLPClassificationInput(BaseModel):
    """Input for NLP classification."""
    document_text: str = Field(..., description="Text content of the document to classify")
    healthcare_domain: str = Field(None, description="Specific healthcare domain if known (e.g., 'radiology', 'cardiology')")
    custom_categories: List[str] = Field(None, description="Optional custom categories to classify against")

class NLPClassificationTool(BaseTool):
    name = "nlp_classification_tool"
    description = "Classify medical documents using NLP and transformer models"
    args_schema: Type[BaseModel] = NLPClassificationInput
    
    # Initialize with pre-trained model
    def __init__(self):
        super().__init__()
        # Load model and tokenizer during initialization
        self.model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        
        # Load standard medical document categories
        self.categories = {
            "patient_record": "Patient health record containing personal and medical history",
            "lab_report": "Laboratory test results and analysis",
            "imaging_report": "Reports from imaging studies such as X-rays, CT scans, MRIs",
            "prescription": "Medication prescriptions and dosage information",
            "clinical_note": "Notes from clinical visits and observations",
            "discharge_summary": "Summary of hospital stay and follow-up instructions",
            "referral_letter": "Letters referring patients to specialists",
            "consent_form": "Patient consent for procedures or data usage",
            "insurance_claim": "Documentation for insurance reimbursement",
            "medical_certificate": "Official medical certification documents"
        }
        
        # Load medical taxonomies
        self._load_medical_taxonomies()
    
    def _load_medical_taxonomies(self):
        """Load medical taxonomies for more detailed classification."""
        # This would typically load from external files or databases
        # For demonstration, we'll define some basic taxonomies here
        self.medical_specialties = [
            "cardiology", "neurology", "oncology", "pediatrics", 
            "radiology", "orthopedics", "obstetrics", "gynecology",
            "psychiatry", "dermatology", "endocrinology", "immunology"
        ]
        
        self.document_priority = [
            "urgent", "routine", "follow-up", "screening"
        ]
    
    def _get_embeddings(self, text: str):
        """Generate embeddings for text using the PubMedBERT model."""
        # Truncate text if too long
        max_length = 512
        if len(text) > max_length * 4:  # Rough estimate
            text = text[:max_length * 4]  # Truncate for processing efficiency
            
        # Tokenize and get embeddings
        inputs = self.tokenizer(text, return_tensors="pt", 
                               max_length=max_length, 
                               padding=True, 
                               truncation=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use CLS token as the document embedding
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        
        return embeddings
    
    def _classify_document(self, document_text: str, categories: dict) -> dict:
        """Classify document against provided categories."""
        document_embedding = self._get_embeddings(document_text)
        
        results = {}
        category_embeddings = {}
        
        # Get embeddings for each category description
        for category, description in categories.items():
            category_embeddings[category] = self._get_embeddings(description)
        
        # Calculate similarity scores
        for category, embedding in category_embeddings.items():
            similarity = cosine_similarity(document_embedding, embedding)[0][0]
            results[category] = float(similarity)
        
        # Sort by similarity score
        sorted_results = {k: v for k, v in sorted(
            results.items(), key=lambda item: item[1], reverse=True)}
        
        return sorted_results
    
    def _extract_medical_entities(self, document_text: str) -> dict:
        """Extract medical entities from text (simplified version)."""
        # This would normally use a medical NER model
        # For demonstration, we'll use a simplified keyword approach
        
        entities = {
            "conditions": [],
            "medications": [],
            "procedures": [],
            "specialties": []
        }
        
        # Check for common conditions (simplified)
        conditions = ["diabetes", "hypertension", "asthma", "cancer", 
                     "depression", "anxiety", "arthritis", "fracture"]
        for condition in conditions:
            if condition.lower() in document_text.lower():
                entities["conditions"].append(condition)
        
        # Check for specialties
        for specialty in self.medical_specialties:
            if specialty.lower() in document_text.lower():
                entities["specialties"].append(specialty)
        
        return entities
        
    def _run(self, document_text: str, healthcare_domain: str = None, 
             custom_categories: List[str] = None):
        """Run classification on medical document."""
        results = {}
        
        # Primary classification against standard categories
        classification = self._classify_document(document_text, self.categories)
        results["primary_classification"] = classification
        
        # Get top category
        top_category = next(iter(classification))
        confidence = classification[top_category]
        results["category"] = top_category
        results["confidence"] = confidence
        
        # Medical entity extraction
        entities = self._extract_medical_entities(document_text)
        results["entities"] = entities
        
        # Domain-specific classification if provided
        if healthcare_domain:
            results["domain"] = healthcare_domain
            
        # Custom category classification if provided
        if custom_categories:
            custom_dict = {cat: cat for cat in custom_categories}
            custom_results = self._classify_document(document_text, custom_dict)
            results["custom_classification"] = custom_results
        
        return json.dumps(results, indent=2)