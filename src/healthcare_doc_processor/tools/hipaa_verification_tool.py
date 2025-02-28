from langchain.tools import BaseTool
from typing import Optional, Type, List
from pydantic import BaseModel, Field
import re
import json

class HIPAAVerificationInput(BaseModel):
    """Input for HIPAA verification."""
    document_text: str = Field(..., description="Text content to verify for HIPAA compliance")
    document_metadata: dict = Field({}, description="Metadata associated with the document")
    verification_level: str = Field("standard", description="Verification level: 'basic', 'standard', 'strict'")

class HIPAAVerificationTool(BaseTool):
    name = "hipaa_verification_tool"
    description = "Verify document content for HIPAA compliance"
    args_schema: Type[BaseModel] = HIPAAVerificationInput
    
    def __init__(self):
        super().__init__()
        # Initialize HIPAA compliance patterns
        self._init_compliance_patterns()
    
    def _init_compliance_patterns(self):
        """Initialize patterns for detecting PHI (Protected Health Information)."""
        # Patient identifiers
        self.patterns = {
            # Names
            "names": r"\b[A-Z][a-z]+ [A-Z][a-z]+\b",
            
            # Phone numbers - various formats
            "phone_numbers": r"\b(\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b",
            
            # Email addresses
            "emails": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            
            # Social Security Numbers
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            
            # Dates (potentially DOB)
            "dates": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
            
            # Medical record numbers (simplified pattern)
            "mrn": r"\bMRN:?\s*\d{6,10}\b",
            
            # Address patterns (simplified)
            "addresses": r"\b\d{1,5}\s[A-Z][a-z]+\s[A-Z][a-z]+\b"
        }
    
    def _detect_phi(self, text: str, level: str = "standard") -> List[dict]:
        """Detect potential PHI in text."""
        findings = []
        
        # Apply patterns based on verification level
        patterns_to_check = self.patterns
        if level == "basic":
            # Only check the most critical identifiers
            patterns_to_check = {k: v for k, v in self.patterns.items() 
                               if k in ["ssn", "mrn", "emails"]}
        
        # Find matches
        for category, pattern in patterns_to_check.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                findings.append({
                    "category": category,
                    "match": match.group(),
                    "position": match.span(),
                    "risk_level": "high" if category in ["ssn", "mrn"] else "medium"
                })
        
        return findings
    
    def _verify_data_handling(self, metadata: dict) -> List[dict]:
        """Verify data handling practices from metadata."""
        issues = []
        
        # Check for required metadata fields
        required_fields = ["access_level", "retention_period", "encryption"]
        for field in required_fields:
            if field not in metadata:
                issues.append({
                    "type": "missing_metadata",
                    "field": field,
                    "description": f"Required HIPAA metadata field '{field}' is missing",
                    "severity": "medium"
                })
        
        # Check encryption status
        if "encryption" in metadata and metadata["encryption"] != "yes":
            issues.append({
                "type": "unencrypted_data",
                "description": "Document is not marked as encrypted",
                "severity": "high"
            })
        
        # Check retention policy
        if "retention_period" in metadata:
            try:
                retention = int(metadata["retention_period"])
                if retention > 10:
                    issues.append({
                        "type": "excessive_retention",
                        "description": f"Retention period of {retention} years may be excessive",
                        "severity": "low"
                    })
            except (ValueError, TypeError):
                pass
        
        return issues
    
    def _run(self, document_text: str, document_metadata: dict = {}, 
             verification_level: str = "standard"):
        """Run HIPAA verification on document content."""
        results = {
            "verification_level": verification_level,
            "timestamp": datetime.datetime.now().isoformat(),
            "document_size": len(document_text),
            "compliant": True,
            "findings": [],
            "recommendations": []
        }
        
        # Detect PHI
        phi_findings = self._detect_phi(document_text, verification_level)
        
        # Check metadata and data handling
        handling_issues = self._verify_data_handling(document_metadata)
        
        # Add findings to results
        if phi_findings:
            results["findings"].extend([{
                "type": "potential_phi",
                "details": finding
            } for finding in phi_findings])
            
            # Add recommendations for PHI
            results["recommendations"].append({
                "action": "anonymize",
                "description": "Anonymize or redact detected PHI",
                "priority": "high"
            })
            
            # Document is not compliant if PHI is found
            results["compliant"] = False
        
        # Add handling issues
        if handling_issues:
            results["findings"].extend([{
                "type": "data_handling",
                "details": issue
            } for issue in handling_issues])
            
            # Add recommendations for handling issues
            for issue in handling_issues:
                if issue["type"] == "missing_metadata":
                    results["recommendations"].append({
                        "action": "add_metadata",
                        "description": f"Add required metadata field: {issue['field']}",
                        "priority": "medium"
                    })
                elif issue["type"] == "unencrypted_data":
                    results["recommendations"].append({
                        "action": "encrypt",
                        "description": "Encrypt document according to HIPAA standards",
                        "priority": "high"
                    })
            
            # Document is not compliant if there are high severity issues
            if any(issue["severity"] == "high" for issue in handling_issues):
                results["compliant"] = False
        # Summary
            results["summary"] = {
              "phi_count": len(phi_findings),
                "handling_issues": len(handling_issues),
                "recommendations_count": len(results["recommendations"]),
             "compliance_status": "Compliant" if results["compliant"] else "Non-compliant"
            }

        return json.dumps(results, indent=2)