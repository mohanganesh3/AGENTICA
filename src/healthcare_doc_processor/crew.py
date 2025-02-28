from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from langchain.chat_models import ChatGoogleGenerativeAI
import os

# Import tools
from healthcare_doc_processor.tools.ocr_tool import OCRTool
from healthcare_doc_processor.tools.format_conversion_tool import FormatConversionTool
from healthcare_doc_processor.tools.nlp_classification_tool import NLPClassificationTool
from healthcare_doc_processor.tools.vector_embedding_tool import VectorEmbeddingTool
from healthcare_doc_processor.tools.hipaa_verification_tool import HIPAAVerificationTool

@CrewBase
class HealthcareDocProcessorCrew():
    """Healthcare Document Processing Crew for processing, analyzing, 
    and ensuring compliance of medical documents"""
    
    def __init__(self):
        # Initialize LLM for all agents
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
    
    @agent
    def document_processor(self) -> Agent:
        """Document Processing Specialist agent that handles OCR and format conversion"""
        return Agent(
            role="Document Processing Specialist",
            goal="Process and normalize documents into standard formats with high accuracy",
            backstory="Specialized in OCR and document transformation with expertise in healthcare formats",
            tools=[OCRTool(), FormatConversionTool()],
            llm=self.llm,
            verbose=True
        )
    
    @agent
    def medical_tagger(self) -> Agent:
        """Medical Content Classifier agent that tags and categorizes medical content"""
        return Agent(
            role="Medical Content Classifier",
            goal="Accurately classify and tag medical documents based on content and context",
            backstory="Trained on extensive medical taxonomy and healthcare documentation standards",
            tools=[NLPClassificationTool()],
            llm=self.llm,
            verbose=True
        )
    
    @agent
    def search_specialist(self) -> Agent:
        """Healthcare Information Retrieval Expert for searchable indices"""
        return Agent(
            role="Healthcare Information Retrieval Expert",
            goal="Create and optimize searchable indices for fast, accurate document retrieval",
            backstory="Specialized in semantic search technologies for healthcare applications",
            tools=[VectorEmbeddingTool()],
            llm=self.llm,
            verbose=True
        )
    
    @agent
    def compliance_auditor(self) -> Agent:
        """Healthcare Compliance Specialist for regulatory requirements"""
        return Agent(
            role="Healthcare Compliance Specialist",
            goal="Ensure all document processing and storage meets HIPAA and other regulatory requirements",
            backstory="Former healthcare compliance officer with deep knowledge of regulatory frameworks",
            tools=[HIPAAVerificationTool()],
            llm=self.llm,
            verbose=True
        )
    
    @task
    def document_ingestion(self) -> Task:
        """Task for processing incoming documents through OCR and format normalization"""
        return Task(
            description="Process incoming documents through OCR and format normalization",
            expected_output="Normalized text content with metadata",
            agent=self.document_processor(),
        )
    
    @task
    def content_classification(self) -> Task:
        """Task for analyzing and classifying document content"""
        return Task(
            description="Analyze document content and assign appropriate medical tags and categories",
            expected_output="Document with healthcare-specific tags and classifications",
            agent=self.medical_tagger(),
            context=[self.document_ingestion()]
        )
    
    @task
    def index_creation(self) -> Task:
        """Task for creating searchable indices for document content"""
        return Task(
            description="Create searchable indices for document content using vector embeddings",
            expected_output="Searchable document index with semantic capabilities",
            agent=self.search_specialist(),
            context=[self.document_ingestion(), self.content_classification()]
        )
    
    @task
    def compliance_check(self) -> Task:
        """Task for verifying document processing meets healthcare regulations"""
        return Task(
            description="Verify all document processing meets healthcare regulatory requirements",
            expected_output="Compliance verification report with any required actions",
            agent=self.compliance_auditor(),
            context=[self.document_ingestion(), self.content_classification()],
            output_file="compliance_report.json"
        )
    
    @crew
    def crew(self) -> Crew:
        """Creates the Healthcare Document Processing crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,    # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=2,
        )