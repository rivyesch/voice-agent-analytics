from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from enum import Enum
from datetime import datetime
import json
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
from extract_thread_messages import extract_thread_messages

load_dotenv()

# Initialize Azure OpenAI client for structured output
client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"]
)


class RequestType(str, Enum):
    """ Primary classification of the request"""
    SERVICE_REQUEST = "service_request"
    INCIDENT = "incident"
    GENERAL_INQUIRY = "general_inquiry"
    OUT_OF_SCOPE = "out_of_scope"


class IncidentCategory(str, Enum):
    """Specific incident categories aligned with KB structure"""
    UNIFLOW_PRINTER = "uniflow_printer"
    CITRIX = "citrix"
    MFA = "multifactor_authentication"
    AD_ACCOUNT = "ad_account"
    OTHER = "other"
    NOT_APPLICABLE = "not_applicable"

class ServiceRequestType(str, Enum):
    """Types of service requests (form-based)"""
    ACCESS_REQUEST = "access_request"
    SOFTWARE_REQUEST = "software_request"
    HARDWARE_REQUEST = "hardware_request"
    FORM_GUIDANCE = "form_guidance"
    OTHER = "other"
    NOT_APPLICABLE = "not_applicable"

class FormName(str, Enum):
    """Types of service requests (form-based)"""
    SAP_ACCESS = "sap_access"
    EMAIL_SHARED_DRIVE = "email_shared_drive"
    BUSINESS_APP_ACCESS = "business_app_access"
    NETWORK_ACCESS = "network_access"
    GENERAL_CATALOG = "general_catalog"
    OTHER = "other"
    NOT_APPLICABLE = "not_applicable"

class ResolutionMethod(str, Enum):
    """How it was resolved"""
    KB_GUIDED = "kb_guided"
    FORM_PROVIDED = "form_provided"
    SIMPLE_ANSWER = "simple_answer"
    ESCALATED = "escalated"
    NOT_RESOLVED = "not_resolved"
    USER_ABANDONED = "user_abandoned"
    OUT_OF_SCOPE = "out_of_scope"
    NOT_APPLICABLE = "not_applicable"

class ResolutionStatus(str, Enum):
    """Final outcome of the conversation"""
    RESOLVED_BY_BOT = "resolved_by_bot"  # Issue fixed without escalation
    RESOLVED_WITH_FORM = "resolved_with_form"  # User directed to correct form
    ESCALATED_TO_HUMAN = "escalated_to_human"  # Ticket created and left open for transfer to agent
    UNRESOLVED = "unresolved"  # No solution found or provided
    USER_ABANDONED = "user_abandoned"  # User left before completion
    OUT_OF_SCOPE = "out_of_scope"  # Non-IT issue

class EsclationReason(str, Enum):
    """Reason for escalation"""
    KB_STEPS_FAILED = "kb_steps_failed"
    NO_KB_ARTICLE = "no_kb_article"
    USER_REQUESTED_HUMAN = "user_requested_human"
    COMPLEX_ISSUE = "complex_issue"
    USER_FRUSTRATED = "user_frustrated"
    TECHNICAL_LIMITATION = "technical_limitation"
    AUTHENTICATION_REQUIRED = "authentication_required"
    NOT_ESCALATED = "not_escalated"

class ConversationQuality(str, Enum):
    """Quality of the bot's handling"""
    EXCELLENT = "excellent"  # Efficient, clear, resolved quickly
    GOOD = "good"  # Standard handling, minor issues
    FAIR = "fair"  # Some confusion or inefficiency
    POOR = "poor"  # Significant issues in handling
    

class UserSentiment(str, Enum):
    """User sentiment during the conversation"""
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    FRUSTRATED = "frustrated"    

class ConversationQuality(str, Enum):
    """Quality of the conversation"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    FAILED = "failed"

class ITHelpdeskConversation(BaseModel):
    """Schema for extracting structured data from IT helpdesk conversations, designed to power comprehensive analytics."""

    # Request Classification
    request_type: RequestType = Field(
        description="Type of request: service_request (form guidance) or incident (troubleshooting)"
    )
    
    incident_category: IncidentCategory = Field(
        description="Category of incident if applicable: uniflow_printer, citrix, mfa, ad_account, or other"
    )
    
    service_request_type: ServiceRequestType = Field(
        description="Type of service request if applicable: form_guidance, access_request, etc. Set tp 'not_applicable' if this is an incident."
    )

    form_name: FormName = Field(
        description="Name of the form provided if applicable: sap_access, email_shared_drive, business_app_access, network_access, general_catalog, or other"
    )
    # Issue Details
    primary_issue_description: str = Field(
        max_length=256,
        description="Brief description of the main issue or request in 1-2 clear sentences. Example: 'User unable to receive MFA approvals after switching to new phone. Needed help setting up Microsoft Authenticator on new device.'"
    )
    
    issue_keywords: List[str] = Field(
        default_factory=list,
        max_length=5,
        description="Key technical terms mentioned (e.g., 'new phone', 'security settings', 'login failed', 'access denied', 'freezing', 'crashing')"
    )
    
    # Resolution Information
    resolution_status: ResolutionStatus = Field(
        description="Final outcome of the conversation"
    )
    
    resolution_method: ResolutionMethod = Field(
        description="""HOW was it resolved? kb_guided, form_provided, simple_answer, escalated, not_resolved"""
    )
    
    resolution_provided: Optional[str] = Field(
        None,
        max_length=300,
        description="""Brief summary of solution provided. Only include if resolved/partially_resolved.
        Example: 'Guided user through security settings to add new Authenticator device via QR code.'"""
    )
    
    first_call_resolution: bool = Field(
        description="True if issue was fully resolved without escalation or follow-up needed"
    )

    automation_success: bool = Field(
        description="True if bot handled the request end-to-end without human agent intervention"
    )

    ticket_created: bool = Field(
        default=False,
        description="Whether a ServiceNow ticket (INC/REQ/IMS)was created"
    )

    ticket_number: Optional[str] = Field(
        None,
        description="ServiceNow ticket number if applicable"
    )

    form_urls_provided: List[str] = Field(
        default_factory=list,
        description="List of form URLs shared with the user"
    )
    
    kb_articles_referenced: bool = Field(
        default=False,
        description="Whether knowledge base articles were referenced or used"
    )
    
    # Escalation Tracking (for improvement areas)
    escalated_to_human: bool = Field(
        default=False,
        description="Whether conversation was escalated to a human agent"
    )

    escalation_reason: EsclationReason = Field(
        description="Primary reason for escalation if applicable"
    )

    escalation_point: Optional[int] = Field(
        None,
        description="Turn number at which escalation occurred (for efficiency analysis)"
    )   

    # Troubleshooting & KB Usage (for quality tracking)
    kb_search_performed: bool = Field(
        default=False,
        description="Whether a knowledge base search was performed"
    )

    kb_articles_found: bool = Field(
        default=False,
        description="Whether a relevant knowledge base articles was found"
    )

    kb_steps_attempted: List[str] = Field(
        default_factory=list,
        description="List of troubleshooting steps attempted from KB (e.g., 'restart phone', 'reinstall app', 'check iOS version')"
    )

    troubleshooting_steps_count: int = Field(
        default=0,
        description="Number of distinct troubleshooting steps attempted from KB"
    )

    kb_steps_successful: bool = Field(
        default=False,
        description="Whether the troubleshooting steps from KB were successful in resolving the issue"
    )

    # Form Handling (for Service Request Tracking)

    form_provided: bool = Field(
        default=False,
        description="Whether a form was provided to the user"
    )

    form_url_sent: Optional[str] = Field(
        None,
        description="The specific form URL provided to the user, if any"
    )

    form_identified_correctly: bool = Field(
        default=True,
        description="Whether the bot identified and provided the correct form based on user request"
    )

    # Conversation Metrics
    total_conversation_turns: int = Field(
        description="Total number of back-and-forth exchanges (user + assistant messages)"
    )
    
    user_message_count: int = Field(
        description="Number of messages from the user"
    )
    
    assistant_message_count: int = Field(
        description="Number of messages from the assistant"
    )
    
    # User Experience
    user_sentiment: UserSentiment = Field(
        description="Overall sentiment of the user throughout the conversation"
    )
    
    user_expressed_satisfaction: Optional[bool] = Field(
        None,
        description="True if user explicitly expressed satisfaction; False if dissatisfaction; None if unclear"
    )
    
    user_expressed_frustration: bool = Field(
        default=False,
        description="Whether user showed signs of frustration during conversation"
    )
    
    frustration_triggers: List[str] = Field(
        default_factory=list,
        description="What caused frustration if applicable (e.g., 'repeated failed steps', 'misunderstanding', 'long wait')"
    )
    
    # Conversation Quality (for bot performance)
    conversation_quality: ConversationQuality = Field(
        description="Overall quality rating of how well the bot handled the conversation"
    )

    followed_protocol: bool = Field(
        default=True,
        description="Whether the bot followed the system prompt protocols (KB search, step-by-step, etc.)"
    )

    issues_with_bot_behaviour: List[str] = Field(
        default_factory=list,
        description="List any problems with bot performance if applicable (e.g., 'skipped KB search', 'gave multiple questions at once', 'unclear instructions')"
    )

    # Additional Context (for follow-up & notes)
    follow_up_required: bool = Field(
        default=False,
        description="Whether a follow-up action is needed from a human agent or IT Team"
    )
    
    user_information_collected: bool = Field(
        default=False,
        description="Whether the bot collected user information (e.g., name, employee ID) for ticket logging"
    )

    multi_issue_conversation: bool = Field(
        default=False,
        description="Whether user raised multiple separate issues in one conversation"
    )
    
    out_of_scope_request: bool = Field(
        default=False,
        description="Whether user asked about non-IT topics (HR, payroll, facilities, etc.)"
    )
    
    technical_terms_confused_user: bool = Field(
        default=False,
        description="Whether technical jargon or terminology confused the user"
    )

    additional_notes: Optional[str] = Field(
        None,
        description="Any other relevant observations, edge cases or context"
    )


def extract_structured_data(conversation_messages: List[dict]) -> ITHelpdeskConversation:
    """
    Extract structured data from conversation messages using GPT-4o.
    
    Args:
        conversation_messages: List of dicts with 'role' and 'content' keys
        
    Returns:
        ITHelpdeskConversation object with extracted data
    """
    # Format the conversation for the prompt
    conversation_text = "\n".join([
        f"{msg['role'].upper()}: {msg['content']}" 
        for msg in conversation_messages
    ])
    
    # Create the extraction prompt
    system_prompt = """You are an expert at analyzing IT helpdesk conversations and extracting structured information.
Analyze the conversation and extract all relevant fields according to the schema provided.

For Asahi IT Helpdesk context:
- Service requests are when users need guidance to forms/URLs
- Incidents are troubleshooting issues in these categories: uniflow_printer, citrix, multifactor_authentication (MFA), ad_account
- Pay attention to resolution quality and user sentiment
- Note any escalations or unresolved issues
"""
    
    user_prompt = f"""Analyze this IT helpdesk conversation and extract structured information:

{conversation_text}

Provide detailed analysis following the schema."""
    
    # Call GPT-4o with structured output
    completion = client.beta.chat.completions.parse(
        model=os.environ["MODEL_DEPLOYMENT_NAME"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format=ITHelpdeskConversation
    )
    
    return completion.choices[0].message.parsed


# Example usage
if __name__ == "__main__":
    # Sample conversation (replace with actual extracted messages)
    # sample_conversation = [
    #     {"role": "user", "content": "Hello"},
    #     {"role": "assistant", "content": "Hi there, you're through to ZenBot. What can I help you with today?"},
    #     {"role": "user", "content": "My authenticator app is not working. I can't log in."},
    #     {"role": "assistant", "content": "I understand you're having issues with your authenticator app. Let me help you troubleshoot this. Are you able to open the Microsoft Authenticator app on your device?"},
    #     {"role": "user", "content": "Yes, I can open it but it's not showing the code"},
    #     {"role": "assistant", "content": "I see. This could be a synchronization issue. Let's try refreshing the app. Can you try pulling down on the screen to refresh the authenticator, and also check if your device's time settings are set to automatic?"},
    #     {"role": "user", "content": "Oh wait, that worked! I see the code now. Thank you!"},
    #     {"role": "assistant", "content": "Great! I'm glad that resolved the issue. Is there anything else I can help you with today?"},
    #     {"role": "user", "content": "No, that's all. Thanks!"}
    # ]
    
    sample_conversation = extract_thread_messages(thread_id="thread_noSA3D5ohhm0RwfohwSfApHF")

    # Extract structured data
    print("Extracting structured data from conversation...")
    structured_data = extract_structured_data(sample_conversation)
    
    # Display results
    print("\n" + "="*80)
    print("STRUCTURED DATA EXTRACTION RESULTS")
    print("="*80 + "\n")
    print(structured_data.model_dump_json(indent=2))
    
    # Save to file
    with open("structured_output.json", "w") as f:
        f.write(structured_data.model_dump_json(indent=2))
    
    print("\n✅ Structured data saved to structured_output.json")


# intent_confidence_score between 0-1 : low scores + escalation = KB gap.
# satisfaction_score
# conversation_id, user_id, session_id - need to check if these are already extracted through other means.