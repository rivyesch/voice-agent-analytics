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
    INCIDENT = "incident"  # Troubleshooting issues
    SERVICE_REQUEST = "service_request"  # Form guidance, access requests
    GENERAL_INQUIRY = "general_inquiry"  # Questions, information
    OUT_OF_SCOPE = "out_of_scope"  # Non-IT topics


class IncidentCategory(str, Enum):
    """Specific incident categories aligned with KB structure"""
    UNIFLOW_PRINTER = "uniflow_printer"  # Printer setup, PIN reset, printing issues
    CITRIX = "citrix"  # Login, setup, access, Workspace app
    MFA = "multifactor_authentication"  # MFA setup, lost device, passkeys
    AD_ACCOUNT = "ad_account"  # Account unlock, password reset, expired account
    HARDWARE = "hardware"  # Desktop, laptop, peripheral issues
    SOFTWARE_APPLICATION = "software_application"  # Other application issues
    OTHER = "other"  # Other IT issues
    NOT_APPLICABLE = "not_applicable"

class ServiceRequestType(str, Enum):
    """Types of service requests from system prompt"""
    SAP_ACCESS = "sap_access"  # SAP Access Request form
    EMAIL_SHARED_DRIVE = "email_shared_drive"  # Email & Share Drive Request form
    BUSINESS_APP_ACCESS = "business_app_access"  # Core Business Applications Access form
    NETWORK_ACCESS = "network_access"  # Network Access Order Guide form
    GENERAL_CATALOG = "general_catalog"  # Generic catalog fallback
    OTHER_FORM = "other_form"
    NOT_APPLICABLE = "not_applicable"

class ResolutionMethod(str, Enum):
    """How the issue was handled - For process improvement tracking"""
    KB_GUIDED_TROUBLESHOOTING = "kb_guided_troubleshooting"  # Followed KB steps
    FORM_PROVIDED = "form_provided"  # Directed to service request form
    SIMPLE_INFORMATION = "simple_information"  # Quick answer without KB/form
    ESCALATED = "escalated"  # Created ticket for agent
    NO_RESOLUTION = "no_resolution"  # Could not resolve
    NOT_APPLICABLE = "not_applicable"

class ResolutionStatus(str, Enum):
    """Final outcome of the conversation - Critical for FCR and Automation Success"""
    RESOLVED_BY_BOT = "resolved_by_bot"  # Issue fixed without escalation
    RESOLVED_WITH_FORM = "resolved_with_form"  # User directed to correct form
    ESCALATED_TO_HUMAN = "escalated_to_human"  # Ticket created and left open for transfer to agent
    USER_ABANDONED = "user_abandoned"  # User left before completion
    OUT_OF_SCOPE = "out_of_scope"  # Non-IT issue
    BOT_FAILURE = "bot_failure"  # Bot went silent or failed to respond

class EscalationReason(str, Enum):
    """Why escalation occurred - Critical for KB gap identification"""
    KB_STEPS_FAILED = "kb_steps_failed"  # Troubleshooting didn't work
    NO_KB_ARTICLE_FOUND = "no_kb_article_found"  # Missing KB content
    KB_STEPS_INFEASIBLE = "kb_steps_infeasible"  # User can't perform steps
    USER_REQUESTED_HUMAN = "user_requested_human"  # Explicit request for agent
    USER_FRUSTRATED = "user_frustrated"  # Frustration during conversation
    COMPLEX_ISSUE = "complex_issue"  # Beyond bot capability
    URGENT_REQUEST = "urgent_request"  # Marked as urgent/emergency
    AUTHENTICATION_REQUIRED = "authentication_required"  # Backend access needed
    NOT_ESCALATED = "not_escalated"

class ConversationQuality(str, Enum):
    """Bot performance quality - For Voice Assistant Quality metric"""
    EXCELLENT = "excellent"  # Perfect protocol adherence, efficient
    GOOD = "good"  # Minor issues, still effective
    ACCEPTABLE = "acceptable"  # Some problems but got there eventually
    POOR = "poor"  # Significant protocol violations
    FAILED = "failed"  # Major failures, bot went silent
    

class UserSentiment(str, Enum):
    """User sentiment - For Employee Satisfaction scoring"""
    VERY_SATISFIED = "very_satisfied"  # Explicitly happy, thanked profusely
    SATISFIED = "satisfied"  # Positive, issue resolved
    NEUTRAL = "neutral"  # No strong emotion either way
    DISSATISFIED = "dissatisfied"  # Negative, some frustration
    VERY_DISSATISFIED = "very_dissatisfied"  # Clearly frustrated or angry 

class BotFailureType(str, Enum):
    """Specific types of bot failures - For debugging and improvement"""
    WENT_SILENT = "went_silent"  # Bot stopped responding after question
    MISSED_KB_SEARCH = "missed_kb_search"  # Didn't search KB when required
    MULTIPLE_QUESTIONS = "multiple_questions"  # Asked multiple questions at once
    WRONG_FORM_PROVIDED = "wrong_form_provided"  # Provided incorrect form
    PROTOCOL_VIOLATION = "protocol_violation"  # Other system prompt violations
    NONE = "none"

class CallEndReason(str, Enum):
    RESOLVED_NORMAL_CLOSE = "resolved_normal_close"  # Issue resolved, proper goodbye
    ESCALATED_CLOSE = "escalated_close"  # Ticket created, call ended
    FORM_PROVIDED_CLOSE = "form_provided_close"  # Form sent, call ended
    USER_ABANDONED = "user_abandoned"  # User stopped responding
    USER_DISCONNECTED = "user_disconnected"  # Technical disconnection
    USER_REQUESTED_END = "user_requested_end"  # User said goodbye without resolution
    BOT_FAILURE = "bot_failure"  # Bot went silent/crashed
    OUT_OF_SCOPE_REDIRECT = "out_of_scope_redirect"  # Non-IT, redirected elsewhere
    URGENT_HANDOFF = "urgent_handoff"  # Urgent escalation quick close

# ============================================================================
# MAIN SCHEMA
# ============================================================================

class ConversationAnalytics(BaseModel):
    """
    Comprehensive conversation analytics schema designed for dashboard metrics.
    
    This schema extracts structured data from IT helpdesk conversations to power:
    - Employee Satisfaction Score
    - First Call Resolution (FCR)
    - Time to Resolution
    - Automation Success Rate
    - KB Gap Identification
    - Top Issues/Intents
    - Escalation Analysis
    """

    # ========================================================================
    # REQUEST CLASSIFICATION - For "Top Issues/Intents" dashboard
    # ========================================================================

    request_type: RequestType = Field(
        description="Primary classification: incident, service_request, general_inquiry, or out_of_scope"
    )
    
    incident_category: IncidentCategory = Field(
        description="Specific incident category if applicable (uniflow_printer, citrix, mfa_authenticator, ad_account, etc.)"
    )
    
    service_request_type: ServiceRequestType = Field(
        description="Specific service request type if applicable (sap_access, email_shared_drive, etc.)"
    )
    
    issue_summary: str = Field(
        max_length=200,
        description="One-sentence summary of the issue/request. Example: 'User unable to print to Uniflow after PIN reset'"
    )
    
    issue_keywords: List[str] = Field(
        default_factory=list,
        description="Key technical terms from conversation (max 5). Example: ['uniflow', 'pin reset', 'printer']"
    )
    
    # ========================================================================
    # RESOLUTION TRACKING - For FCR, Automation Success, Time to Resolution
    # ========================================================================
    
    resolution_status: ResolutionStatus = Field(
        description="""Final outcome:
        - resolved_by_bot: Issue fixed without escalation (counts as FCR + Automation Success)
        - resolved_with_form: Form provided successfully (counts as FCR + Automation Success)
        - escalated_to_agent: Ticket created for follow-up (NOT FCR, NOT Automation Success)
        - user_abandoned: User left before completion (NOT FCR)
        - out_of_scope: Non-IT issue (NOT counted in metrics)
        - bot_failure: Bot went silent/failed (counts as failure)
        """
    )
    
    resolution_method: ResolutionMethod = Field(
        description="How it was handled: kb_guided_troubleshooting, form_provided, simple_information, escalated, no_resolution"
    )
    
    resolution_provided: Optional[str] = Field(
        None,
        max_length=300,
        description="Brief summary of solution if resolved. Example: 'Guided user to reset Uniflow PIN via security settings'"
    )

    # ========================================================================
    # ESCALATION ANALYSIS - For "Escalation Reasons" dashboard & KB Gap ID
    # ========================================================================
    
    escalation_reason: EscalationReason = Field(
        description="""Primary reason for escalation - CRITICAL for KB gap identification:
        - kb_steps_failed: Steps didn't work (need better KB)
        - no_kb_article_found: No KB content (definite gap)
        - kb_steps_infeasible: Steps too complex (need simplification)
        - user_requested_human: Explicit request
        - user_frustrated: Escalated due to frustration
        - complex_issue: Beyond bot scope
        - urgent_request: Emergency handling
        - authentication_required: Backend access needed
        """
    )
    
    escalation_turn_number: Optional[int] = Field(
        None,
        description="Turn number when escalation occurred (for efficiency analysis)"
    )

    # ========================================================================
    # KB EFFECTIVENESS - For KB Gap Identification & Process Improvement
    # ========================================================================
    
    kb_search_performed: bool = Field(
        description="Whether KB search tool was called (per system prompt requirement)"
    )
    
    kb_article_found: bool = Field(
        description="Whether a relevant KB article was returned from search"
    )
    
    kb_steps_attempted: List[str] = Field(
        default_factory=list,
        description="List of troubleshooting steps from KB that were attempted. Example: ['Reset PIN', 'Clear cache', 'Reinstall app']"
    )
    
    kb_steps_count: int = Field(
        default=0,
        description="Number of distinct troubleshooting steps attempted"
    )
    
    kb_steps_successful: bool = Field(
        description="Whether KB steps resolved the issue (TRUE = good KB, FALSE = potential gap)"
    )
    
    # ========================================================================
    # FORM HANDLING - For "Generic Form Usage" dashboard
    # ========================================================================
    
    form_provided: bool = Field(
        description="Whether a service request form was provided to user"
    )
    
    form_type_provided: Optional[ServiceRequestType] = Field(
        None,
        description="Which form was provided (if applicable)"
    )
    
    form_url_sent: Optional[str] = Field(
        None,
        description="The form URL that was sent via Teams"
    )
    
    correct_form_provided: bool = Field(
        default=True,
        description="Whether the bot identified and provided the CORRECT form (FALSE = process failure)"
    )

    # ========================================================================
    # TICKET TRACKING - Links to server-side data
    # ========================================================================
    
    ims_ticket_created: bool = Field(
        description="Whether an IMS interaction ticket was created (should be TRUE for all IT calls per protocol)"
    )
    
    ims_ticket_number: Optional[str] = Field(
        None,
        description="IMS ticket number if created"
    )
    
    inc_ticket_created: bool = Field(
        description="Whether an INC incident ticket was created (indicates escalation)"
    )
    
    inc_ticket_number: Optional[str] = Field(
        None,
        description="INC ticket number if created"
    )

    # ========================================================================
    # CONVERSATION METRICS - For Time/Duration analysis
    # ========================================================================
    
    total_turns: int = Field(
        description="Total conversation turns (user + assistant messages)"
    )
    
    user_turns: int = Field(
        description="Number of user messages"
    )
    
    assistant_turns: int = Field(
        description="Number of assistant messages"
    )
    
    conversation_ended_naturally: bool = Field(
        description="""TRUE if conversation ended with proper closing (bot farewell or user goodbye).
        FALSE if bot went silent, user abandoned, or technical disconnection."""
    )
    
    # ========================================================================
    # USER EXPERIENCE - For Employee Satisfaction Score (PRIMARY KPI)
    # ========================================================================
    
    user_sentiment: UserSentiment = Field(
        description="""Overall user sentiment throughout conversation:
        - very_satisfied: Explicitly happy, thanked multiple times
        - satisfied: Positive, issue resolved, no complaints
        - neutral: No strong emotion
        - dissatisfied: Some frustration, negative comments
        - very_dissatisfied: Clearly frustrated or angry
        
        This feeds the Employee Satisfaction Score dashboard KPI."""
    )
    
    satisfaction_score: int = Field(
        ge=1,
        le=5,
        description="""Numeric satisfaction score 1-5 for dashboard:
        5 = Very Satisfied
        4 = Satisfied
        3 = Neutral
        2 = Dissatisfied
        1 = Very Dissatisfied
        
        Maps directly to UserSentiment enum for Employee Satisfaction Score metric."""
    )
    
    user_expressed_satisfaction: Optional[bool] = Field(
        None,
        description="TRUE if user explicitly said thanks/expressed satisfaction. FALSE if expressed dissatisfaction. None if unclear."
    )
    
    user_expressed_frustration: bool = Field(
        description="Whether user showed signs of frustration during conversation"
    )
    
    frustration_triggers: List[str] = Field(
        default_factory=list,
        description="What caused frustration if applicable. Example: ['repeated failed steps', 'bot went silent', 'unclear instructions']"
    )

    call_end_reason: CallEndReason = Field(
        description="Specific reason how/why the call ended - for 'Reason Call Ended' dashboard"
    )

    # ========================================================================
    # BOT PERFORMANCE - For Voice Assistant Quality dashboard
    # ========================================================================
    
    conversation_quality: ConversationQuality = Field(
        description="""Overall quality of bot handling:
        - excellent: Perfect protocol, efficient, clear
        - good: Minor issues but effective
        - acceptable: Some problems but resolved eventually
        - poor: Significant protocol violations
        - failed: Major failures, bot went silent
        
        This feeds Voice Assistant Quality metric."""
    )
    
    quality_score: int = Field(
        ge=1,
        le=5,
        description="""Numeric quality score 1-5 for dashboard:
        5 = Excellent
        4 = Good
        3 = Acceptable
        2 = Poor
        1 = Failed
        
        Maps to ConversationQuality enum for Voice Assistant Quality dashboard."""
    )
    
    bot_followed_protocol: bool = Field(
        description="""Whether bot followed system prompt protocols:
        - Called KB tool before troubleshooting
        - Asked one question at a time
        - Collected triage details before creating tickets
        - Used proper closing protocol
        """
    )
    
    bot_failure_occurred: bool = Field(
        description="Whether any bot failure/malfunction occurred during conversation"
    )
    
    bot_failure_type: BotFailureType = Field(
        description="""Type of bot failure if applicable:
        - went_silent: Bot stopped responding (from your 64% analysis)
        - missed_kb_search: Didn't search KB when required
        - multiple_questions: Asked multiple questions at once (protocol violation)
        - wrong_form_provided: Provided incorrect form
        - protocol_violation: Other system prompt violations
        """
    )
    
    protocol_violations: List[str] = Field(
        default_factory=list,
        description="Specific protocol violations if any. Example: ['skipped KB search', 'asked 3 questions at once', 'no closing protocol']"
    )

    # ========================================================================
    # URGENT REQUEST HANDLING
    # ========================================================================
    
    urgent_request: bool = Field(
        description="Whether user indicated urgency (urgent, emergency, critical, need help now)"
    )
    
    urgent_handled_correctly: Optional[bool] = Field(
        None,
        description="If urgent: was it handled per urgent protocol (quick triage, ticket, close)?"
    )

    # ========================================================================
    # MULTI-ISSUE & EDGE CASES
    # ========================================================================
    
    multi_issue_conversation: bool = Field(
        description="Whether user raised multiple separate issues in one conversation"
    )
    
    secondary_issues: List[str] = Field(
        default_factory=list,
        description="List of secondary issues raised if multi_issue_conversation=True"
    )

    third_party_request: bool = Field(
        description="Whether caller was calling on behalf of someone else (affects user identification)"
    )
    
    technical_confusion: bool = Field(
        description="Whether user was confused by technical terms or jargon"
    )

def extract_structured_data(conversation_messages: List[dict]) -> ConversationAnalytics:
    """
    Extract structured data from conversation messages using GPT 4.1.
    
    Args:
        conversation_messages: List of dicts with 'role' and 'content' keys
        
    Returns:
        ConversationAnalytics object with extracted data
    """
    # Format the conversation for the prompt
    conversation_text = "\n".join([
        f"{msg['role'].upper()}: {msg['content']}" 
        for msg in conversation_messages
    ])
    
    # Create the extraction prompt
    system_prompt = """You are an expert IT helpdesk conversation analyst specializing in:
    - Detecting bot failures
    - Acurately assessing user sentiment and satisfaction 
    - Identifying knowledge base gaps and resolution outcomes
    
    Use the full 1-5 scoring range. Be conservative - high scores must be earned.
    
    Extract all relevant fields according to the schema provided.
    """
    
    user_prompt = f"""Analyze this IT helpdesk conversation and extract structured information:

    {conversation_text}

    **IMPORTANT NOTES:**
    - Analyze the ENTIRE conversation to determine the outcome
    - Pay attention to the LAST few message to understand how it ended
    - If bot asked a question but never responded after user's answer, that's a bot failure
    - User saying "hello" or "are you there? after bot question = likely bot went silent
    - Be conservative with satisfaction/quality score - must be earned
    - Separate IMS tickets (always created for IT calls) from INC tickets (esclations only)
    """
    
    # Call GPT-4o with structured output
    completion = client.beta.chat.completions.parse(
        model=os.environ["MODEL_DEPLOYMENT_NAME"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format=ConversationAnalytics
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