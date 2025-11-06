# from dotenv import load_dotenv
# import os
# from azure.ai.projects import AIProjectClient
# from azure.identity import DefaultAzureCredential

# load_dotenv()

# # Initialize Azure AI Project Client
# project = AIProjectClient(
#     credential=DefaultAzureCredential(),
#     endpoint=os.environ["AZURE_AI_PROJECT"]
# )

# print("Available methods on project.agents.messages:")
# print([method for method in dir(project.agents.messages) if not method.startswith('_')])

# print("\n" + "="*80)
# print("All public methods on messages:")
# for method in dir(project.agents.messages):
#     if not method.startswith('_'):
#         print(f"  - {method}")

# print("\n" + "="*80)
# print("Type of messages object:")
# print(type(project.agents.messages))

import json
from typing import List, Dict
from dotenv import load_dotenv
import os
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

# Load environment variables
load_dotenv()

# Initialize Azure AI Project Client
project = AIProjectClient(
    credential=DefaultAzureCredential(),
    endpoint=os.environ["AZURE_AI_PROJECT"]
)

def extract_thread_messages(thread_id: str, output_file: str = None) -> List[Dict[str, str]]:
    """
    Extract all messages from a thread in a structured format.
    
    Args:
        thread_id: The ID of the thread to extract messages from
        output_file: Optional path to save messages as JSONL file
        
    Returns:
        List of message dictionaries with 'role' and 'content' keys
    """
    try:
        # Retrieve all messages from the thread using the correct method
        messages = project.agents.messages.list(thread_id=thread_id)
        
        # Structure the messages
        structured_messages = []
        
        # Messages are returned in reverse chronological order, so we reverse them
        for message in reversed(list(messages)):
            role = message.role  # 'user' or 'assistant'
            
            # Extract text content from the message
            content_parts = []
            for content_item in message.content:
                if hasattr(content_item, 'text'):
                    # Text content
                    content_parts.append(content_item.text.value)
                elif hasattr(content_item, 'type') and content_item.type == 'text':
                    # Alternative text structure
                    content_parts.append(content_item.text.value)
            
            # Combine all content parts
            full_content = " ".join(content_parts).strip()
            
            if full_content:  # Only add if there's actual content
                structured_messages.append({
                    "role": role,
                    "content": full_content
                })
        
        # Save to JSONL file if specified
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                for msg in structured_messages:
                    f.write(json.dumps(msg, ensure_ascii=False) + '\n')
            print(f"Messages saved to {output_file}")
        
        return structured_messages
        
    except Exception as e:
        print(f"Error extracting messages: {str(e)}")
        raise


def print_conversation(messages: List[Dict[str, str]]):
    """Pretty print the conversation."""
    print("\n" + "="*80)
    print("CONVERSATION THREAD")
    print("="*80 + "\n")
    
    for msg in messages:
        role_label = "🤖 Assistant" if msg['role'] == 'assistant' else "👤 User"
        print(f"{role_label}:")
        print(f"{msg['content']}\n")
        print("-" * 80 + "\n")


# Example usage
if __name__ == "__main__":
    # Specify your thread ID
    thread_id = 'thread_noSA3D5ohhm0RwfohwSfApHF'
    
    # Extract messages
    print(f"Extracting messages from thread: {thread_id}")
    messages = extract_thread_messages(
        thread_id=thread_id,
        output_file="conversation_thread.jsonl"
    )
    
    # Display the conversation
    print(f"\nTotal messages extracted: {len(messages)}")
    print_conversation(messages)
    
    # Example: Print as JSON for inspection
    print("\n" + "="*80)
    print("JSON OUTPUT")
    print("="*80 + "\n")
    print(json.dumps(messages, indent=2, ensure_ascii=False))
    
    # Now you can use this for Pydantic structured output processing
    print("\n✅ Messages ready for Pydantic structured output processing")