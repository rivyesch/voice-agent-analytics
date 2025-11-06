# Voice Agent Analytics

A tool for extracting and analyzing conversation data from Azure AI Foundry agents, transforming unstructured conversations into structured analytics using GPT-4o and Pydantic.

## 🚀 Features

- Extract conversation threads from Azure AI Foundry agents
- Transform unstructured conversations into structured analytics
- Generate actionable insights and metrics
- Type-safe data validation with Pydantic
- Easy integration with Azure AI services

## 📋 Prerequisites

- Python 3.8+
- Azure subscription with AI Foundry access
- Azure AI Project configured
- OpenAI API key (for GPT-4o structured extraction)

## 🔧 Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/rivyesch/voice-agent-analytics.git
   cd voice-agent-analytics
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with the following variables:
   ```env
   # Azure AI Foundry
   AZURE_AI_ENDPOINT=your_ai_endpoint
   AZURE_AI_KEY=your_ai_key
   AZURE_AI_PROJECT=your_project_name
   
   # OpenAI (for structured extraction)
   OPENAI_API_KEY=your_openai_key
   
   # Optional: Configure model and deployment
   OPENAI_MODEL=gpt-4o
   OPENAI_DEPLOYMENT=your_deployment_name
   ```

## 🛠️ Usage

### 1. Extract Thread Messages

Extract raw conversation messages from a specific thread:

```python
from extract_thread_messages import extract_thread_messages

# Replace with your thread ID
thread_id = "your_thread_id"
messages = extract_thread_messages(thread_id)
```

### 2. Extract Structured Analytics

Process the messages to extract structured analytics:

```python
from extract_pydantic_structured_outputs import extract_structured_data

analytics = extract_structured_data(messages)
print(analytics.model_dump_json(indent=2))
```

## 🏗️ Project Structure

```
voice-agent-analytics/
├── extract_thread_messages.py   # Extracts raw messages from Azure AI threads
├── extract_pydantic_structured_outputs.py  # Processes messages into structured data
├── pyproject.toml              # Project metadata and dependencies
└── .env.example                # Example environment variables
```

## 📊 Structured Output Extraction

### What It Does

Transforms raw conversation threads into structured analytics including:
- **Classification**: Request type, categories, subcategories
- **Resolution tracking**: Status, method, first call resolution, escalations
- **Quality metrics**: User sentiment, satisfaction, conversation quality
- **Automation insights**: KB usage, form provision, troubleshooting steps
- **Conversation metrics**: Turn counts, duration, escalation points

### Pydantic Schema

The schema enforces data consistency and type safety:

```python
class ITHelpdeskAnalytics(BaseModel):
    request_type: RequestType  # Must be one of: incident, service_request, etc.
    resolution_status: ResolutionStatus  # Enum ensures consistency
    user_sentiment: UserSentiment  # Validated categories
    issue_keywords: List[str]  # List of technical terms
    # ... more fields
```

### Output Example

```json
{
  "request_type": "incident",
  "incident_category": "multifactor_authentication",
  "resolution_status": "resolved_by_bot",
  "first_call_resolution": true,
  "user_sentiment": "positive"
}
```

## 🤖 Development

### Dependencies

This project uses `uv` for dependency management. The `pyproject.toml` and `uv.lock` files are included for reproducible builds.

To set up the development environment:

```bash
# Install dependencies
uv sync

# Or if you need to update dependencies
uv sync --upgrade
```

### Testing

Run tests to ensure everything works as expected:

```bash
pytest
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  Made with ❤️ for better conversation analytics
</div>
