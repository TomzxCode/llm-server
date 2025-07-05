# LLM Server

A Flask-based REST API server that provides a web interface for the [LLM Python SDK](https://github.com/simonw/llm). This server enables you to interact with various language models through HTTP endpoints, supporting chat, prompts, templates, embeddings, and more.

## Features

- **Chat Interface**: Send messages and receive responses from LLM models
- **Streaming Support**: Real-time streaming responses for chat interactions
- **Model Management**: List and query available models
- **Template System**: Create, manage, and use prompt templates
- **Conversation History**: Access and review past conversations
- **Embeddings**: Generate text embeddings using embedding models
- **Plugin Support**: List and manage installed LLM plugins
- **Health Monitoring**: Built-in health check endpoint

## Prerequisites

- Python 3.12 or higher

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd llm-server
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```

## Usage

### Starting the Server

```bash
# Using the installed script
uv run llm-server
```

Example:
```bash
uv run llm-server --host 0.0.0.0 --port 8000
```

## API Endpoints

### Health Check

**GET** `/health`

Check if the server is running.

**Response:**
```json
{
  "status": "healthy",
  "service": "llm-server-sdk"
}
```

### Chat

**POST** `/chat`

Send a chat message and receive a response.

**Request Body:**
```json
{
  "message": "Hello, how are you?",
  "model": "gpt-4",  // optional
  "system": "You are a helpful assistant.",  // optional
  "temperature": 0.7,  // optional
  "max_tokens": 1000,  // optional
  "stream": false  // optional, for streaming responses
}
```

**Response:**
```json
{
  "success": true,
  "response": "I'm doing well, thank you for asking! How can I help you today?",
  "model": "gpt-4"
}
```

**Streaming Response:**
When `stream: true`, the response is sent as Server-Sent Events:
```
data: {"content": "I'm"}
data: {"content": " doing"}
data: {"content": " well"}
```

### Direct Prompt

**POST** `/prompt`

Send a direct prompt without conversation context.

**Request Body:**
```json
{
  "prompt": "Explain quantum computing in simple terms",
  "model": "gpt-4",  // optional
  "temperature": 0.7,  // optional
  "max_tokens": 500  // optional
}
```

### Models

**GET** `/models`

List all available models.

**Response:**
```json
{
  "success": true,
  "models": [
    {
      "id": "gpt-4",
      "name": "GPT-4",
      "provider": "openai",
      "supports_async": true,
      "supports_streaming": true
    }
  ]
}
```

**GET** `/models/{model_name}`

Get detailed information about a specific model.

### Templates

**GET** `/templates`

List all available templates.

**POST** `/templates`

Create a new template.

**Request Body:**
```json
{
  "name": "code_review",
  "prompt": "Review this code for best practices:\n\n{code}",
  "description": "Template for code review"
}
```

**GET** `/templates/{template_name}`

Get a specific template.

**DELETE** `/templates/{template_name}`

Delete a template.

### Conversations

**GET** `/conversations`

List recent conversations (last 50).

**GET** `/conversations/{conversation_id}`

Get details of a specific conversation.

### Embeddings

**POST** `/embed`

Generate embeddings for text.

**Request Body:**
```json
{
  "text": "This is the text to embed",
  "model": "text-embedding-ada-002"  // optional
}
```

**Response:**
```json
{
  "success": true,
  "embedding": [0.1, 0.2, 0.3, ...],
  "model": "text-embedding-ada-002",
  "dimension": 1536
}
```

### Plugins

**GET** `/plugins`

List installed plugins.

**Response:**
```json
{
  "success": true,
  "plugins": [
    {
      "name": "llm-embed-all",
      "version": "0.1.0",
      "description": "Plugin for generating embeddings",
      "enabled": true
    }
  ]
}
```

### Version

**GET** `/version`

Get version information.

**Response:**
```json
{
  "success": true,
  "llm_version": "0.26.0",
  "server_version": "1.0.0",
  "python_version": "3.12.0"
}
```

## Error Handling

All endpoints return consistent error responses:

```json
{
  "error": "Error message description"
}
```

Common HTTP status codes:
- `400`: Bad Request (invalid input)
- `404`: Not Found (resource doesn't exist)
- `405`: Method Not Allowed (wrong HTTP method)
- `500`: Internal Server Error (server-side error)

## Development

### Running Tests

```bash
# Install development dependencies
uv sync --dev

# Run tests
pytest
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built on top of the excellent [LLM Python SDK](https://github.com/simonw/llm) by Simon Willison
- Uses Flask for the web framework
- Supports all LLM SDK features including models, templates, and plugins
