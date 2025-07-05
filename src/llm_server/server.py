#!/usr/bin/env python3
"""
LLM Server - A Flask-based REST API using the LLM Python SDK
Supports all major LLM operations including chat, prompts, templates, and model management.
"""

import os
import json
import asyncio
from typing import Dict, Any, Optional, List, Generator
from flask import Flask, request, jsonify, Response, stream_with_context
from werkzeug.exceptions import BadRequest, InternalServerError
import logging
from functools import wraps
import threading

# LLM SDK imports
import llm
from llm import Model, Template, Prompt, get_model_aliases, get_models_with_aliases

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class LLMServer:
    """Main server class that handles LLM SDK operations"""
    
    def __init__(self):
        # Initialize LLM with default settings
        self.default_model = None
        self._setup_models()
    
    def _setup_models(self):
        """Initialize available models"""
        try:
            # Try to get a default model
            model_aliases = get_model_aliases()
            if model_aliases:
                # Get the first available model as default
                self.default_model = list(model_aliases.keys())[0]
                logger.info(f"Default model set to: {self.default_model}")
        except Exception as e:
            logger.warning(f"Could not set default model: {e}")
    
    def get_model(self, model_name: str = None) -> Model:
        """Get a model instance"""
        try:
            if model_name:
                return get_model_aliases()[model_name]
            elif self.default_model:
                return get_model_aliases()[self.default_model]
            else:
                raise ValueError("No model specified and no default model available")
        except Exception as e:
            logger.error(f"Error getting model {model_name}: {e}")
            raise
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models"""
        try:
            models = []
            model_aliases = get_model_aliases()
            for model_id, model in model_aliases.items():
                try:
                    models.append({
                        "id": model_id,
                        "name": getattr(model, 'name', model_id),
                        "provider": getattr(model, 'provider', 'unknown'),
                        "supports_async": hasattr(model, 'prompt_async'),
                        "supports_streaming": hasattr(model, 'prompt') and hasattr(model.prompt, '__call__')
                    })
                except Exception as e:
                    logger.warning(f"Could not get info for model {model_id}: {e}")
                    models.append({
                        "id": model_id,
                        "name": model_id,
                        "provider": "unknown",
                        "error": str(e)
                    })
            return models
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            raise
    
    def generate_response(self, model_name: str, prompt: str, system: str = None, 
                         temperature: float = None, max_tokens: int = None,
                         stream: bool = False) -> Generator[str, None, None]:
        """Generate response from model"""
        try:
            model = self.get_model(model_name)
            
            # Build options
            options = {}
            if temperature is not None:
                options['temperature'] = temperature
            if max_tokens is not None:
                options['max_tokens'] = max_tokens
            
            # Create conversation if system prompt provided
            conversation = None
            if system:
                conversation = model.conversation()
                conversation.system = system
            
            if stream and hasattr(model, 'prompt') and callable(model.prompt):
                # Streaming response
                if conversation:
                    response = conversation.prompt(prompt, **options)
                else:
                    response = model.prompt(prompt, **options)
                
                if hasattr(response, '__iter__'):
                    for chunk in response:
                        if hasattr(chunk, 'text'):
                            yield chunk.text()
                        elif hasattr(chunk, 'content'):
                            yield chunk.content
                        else:
                            yield str(chunk)
                else:
                    yield response.text() if hasattr(response, 'text') else str(response)
            else:
                # Non-streaming response
                if conversation:
                    response = conversation.prompt(prompt, **options)
                else:
                    response = model.prompt(prompt, **options)
                
                if hasattr(response, 'text'):
                    yield response.text()
                elif hasattr(response, 'content'):
                    yield response.content
                else:
                    yield str(response)
                    
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            yield f"Error: {str(e)}"

# Initialize server
llm_server = LLMServer()

def handle_errors(f):
    """Decorator to handle common errors"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except BadRequest as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logger.error(f"Unexpected error in {f.__name__}: {e}")
            return jsonify({"error": f"Internal server error: {str(e)}"}), 500
    return decorated_function

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "llm-server-sdk"})

@app.route('/chat', methods=['POST'])
@handle_errors
def chat():
    """Handle chat requests using LLM SDK"""
    data = request.get_json()
    if not data:
        raise BadRequest("No JSON data provided")
    
    message = data.get('message', '')
    model_name = data.get('model', '')
    system_prompt = data.get('system', '')
    temperature = data.get('temperature')
    max_tokens = data.get('max_tokens')
    stream_response = data.get('stream', False)
    
    if not message:
        raise BadRequest("Message is required")
    
    if stream_response:
        def generate():
            try:
                for chunk in llm_server.generate_response(
                    model_name, message, system_prompt, temperature, max_tokens, stream=True
                ):
                    yield f"data: {json.dumps({'content': chunk})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        return Response(
            stream_with_context(generate()),
            content_type='text/event-stream',
            headers={'Cache-Control': 'no-cache'}
        )
    else:
        response_text = ""
        for chunk in llm_server.generate_response(
            model_name, message, system_prompt, temperature, max_tokens, stream=False
        ):
            response_text += chunk
        
        return jsonify({
            "success": True,
            "response": response_text,
            "model": model_name or "default"
        })

@app.route('/prompt', methods=['POST'])
@handle_errors
def prompt():
    """Handle direct prompts using LLM SDK"""
    data = request.get_json()
    if not data:
        raise BadRequest("No JSON data provided")
    
    prompt_text = data.get('prompt', '')
    model_name = data.get('model', '')
    system_prompt = data.get('system', '')
    temperature = data.get('temperature')
    max_tokens = data.get('max_tokens')
    
    if not prompt_text:
        raise BadRequest("Prompt is required")
    
    response_text = ""
    for chunk in llm_server.generate_response(
        model_name, prompt_text, system_prompt, temperature, max_tokens, stream=False
    ):
        response_text += chunk
    
    return jsonify({
        "success": True,
        "response": response_text,
        "model": model_name or "default"
    })

@app.route('/models', methods=['GET'])
@handle_errors
def list_models():
    """List available models using LLM SDK"""
    models = llm_server.list_models()
    return jsonify({
        "success": True,
        "models": models,
        "count": len(models)
    })

@app.route('/models/<model_name>', methods=['GET'])
@handle_errors
def get_model_info(model_name):
    """Get information about a specific model"""
    try:
        model = llm_server.get_model(model_name)
        info = {
            "id": model_name,
            "name": getattr(model, 'name', model_name),
            "provider": getattr(model, 'provider', 'unknown'),
            "supports_async": hasattr(model, 'prompt_async'),
            "supports_streaming": hasattr(model, 'prompt') and callable(model.prompt),
            "model_class": model.__class__.__name__
        }
        
        # Try to get additional model-specific information
        if hasattr(model, 'options'):
            info['options'] = model.options
        if hasattr(model, 'model_id'):
            info['model_id'] = model.model_id
            
        return jsonify({
            "success": True,
            "model": info
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Model '{model_name}' not found: {str(e)}"
        }), 404

@app.route('/templates', methods=['GET'])
@handle_errors
def list_templates():
    """List available templates using LLM SDK"""
    try:
        # Get templates from LLM's template system
        templates = []
        template_dir = llm.user_dir() / "templates"
        
        if template_dir.exists():
            for template_file in template_dir.glob("*.txt"):
                template_name = template_file.stem
                templates.append({
                    "name": template_name,
                    "path": str(template_file)
                })
        
        return jsonify({
            "success": True,
            "templates": templates,
            "count": len(templates)
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/templates', methods=['POST'])
@handle_errors
def create_template():
    """Create a new template using LLM SDK"""
    data = request.get_json()
    if not data:
        raise BadRequest("No JSON data provided")
    
    name = data.get('name', '')
    content = data.get('content', '')
    
    if not name or not content:
        raise BadRequest("Template name and content are required")
    
    try:
        template_dir = llm.user_dir() / "templates"
        template_dir.mkdir(exist_ok=True)
        
        template_file = template_dir / f"{name}.txt"
        template_file.write_text(content)
        
        return jsonify({
            "success": True,
            "message": f"Template '{name}' created successfully",
            "path": str(template_file)
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/templates/<template_name>', methods=['GET'])
@handle_errors
def get_template(template_name):
    """Get template content using LLM SDK"""
    try:
        template_dir = llm.user_dir() / "templates"
        template_file = template_dir / f"{template_name}.txt"
        
        if not template_file.exists():
            return jsonify({
                "success": False,
                "error": f"Template '{template_name}' not found"
            }), 404
        
        content = template_file.read_text()
        return jsonify({
            "success": True,
            "template": {
                "name": template_name,
                "content": content,
                "path": str(template_file)
            }
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/templates/<template_name>', methods=['DELETE'])
@handle_errors
def delete_template(template_name):
    """Delete a template using LLM SDK"""
    try:
        template_dir = llm.user_dir() / "templates"
        template_file = template_dir / f"{template_name}.txt"
        
        if not template_file.exists():
            return jsonify({
                "success": False,
                "error": f"Template '{template_name}' not found"
            }), 404
        
        template_file.unlink()
        return jsonify({
            "success": True,
            "message": f"Template '{template_name}' deleted successfully"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/conversations', methods=['GET'])
@handle_errors
def list_conversations():
    """List conversation history using LLM SDK"""
    try:
        # Get conversation logs from LLM's database
        logs_db = llm.logs_db()
        conversations = []
        
        # Query recent conversations
        for row in logs_db.execute(
            "SELECT id, model, datetime(datetime, 'localtime') as timestamp FROM responses ORDER BY datetime DESC LIMIT 50"
        ):
            conversations.append({
                "id": row[0],
                "model": row[1],
                "timestamp": row[2]
            })
        
        return jsonify({
            "success": True,
            "conversations": conversations,
            "count": len(conversations)
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/conversations/<int:conversation_id>', methods=['GET'])
@handle_errors
def get_conversation(conversation_id):
    """Get a specific conversation using LLM SDK"""
    try:
        logs_db = llm.logs_db()
        
        # Get conversation details
        conversation = logs_db.execute(
            "SELECT * FROM responses WHERE id = ?", (conversation_id,)
        ).fetchone()
        
        if not conversation:
            return jsonify({
                "success": False,
                "error": "Conversation not found"
            }), 404
        
        return jsonify({
            "success": True,
            "conversation": {
                "id": conversation[0],
                "model": conversation[1],
                "prompt": conversation[2],
                "response": conversation[3],
                "timestamp": conversation[4]
            }
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/embed', methods=['POST'])
@handle_errors
def embed_text():
    """Generate embeddings using LLM SDK"""
    data = request.get_json()
    if not data:
        raise BadRequest("No JSON data provided")
    
    text = data.get('text', '')
    model_name = data.get('model', '')
    
    if not text:
        raise BadRequest("Text is required")
    
    # Try to get an embedding model
    if model_name:
        model = llm_server.get_model(model_name)
    else:
        # Look for embedding models
        model_aliases = get_model_aliases()
        embedding_models = [m for m in model_aliases.keys() if 'embed' in m.lower()]
        if embedding_models:
            model = llm_server.get_model(embedding_models[0])
        else:
            raise ValueError("No embedding model found")
        
    # Generate embedding
    if hasattr(model, 'embed'):
        embedding = model.embed(text)
        return jsonify({
            "success": True,
            "embedding": embedding,
            "model": model_name or "default",
            "dimension": len(embedding) if isinstance(embedding, list) else None
        })
    else:
        return jsonify({
            "success": False,
            "error": "Model does not support embeddings"
        }), 400

@app.route('/plugins', methods=['GET'])
@handle_errors
def list_plugins():
    """List installed plugins using LLM SDK"""
    try:
        plugins = []
        # Get plugins from LLM's plugin system
        for plugin_name in llm.pm.list_plugin_names():
            plugin = llm.pm.get_plugin(plugin_name)
            plugins.append({
                "name": plugin_name,
                "version": getattr(plugin, '__version__', 'unknown'),
                "description": getattr(plugin, '__doc__', ''),
                "enabled": True  # If it's listed, it's enabled
            })
        
        return jsonify({
            "success": True,
            "plugins": plugins,
            "count": len(plugins)
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/version', methods=['GET'])
@handle_errors
def get_version():
    """Get LLM SDK version information"""
    try:
        return jsonify({
            "success": True,
            "llm_version": llm.__version__,
            "server_version": "1.0.0",
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.errorhandler(404)
def handle_not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(405)
def handle_method_not_allowed(e):
    return jsonify({"error": "Method not allowed"}), 405

def main():
    """Main function to start the LLM server"""
    # Configuration
    import argparse
    
    parser = argparse.ArgumentParser(description='LLM Server - A Flask-based REST API using the LLM Python SDK')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind the server to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind the server to (default: 8000)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    print(f"Starting LLM SDK Server on {args.host}:{args.port}")
    print("Available endpoints:")
    print("  POST /chat - Chat with LLM (supports streaming)")
    print("  POST /prompt - Send direct prompt")
    print("  GET /models - List available models")
    print("  GET /models/<name> - Get model information")
    print("  GET /templates - List templates")
    print("  POST /templates - Create template")
    print("  GET /templates/<name> - Get template")
    print("  DELETE /templates/<name> - Delete template")
    print("  GET /conversations - List conversation history")
    print("  GET /conversations/<id> - Get specific conversation")
    print("  POST /embed - Generate text embeddings")
    print("  GET /plugins - List installed plugins")
    print("  GET /version - Get version information")
    print("  GET /health - Health check")
    
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == '__main__':
    main()