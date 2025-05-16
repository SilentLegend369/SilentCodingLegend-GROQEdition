# SilentCodingLegend AI Assistant

![Powered by Groq](https://groq.com/wp-content/uploads/2024/03/PBG-mark1-color.svg)

A powerful AI assistant built with Streamlit and powered by Groq's ultra-fast LLM inference API.

## Features

- 💬 **Smart Chat Interface**: Engage in natural conversations with multiple Groq LLM models
- 👁️ **VisionAI**: Upload and analyze images with multimodal capabilities
- 📚 **Knowledge Base**: Upload documents and query their contents
- 🧠 **DeepThinker**: Solve complex problems with step-by-step reasoning
- 📱 **Mobile Optimization**: Responsive design that works across devices
- 📊 **Model Metrics**: Track performance and costs across different models
- 💻 **Advanced Tools**: Code execution, batch processing, and API integration
- 🔄 **Chat Templates**: Predefined conversation templates for specific use cases
- ⚡ **Performance Optimizations**: Response caching for improved efficiency

## Prerequisites

- Python 3.9+
- [Groq API Key](https://console.groq.com/keys)
- Internet connection

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/scl-groq.git
   cd scl-groq
   ```

2. Run the setup script:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

   This will:
   - Create a Python virtual environment
   - Install required dependencies
   - Set up necessary directories
   - Guide you through configuration

3. Set up your Groq API key:
   - Create a `.env` file in the project root:
     ```
     GROQ_API_KEY=your_groq_api_key_here
     ```

## Usage

1. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

2. Start the application:
   ```bash
   streamlit run app.py
   ```

3. Open your browser and navigate to the URL displayed in the terminal (typically http://localhost:8501)

## Features in Detail

### Main Chat Interface

The primary interface for interacting with Groq's LLMs. Select models, adjust parameters, and use templates to guide conversations.

### VisionAI

Upload images and get AI-generated descriptions and analyses. VisionAI can:
- Describe image contents
- Answer questions about images
- Analyze specific regions of images
- Compare and annotate images

### Knowledge Base

Upload documents (PDF, DOCX, TXT) and query their contents. The system will:
- Process documents into chunks
- Find relevant information based on your questions
- Maintain conversation history for each document

### DeepThinker

Specialized for complex reasoning tasks like:
- Mathematical problems
- Logical deduction
- Computer science problems
- Critical analysis

### Model Metrics

Track and analyze your usage of different models, including:
- Response times
- Token usage
- Cost estimates
- Success rates
- Model comparisons

## Configuration Options

You can customize the application through:

- `src/config.py`: Model settings, file paths, and application constants
- `.env` file: API keys and sensitive information
- Chat Templates: Create and save custom templates for specific use cases

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Powered by [Groq](https://groq.com)
- Built with [Streamlit](https://streamlit.io)
