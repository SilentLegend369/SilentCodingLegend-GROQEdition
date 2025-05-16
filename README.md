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

![Screenshot From 2025-05-16 12-43-07](https://github.com/user-attachments/assets/b23c63d9-be02-4647-aace-1f44d8b220fc)
![Screenshot From 2025-05-16 12-43-02](https://github.com/user-attachments/assets/4364a1fa-c085-4966-ac27-2ca023a93f72)
![Screenshot From 2025-05-16 12-42-51](https://github.com/user-attachments/assets/62b6b072-2fd0-4670-9b34-f1b344aa3560)
![Screenshot From 2025-05-16 12-42-47](https://github.com/user-attachments/assets/ede74392-7a94-4c11-8724-6f0c5c43d229)
![Screenshot From 2025-05-16 12-42-44](https://github.com/user-attachments/assets/c8a0f43f-6524-47bf-953b-b22b077a9d9e)
![Screenshot From 2025-05-16 12-42-31](https://github.com/user-attachments/assets/2ac9c542-f490-40f2-b338-c8603ad38b86)
![Screenshot From 2025-05-16 12-42-22](https://github.com/user-attachments/assets/672796df-5657-4d22-97a7-836a7159571e)
![Screenshot From 2025-05-16 12-42-06](https://github.com/user-attachments/assets/3a05a5c5-e3db-4640-a6ee-46c7b5d8c62c)
![Screenshot From 2025-05-16 12-42-00](https://github.com/user-attachments/assets/bb664db3-5bb7-46eb-a36f-1a602225c944)
![Screenshot From 2025-05-16 12-41-44](https://github.com/user-attachments/assets/554aa5cb-937b-4a75-b85b-87d2c8f0b8a9)
![Screenshot From 2025-05-16 12-41-41](https://github.com/user-attachments/assets/30461f3e-4074-44c0-82ba-6d6e479b83bf)
![Screenshot From 2025-05-16 12-41-37](https://github.com/user-attachments/assets/85fb030f-d71c-4119-a74e-066a37685056)
![Screenshot From 2025-05-16 12-41-27](https://github.com/user-attachments/assets/9045fa2e-39a4-4301-b8b8-0193353464c0)
![Screenshot From 2025-05-16 12-41-15](https://github.com/user-attachments/assets/acd81cf5-9d90-46f5-933b-655f06b37363)
![Screenshot From 2025-05-16 12-41-02](https://github.com/user-attachments/assets/c4fde36b-1956-4c14-ae3c-e7fabc5b2089)
![Screenshot From 2025-05-16 12-40-50](https://github.com/user-attachments/assets/b832c304-3909-4fc5-8786-7a4d942a9e8d)
![Screenshot From 2025-05-16 12-40-42](https://github.com/user-attachments/assets/3185d6b1-a6c2-427c-8f85-fcb4533314af)
![Screenshot From 2025-05-16 12-40-30](https://github.com/user-attachments/assets/e63642ed-f6f2-4e1e-aa32-7ceff6637ca2)
![Screenshot From 2025-05-16 12-40-21](https://github.com/user-attachments/assets/a7877c19-cd50-49a0-b985-543e0a2590f8)
![Screenshot From 2025-05-16 12-40-14](https://github.com/user-attachments/assets/c3c65ba0-65da-4b8d-99d4-d8db9c92c1eb)
![Screenshot From 2025-05-16 12-40-01](https://github.com/user-attachments/assets/017a5ddf-edd9-4451-8b07-f100da9e4f9f)
![Screenshot From 2025-05-16 12-39-52](https://github.com/user-attachments/assets/790f775e-3858-47b0-8e1a-278380314525)




## Acknowledgments

- Powered by [Groq](https://groq.com)
- Built with [Streamlit](https://streamlit.io)
