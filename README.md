# Multilingual Voice Chatbot with AWS Bedrock and SARVAM AI

A Streamlit-based multilingual voice chatbot that integrates AWS Bedrock's Claude 3 Sonnet model with SARVAM AI's speech-to-text capabilities. The application supports voice input, document uploads, image processing, and specialized government scheme information retrieval.

## Features

- **Voice Input**: Record audio messages transcribed using SARVAM AI's multilingual speech-to-text
- **Text Chat**: Traditional text-based conversation interface
- **Document Upload**: Support for PDF, DOCX, TXT, CSV, and JSON files (up to 5 documents, 4.5MB each)
- **Image Upload**: Support for PNG, JPG, JPEG, GIF, BMP, WEBP formats (up to 20 images, 3.75MB each)
- **Government Scheme Information**: Specialized tool for retrieving information about government schemes and MSME programs
- **Multilingual Support**: Responds in the same language as user input
- **Claude 3 Sonnet**: Powered by Anthropic's Claude 3 Sonnet model via AWS Bedrock

## Prerequisites

- Python 3.13
- FFmpeg (required for audio processing)
- AWS account with access to:
  - AWS Bedrock (Claude 3 Sonnet model)
  - AWS Bedrock Agents (for government scheme information)
- SARVAM AI API key
- Configured AWS credentials

### Install FFmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
1. Download FFmpeg from https://ffmpeg.org/download.html
2. Extract to a folder (e.g., `C:\ffmpeg`)
3. Add `C:\ffmpeg\bin` to your system PATH environment variable

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd multilingual-chatbot
```

### 2. Create Virtual Environment

```bash
python3.13 -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure AWS Credentials

Set up your AWS credentials using one of these methods:

**Option A: AWS CLI**
```bash
aws configure
```

**Option B: Environment Variables**
```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=your_preferred_region
```

**Option C: IAM Role** (recommended for EC2/ECS deployments)

### 5. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
touch .env
```

Add the following variables to `.env`:

```env
# SARVAM AI Configuration
SARVAM_API_KEY=your_sarvam_api_key_here

# AWS Bedrock Agent Configuration
BEDROCK_AGENT_ID=your_bedrock_agent_id
BEDROCK_AGENT_ALIAS_ID=your_bedrock_agent_alias_id

# AWS Bedrock Model Configuration
BEDROCK_MODEL_ID=your_bedrock_model_id
AWS_DEFAULT_REGION=your_aws_region

# System Prompt Configuration (optional - customize assistant behavior)
SYSTEM_PROMPT=You are a helpful assistant. Follow these rules strictly: 1) For ANY question about government schemes, budgets, MSME, subsidies, grants, policies, or government programs - ALWAYS use the government_scheme_info tool FIRST, even if documents are uploaded. 2) Only after using the tool, you may supplement with document information if relevant. 3) For non-government questions, use uploaded documents normally or your own knowledge. 4) Always respond in the user's language. 5) Never skip the government_scheme_info tool for government-related queries. 6) NEVER explain your tool usage decisions or mention tools in your responses - provide only the final answer directly.

```

## Getting Required Credentials

### SARVAM AI API Key

Get your API key from [SARVAM AI Console](https://dashboard.sarvam.ai/)

### AWS Bedrock Agent Configuration

1. **Create Bedrock Agent** (if not exists):
   - Go to AWS Bedrock Console
   - Navigate to Agents
   - Create a new agent for government scheme information
   - Note the Agent ID and Alias ID

2. **Get Agent IDs**:
   - Agent ID: Found in Bedrock Agents console
   - Agent Alias ID: Found in the agent's alias configuration

### Required AWS Permissions

Ensure your AWS credentials have the following permissions:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:InvokeAgent"
            ],
            "Resource": [
                "arn:aws:bedrock:*:*:agent/*",
                "arn:aws:bedrock:*:*:model/*"
            ]
        }
    ]
}
```

## Usage

### 1. Start the Application

```bash
streamlit run app.py
```

### 2. Access the Application

Open your web browser and navigate to `http://localhost:8501`

### 3. Using the Application

- **Voice Input**: Click "Click to record" to start recording, click again to stop
- **Text Input**: Type messages in the chat input field
- **File Upload**: Use the sidebar to upload documents and images
- **Clear Chat**: Use the "Clear Conversation" button to reset the chat history

**Note**: Chat history is automatically limited to the last 50 messages to prevent memory issues.

## Project Structure

```
multilingual-chatbot/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables (create this)
├── README.md             # This file
└── myenv/                # Virtual environment (created during setup)
```

## Configuration Options

### System Prompt Customization

Customize the assistant's behavior by modifying the `SYSTEM_PROMPT` variable in your `.env` file.

### File Upload Limits

Current limits (configurable in `app.py`):
- Documents: 5 files, 4.5MB each
- Images: 20 files, 3.75MB each

## Troubleshooting

### Common Issues

1. **Audio transcription fails**:
   - Verify SARVAM_API_KEY is correct
   - Check internet connectivity
   - Ensure audio file is in supported format

2. **AWS Bedrock errors**:
   - Verify AWS credentials are configured
   - Check Bedrock model access in your AWS region
   - Ensure proper IAM permissions

3. **File upload issues**:
   - Check file size limits
   - Verify file format is supported
   - Ensure sufficient disk space

### Environment Variables Not Loading

If environment variables aren't loading:

1. Verify `.env` file is in the project root
2. Check file permissions
3. Restart the Streamlit application

## Security Considerations

- Keep your `.env` file secure and never commit it to version control
- Use IAM roles instead of access keys when possible
- Regularly rotate API keys and credentials
- Monitor AWS CloudTrail for API usage

## Dependencies

Key dependencies include:
- `streamlit==1.45.0` - Web framework
- `boto3==1.38.8` - AWS SDK
- `pydub==0.25.1` - Audio processing
- `streamlit-audiorecorder==0.0.6` - Audio recording component
- `python-dotenv==1.1.1` - Environment variable loading
- `requests==2.32.3` - HTTP requests for SARVAM API
- `audioop-lts==0.2.1` - Python 3.13 audio compatibility

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review AWS Bedrock documentation
3. Check SARVAM AI API documentation
4. Open an issue in the repository