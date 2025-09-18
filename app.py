import streamlit as st
import boto3
import logging
import re
import uuid
import json
import io
import os
import requests
from botocore.exceptions import ClientError
from audiorecorder import audiorecorder
from pydub import AudioSegment
from dotenv import load_dotenv

load_dotenv()

# ---------- CONFIGURATION ----------
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"
BEDROCK_AGENT_ID = os.getenv("BEDROCK_AGENT_ID")
BEDROCK_AGENT_ALIAS_ID = os.getenv("BEDROCK_AGENT_ALIAS_ID")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT")

# ---------- LOGGING ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- SESSION INIT ----------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Limit chat history to prevent memory issues
if len(st.session_state.chat_history) > 50:
    st.session_state.chat_history = st.session_state.chat_history[-50:]
if "last_uploaded_docs" not in st.session_state:
    st.session_state.last_uploaded_docs = []
if "last_uploaded_images" not in st.session_state:
    st.session_state.last_uploaded_images = []
if "enable_tts" not in st.session_state:
    st.session_state.enable_tts = False
if "tts_language" not in st.session_state:
    st.session_state.tts_language = "hi-IN"

# ---------- HELPERS ----------
def sanitize_filename(filename):
    name_without_ext = filename.rsplit('.', 1)[0]
    sanitized = re.sub(r"[^A-Za-z0-9\s\-\(\)\[\]]", "_", name_without_ext)
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    return sanitized

def tool_config():
    return {
        "tools": [
            {
                "toolSpec": {
                    "name": "government_scheme_info",
                    "description": "MANDATORY: Always use this tool for ANY question about: government schemes, budgets, MSME, subsidies, grants, policies, ministries, or government programs. Use this tool FIRST before analyzing any uploaded documents when the question relates to government topics. Do not attempt to answer government-related questions from uploaded documents alone.",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The exact user question about government schemes, budgets, MSME, subsidies, grants, or policies."
                                }
                            },
                            "required": ["query"]
                        }
                    }
                }
            }
        ]
    }

def decode_unicode_text(text: str) -> str:
    try:
        if "\\u" in text:
            return bytes(text, "utf-8").decode("unicode_escape")
        return text
    except Exception as e:
        logging.warning(f"Failed to decode unicode text: {e}")
        return text

def get_government_scheme_info(query: str) -> dict:
    if not BEDROCK_AGENT_ID or not BEDROCK_AGENT_ALIAS_ID:
        return {"error": "Bedrock agent configuration missing"}
        
    bedrock_agent_client = boto3.client("bedrock-agent-runtime")
    session_id = str(uuid.uuid4())

    result = ''
    try:
        response = bedrock_agent_client.invoke_agent(
            agentId=BEDROCK_AGENT_ID,
            agentAliasId=BEDROCK_AGENT_ALIAS_ID,
            sessionId=session_id,
            inputText=query
        )
        for event in response.get("completion"):
            chunk = event["chunk"]
            result += chunk["bytes"].decode()
    except Exception as e:
        logger.exception("Failed to invoke Bedrock agent.")
        return {"error": "Failed to retrieve information due to internal error."}

    if not result:
        return {"error": "No relevant information found for your query."}

    return {"detail": result}

def transcribe_audio(file_path: str) -> str:
    if not SARVAM_API_KEY:
        raise Exception("SARVAM API key not configured")
    
    # Validate file exists
    if not os.path.exists(file_path):
        raise Exception(f"Audio file not found: {file_path}")
        
    headers = {
        "api-subscription-key": SARVAM_API_KEY
    }
    
    try:
        with open(file_path, "rb") as f:
            files = {"file": ("audio.wav", f, "audio/wav")}
            response = requests.post(SARVAM_STT_URL, headers=headers, files=files, timeout=45)
            response.raise_for_status()  # Raise exception for HTTP error status codes
        
        result = response.json()
        transcript = result.get("transcript", "")
        if not transcript:
            raise Exception("Empty transcript received from SARVAM API")
        return transcript
            
    except requests.exceptions.HTTPError as e:
        error_msg = f"HTTP error during audio transcription: {e.response.status_code}"
        try:
            error_detail = e.response.json().get("message", "Unknown error")
            error_msg += f" - {error_detail}"
        except:
            error_msg += f" - {e.response.text}"
        raise Exception(error_msg)
    except requests.exceptions.Timeout:
        raise Exception("Audio transcription timed out after 45 seconds")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error during audio transcription: {str(e)}")
    except Exception as e:
        if "Audio transcription" in str(e) or "HTTP error" in str(e):
            raise  # Re-raise our custom exceptions
        raise Exception(f"Unexpected error during audio transcription: {str(e)}")

def text_to_speech(text: str, language: str = "hi-IN") -> bytes:
    if not SARVAM_API_KEY:
        raise Exception("SARVAM API key not configured")
        
    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json"
    }
    
    data = {
        "text": text,
        "target_language_code": language
    }
    
    response = requests.post(SARVAM_TTS_URL, headers=headers, json=data, timeout=45)
    response.raise_for_status()  # Raise exception for HTTP error status codes
    
    result = response.json()
    audio_data = result.get("audios", [])[0] if result.get("audios") else None
    if audio_data:
        import base64
        return base64.b64decode(audio_data)
    else:
        raise Exception("No audio data in response")

def generate_message(bedrock_client, model_id, input_text, documents, images, chat_history=None):
    if chat_history is None:
        chat_history = st.session_state.chat_history

    # Validate input
    if not input_text or not input_text.strip():
        raise ValueError("Input text cannot be empty")
        
    content_blocks = [{"text": input_text}]
    for doc in documents[:5]:
        if doc.size > 4.5 * 1024 * 1024:
            raise ValueError(f"Document '{doc.name}' exceeds 4.5MB limit.")
        doc_format = doc.name.split('.')[-1].lower()
        if doc_format not in ['txt', 'pdf', 'docx', 'csv', 'json']:
            raise ValueError(f"Unsupported document format: {doc_format}")
        doc_name = sanitize_filename(doc.name)
        doc.seek(0)  # Reset file pointer
        content_blocks.append({
            "document": {
                "name": doc_name,
                "format": doc_format,
                "source": {"bytes": doc.read()}
            }
        })
    for img in images[:20]:
        if img.size > 3.75 * 1024 * 1024:
            raise ValueError(f"Image '{img.name}' exceeds 3.75MB limit.")
        img_format = img.type.split('/')[-1]
        img_name = sanitize_filename(img.name)
        img.seek(0)  # Reset file pointer
        content_blocks.append({
            "image": {
                "format": img_format,
                "source": {"bytes": img.read()}
            }
        })

    messages = []
    for message in chat_history:
        if 'text' in message:
            messages.append({
                "role": message['role'],
                "content": [{"text": message['text']}]
            })

    messages.append({
        "role": "user",
        "content": content_blocks
    })
    logger.info(f"Processing message with {len(content_blocks)} content blocks")

    response = bedrock_client.converse(
        modelId=model_id,
        messages=messages,
        system=[{"text": SYSTEM_PROMPT}],
        toolConfig=tool_config()
    )

    output_message = response['output']['message']
    stop_reason = response['stopReason']
    messages.append(output_message)

    if stop_reason == 'tool_use':
        tool_requests = response['output']['message']['content']
        for tool_request in tool_requests:
            if 'toolUse' in tool_request:
                tool = tool_request['toolUse']
                if tool['name'] == 'government_scheme_info':
                    try:
                        result = get_government_scheme_info(tool['input']['query'])
                        tool_result = {
                            "toolUseId": tool['toolUseId'],
                            "content": [{"json": {"result": "Tool Result: " + json.dumps(result)}}]
                        }
                    except Exception as err:
                        tool_result = {
                            "toolUseId": tool['toolUseId'],
                            "content": [{"text": "Tool Result: " + str(err)}],
                            "status": 'error'
                        }

                    tool_result_message = {
                        "role": "user",
                        "content": [{"toolResult": tool_result}]
                    }
                    messages.append(tool_result_message)
                    logger.info("Tool is used")

                    response = bedrock_client.converse(
                        modelId=model_id,
                        messages=messages,
                        toolConfig=tool_config()
                    )
                    return response['output']['message']

    return output_message

# ---------- MAIN UI ----------
def main():
    st.set_page_config(page_title="Converse with Bedrock", layout="wide")

    with st.sidebar:
        st.title("📎 Upload Files")
        documents = st.file_uploader("Upload up to 5 documents (≤ 4.5MB each)",
                                     type=["txt", "pdf", "docx", "csv", "json"],
                                     accept_multiple_files=True)
        images = st.file_uploader("Upload up to 20 images (≤ 3.75MB each)",
                                  type=["png", "jpg", "jpeg", "gif", "bmp", "webp"],
                                  accept_multiple_files=True)

        if st.button("🗑️ Clear Conversation"):
            st.session_state.chat_history = []
            st.rerun()
            
        st.title("🔊 Speech Settings")
        st.session_state.enable_tts = st.checkbox("Enable Text-to-Speech", value=st.session_state.enable_tts)
        
        if st.session_state.enable_tts:
            language_options = {
                "Bengali": "bn-IN",
                "English": "en-IN",
                "Gujarati": "gu-IN",
                "Hindi": "hi-IN",
                "Kannada": "kn-IN",
                "Malayalam": "ml-IN",
                "Marathi": "mr-IN",
                "Odia": "od-IN",
                "Punjabi": "pa-IN",
                "Tamil": "ta-IN",
                "Telugu": "te-IN"
            }
            selected_lang = st.selectbox(
                "Select TTS Language",
                options=list(language_options.keys()),
                index=list(language_options.values()).index(st.session_state.tts_language)
            )
            st.session_state.tts_language = language_options[selected_lang]

    st.title("💬 Vernacular AI Agent")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["text"])

    st.subheader("🎙️ Record Audio")

    # Audio recorder with dynamic key
    if "audio_recorder_key" not in st.session_state:
        st.session_state.audio_recorder_key = str(uuid.uuid4())

    if st.button("🔄 Reset Recorder"):
        st.session_state.audio_recorder_key = str(uuid.uuid4())

    audio = audiorecorder("Click to record", "Click to stop recording", key=st.session_state.audio_recorder_key)

    # Chat input and audio transcription
    text_input = st.chat_input("Type your message or record audio")
    audio_prompt = None

    if len(audio) > 0:
        mp3_io = io.BytesIO()
        audio.export(mp3_io, format="mp3")
        mp3_io.seek(0)

        audio_segment = AudioSegment.from_file(mp3_io, format="mp3")
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_io.seek(0)

        st.audio(wav_io, format="audio/wav")

        try:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                temp_wav.write(wav_io.read())
                temp_wav.flush()  # Ensure data is written to disk
                temp_path = temp_wav.name

            audio_prompt = transcribe_audio(temp_path)
            audio_prompt = decode_unicode_text(audio_prompt)
            
            # Clean up temp file
            os.unlink(temp_path)

        except Exception as e:
            st.error(f"Audio transcription failed: {e}")
            return

    # Check if files were just uploaded (prevent auto-processing)
    docs_changed = documents != st.session_state.last_uploaded_docs
    images_changed = images != st.session_state.last_uploaded_images
    
    if docs_changed:
        st.session_state.last_uploaded_docs = documents
    if images_changed:
        st.session_state.last_uploaded_images = images
    
    # Only process if there's actual user input (not just file upload)
    prompt = text_input.strip() if text_input else (audio_prompt.strip() if audio_prompt else "")
    
    if prompt and not (docs_changed or images_changed):
        try:
            if not prompt.strip():
                st.warning("Please enter a prompt.")
                return

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    bedrock_client = boto3.client("bedrock-runtime")
                    output_message = generate_message(bedrock_client, BEDROCK_MODEL_ID, prompt, documents, images)
                    assistant_response = output_message['content'][0]['text']
                    st.markdown(assistant_response)
                    
                    if st.session_state.enable_tts:
                        try:
                            audio_data = text_to_speech(assistant_response, st.session_state.tts_language)
                            st.audio(audio_data, format="audio/wav")
                        except Exception as e:
                            st.warning(f"TTS failed: {e}")

                    st.session_state.chat_history.append({"role": "user", "text": prompt})
                    st.session_state.chat_history.append({"role": "assistant", "text": assistant_response})

        except ClientError as err:
            st.error(f"A client error occurred: {err.response['Error']['Message']}")
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    main()
