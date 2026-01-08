# Audio Transcription & Q&A System

A powerful web application that downloads YouTube videos, transcribes audio using OpenAI Whisper, and provides intelligent Q&A capabilities using Google Gemini AI.

## Features

- **YouTube Audio Download**: Download audio from any YouTube video
- **Multilingual Transcription**: Transcribe audio in multiple languages (Telugu, Hindi, English, etc.)
- **Translation Support**: Translate non-English audio to English
- **Intelligent Q&A**: Ask questions about the transcribed content
- **Semantic Search**: Find relevant information using vector embeddings
- **Text Summarization**: Generate summaries of transcribed content
- **Modern Web Interface**: Beautiful, responsive UI with real-time feedback

## System Architecture

```
YouTube Video → Audio Download → Whisper Transcription → Text Processing → Vector Embeddings → FAISS Index → Q&A System
```

### Components:
1. **yt-dlp**: Downloads YouTube videos and extracts audio
2. **OpenAI Whisper**: Speech-to-text transcription with translation
3. **LangChain**: Text processing and chunking
4. **Sentence Transformers**: Creates semantic embeddings
5. **FAISS**: Vector similarity search
6. **Google Gemini**: Intelligent question answering and summarization

## Installation

### Prerequisites
- Python 3.8 or higher
- FFmpeg (for audio processing)

### Setup

1. **Clone or download the project files**

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install FFmpeg** (if not already installed):
   - **Windows**: Download from https://ffmpeg.org/download.html
   - **macOS**: `brew install ffmpeg`
   - **Ubuntu/Debian**: `sudo apt install ffmpeg`

4. **Set up Google API Key**:
   - Get a Google API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Update the API key in `app.py` (line 32):
   ```python
   os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY_HERE"
   ```

## Usage

### Starting the Application

1. **Run the Flask application**:
   ```bash
   python app.py
   ```

2. **Open your browser** and go to:
   ```
   http://localhost:5000
   ```

### Using the Web Interface

1. **Download & Transcribe**:
   - Enter a YouTube URL
   - Choose processing type (translate or transcribe)
   - Click "Download & Process"
   - Wait for audio download and transcription

2. **Process for Q&A**:
   - Click "Process for Q&A" to create vector embeddings
   - This enables the question-answering functionality

3. **Ask Questions**:
   - Type your question about the video content
   - Get intelligent answers based on the transcribed text
   - View relevant context chunks used for the answer

4. **Generate Summary**:
   - Click "Generate Summary" for a concise overview
   - Perfect for getting the main points quickly

## API Endpoints

The application provides the following REST API endpoints:

- `POST /api/download-audio`: Download audio from YouTube URL
- `POST /api/transcribe`: Transcribe audio file using Whisper
- `POST /api/process-text`: Process text for Q&A system
- `POST /api/ask-question`: Answer questions using the Q&A system
- `POST /api/summarize`: Generate summary of transcribed text

## Example Workflow

1. **Input**: YouTube URL (e.g., https://youtu.be/FC7TQAvwZ1E)
2. **Download**: Audio extracted and saved as MP3
3. **Transcribe**: Audio converted to text using Whisper
4. **Process**: Text chunked and embedded for semantic search
5. **Q&A**: Ask questions like "What are the main points discussed?"
6. **Summary**: Get a concise overview of the content

## Technical Details

### Models Used
- **Whisper**: OpenAI's speech recognition model (medium size)
- **Sentence Transformers**: `all-mpnet-base-v2` for embeddings
- **FAISS**: Vector similarity search index
- **Google Gemini**: Large language model for Q&A and summarization

### File Structure
```
major_project/
├── app.py                 # Flask backend
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── templates/
│   └── index.html        # Web frontend
└── major_project .ipynb  # Original Jupyter notebook
```

### Performance Notes
- First run will download Whisper and sentence transformer models (~2GB)
- Transcription speed depends on audio length and hardware
- GPU acceleration available for faster processing

## Troubleshooting

### Common Issues

1. **FFmpeg not found**:
   - Install FFmpeg and ensure it's in your system PATH
   - Restart your terminal/command prompt after installation

2. **Google API Key error**:
   - Verify your API key is correct and has proper permissions
   - Check that the key has access to Gemini models

3. **Model download issues**:
   - Ensure stable internet connection for initial model downloads
   - Models are cached locally after first download

4. **Memory issues**:
   - Close other applications to free up RAM
   - Consider using smaller Whisper model if needed

### Error Messages

- **"Audio file not found"**: Check if the YouTube URL is valid and accessible
- **"Transcription failed"**: Ensure audio file is not corrupted
- **"Processing error"**: Check if all dependencies are properly installed

## Customization

### Changing Models
- **Whisper**: Modify `whisper.load_model("medium")` in `app.py`
- **Embeddings**: Change `SentenceTransformer('all-mpnet-base-v2')` to other models
- **LLM**: Update the Gemini model name in the GoogleGenerativeAI initialization

### Adjusting Parameters
- **Chunk size**: Modify `chunk_size=200` in the text splitter
- **Search results**: Change `k=3` in the FAISS search
- **Temperature**: Adjust `temperature=0.2` for different response creativity

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the system.

## Support

For questions or issues, please check the troubleshooting section above or create an issue in the project repository.
