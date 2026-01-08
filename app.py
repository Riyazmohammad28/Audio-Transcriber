from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
from faster_whisper import WhisperModel
import yt_dlp
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except Exception:
    try:
        from langchain_core.text_splitter import RecursiveCharacterTextSplitter
    except Exception:
        # Minimal fallback implementation to avoid requiring langchain
        class RecursiveCharacterTextSplitter:
            def __init__(self, chunk_size=1000, chunk_overlap=0, length_function=len, add_start_index=False):
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap
                self.length_function = length_function
                self.add_start_index = add_start_index

            def split_text(self, text: str):
                if not text:
                    return []
                chunks = []
                start = 0
                text_len = len(text)
                step = max(1, self.chunk_size - self.chunk_overlap)
                while start < text_len:
                    end = start + self.chunk_size
                    chunks.append(text[start:end])
                    start += step
                return chunks
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import tempfile
import uuid

app = Flask(__name__)
CORS(app)

# Directory to store downloaded and generated user files
DOWNLOADS_DIR = os.path.join(os.getcwd(), "downloads")
os.makedirs(DOWNLOADS_DIR, exist_ok=True)

# Global variables to store models and data
whisper_model = None
embedding_model = None
faiss_index = None
chunks = []
llm = None

def initialize_models():
    """Initialize all models on startup"""
    global whisper_model, embedding_model, llm
    
    print("Loading Whisper model...")
    # Use faster-whisper which has better Python 3.13 compatibility
    try:
        whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
        print("Loaded Faster Whisper base model")
    except Exception as e:
        print(f"Failed to load base model: {e}")
        try:
            whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
            print("Loaded Faster Whisper tiny model")
        except Exception as e2:
            print(f"Failed to load tiny model: {e2}")
            # Last resort - try medium model
            whisper_model = WhisperModel("medium", device="cpu", compute_type="int8")
            print("Loaded Faster Whisper medium model")
    
    print("Loading sentence transformer model...")
    embedding_model = SentenceTransformer('all-mpnet-base-v2')
    
    print("Initializing Gemini model...")
    os.environ["GOOGLE_API_KEY"] = "AIzaSyB8hvVgVQP61-fPtkTI3mAq-7RxtQkzlLg"
    llm = GoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    
    print("All models loaded successfully!")

@app.route('/')
def index():
    # Make Download & Transcribe the home page
    return render_template('download.html')


@app.route('/download')
def download_page():
    return render_template('download.html')


@app.route('/process')
def process_page():
    return render_template('process.html')


@app.route('/qa')
def qa_page():
    return render_template('qa.html')


@app.route('/summary')
def summary_page():
    return render_template('summary.html')

@app.route('/api/download-audio', methods=['POST'])
def download_audio():
    """Download audio from YouTube URL"""
    try:
        data = request.get_json()
        youtube_url = data.get('url')
        
        if not youtube_url:
            return jsonify({'error': 'YouTube URL is required'}), 400
        
        # Create unique filename and place in downloads folder
        file_id = str(uuid.uuid4())
        output_filename = os.path.join(DOWNLOADS_DIR, f"audio_{file_id}.%(ext)s")
        
        # Configure yt-dlp
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': output_filename,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
        }
        
        # Try to find FFmpeg in common locations
        ffmpeg_paths = [
            'ffmpeg',  # If in PATH
            r'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
            r'C:\ffmpeg\bin\ffmpeg.exe',
            os.path.expanduser(r'~\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-7.1.1-full_build\bin\ffmpeg.exe')
        ]
        
        ffmpeg_found = False
        for path in ffmpeg_paths:
            try:
                result = os.system(f'"{path}" -version')
                if result == 0:
                    ydl_opts['ffmpeg_location'] = os.path.dirname(path)
                    ffmpeg_found = True
                    print(f"FFmpeg found at: {path}")
                    break
            except:
                continue
        
        if not ffmpeg_found:
            print("FFmpeg not found, will download audio without conversion")
            ydl_opts['postprocessors'] = []  # Remove postprocessors if FFmpeg not available
        
        # Download audio
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            video_title = info.get('title', 'Unknown')
            
            # Find the downloaded file in downloads folder
            audio_file = None
            for file in os.listdir(DOWNLOADS_DIR):
                if file.startswith(f"audio_{file_id}") and (file.endswith('.mp3') or file.endswith('.webm') or file.endswith('.m4a')):
                    audio_file = os.path.join(DOWNLOADS_DIR, file)
                    break
        
        if not audio_file:
            return jsonify({'error': 'Failed to download audio'}), 500
        
        return jsonify({
            'success': True,
            'audio_file': audio_file,
            'video_title': video_title,
            'file_id': file_id
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/transcribe', methods=['POST'])
def transcribe_audio():
    """Transcribe audio file using Whisper"""
    try:
        data = request.get_json()
        audio_file = data.get('audio_file')
        task = data.get('task', 'translate')  # 'translate' or 'transcribe'
        
        print(f"Transcribing file: {audio_file}")
        print(f"File exists: {os.path.exists(audio_file)}")
        print(f"Current directory: {os.getcwd()}")
        
        if not audio_file or not os.path.exists(audio_file):
            return jsonify({'error': f'Audio file not found: {audio_file}'}), 400
        
        # Transcribe using Whisper
        print("Starting Whisper transcription...")
        
        # Faster-whisper doesn't need FFmpeg path setting
        print("Using Faster Whisper - no FFmpeg configuration needed")
        
        try:
            # Use faster-whisper which has better audio handling
            print("Using Faster Whisper for transcription...")
            
            # Transcribe using faster-whisper
            segments, info = whisper_model.transcribe(audio_file, task=task)
            
            # Combine all segments into one text
            transcribed_text = " ".join([segment.text for segment in segments])
            print(f"Transcription completed. Text length: {len(transcribed_text)}")
            
        except Exception as e:
            print(f"Faster Whisper transcription failed: {e}")
            # Fallback: try with different settings
            try:
                print("Trying with different settings...")
                segments, info = whisper_model.transcribe(audio_file, task=task, language=None)
                transcribed_text = " ".join([segment.text for segment in segments])
                print(f"Transcription completed with different settings. Text length: {len(transcribed_text)}")
            except Exception as e2:
                print(f"All transcription methods failed: {e2}")
                raise e
        
        # Save transcribed text into downloads folder
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        output_file = os.path.join(DOWNLOADS_DIR, f"transcribed_{base_name}.txt")

        print(f"Saving to file: {output_file}")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(transcribed_text)
        
        print(f"File saved successfully: {output_file}")
        
        return jsonify({
            'success': True,
            'transcribed_text': transcribed_text,
            'output_file': output_file
        })
        
    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\nTraceback: {traceback.format_exc()}"
        print(error_msg)
        return jsonify({'error': error_msg}), 500

@app.route('/api/process-text', methods=['POST'])
def process_text():
    """Process transcribed text for Q&A system"""
    try:
        data = request.get_json()
        text_file = data.get('text_file')
        
        if not text_file or not os.path.exists(text_file):
            return jsonify({'error': 'Text file not found'}), 400
        
        # Read text file
        with open(text_file, 'r', encoding="utf-8") as file:
            text = file.read()
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=50,
            length_function=len,
            add_start_index=True,
        )
        chunks = text_splitter.split_text(text)
        
        # Create embeddings
        embeddings = embedding_model.encode(chunks, show_progress_bar=False)
        
        # Create FAISS index
        embedding_matrix = np.array(embeddings).astype("float32")
        dimension = embedding_matrix.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embedding_matrix)
        
        # Save index and chunks into downloads folder
        index_file = os.path.join(DOWNLOADS_DIR, f"faiss_index_{os.path.basename(text_file).replace('.txt', '.idx')}")
        chunks_file = os.path.join(DOWNLOADS_DIR, f"chunks_{os.path.basename(text_file).replace('.txt', '.pkl')}")

        faiss.write_index(index, index_file)
        with open(chunks_file, "wb") as f:
            pickle.dump(chunks, f)
        
        return jsonify({
            'success': True,
            'num_chunks': len(chunks),
            'index_file': index_file,
            'chunks_file': chunks_file
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ask-question', methods=['POST'])
def ask_question():
    """Answer questions using the Q&A system"""
    try:
        data = request.get_json()
        question = data.get('question')
        index_file = data.get('index_file')
        chunks_file = data.get('chunks_file')
        
        if not all([question, index_file, chunks_file]):
            return jsonify({'error': 'Question, index file, and chunks file are required'}), 400
        
        if not all(os.path.exists(f) for f in [index_file, chunks_file]):
            return jsonify({'error': 'Index or chunks file not found'}), 400
        
        # Load index and chunks
        index = faiss.read_index(index_file)
        with open(chunks_file, "rb") as f:
            chunks = pickle.load(f)
        
        # Embed the question
        question_embedding = embedding_model.encode([question]).astype("float32")
        
        # Search for relevant chunks
        D, I = index.search(question_embedding, k=3)
        top_chunks = [chunks[i] for i in I[0]]
        
        # Create prompt template
        prompt_template = PromptTemplate.from_template("""
        Use the following context to answer the question.

        Context:
        {context}

        Question: {question}

        Answer:
        """)
        
        # Prepare prompt
        context = "\n".join(top_chunks)
        final_prompt = prompt_template.format(context=context, question=question)
        
        # Get answer from LLM
        response = llm.invoke(final_prompt)
        
        return jsonify({
            'success': True,
            'answer': str(response),
            'relevant_chunks': top_chunks
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/summarize', methods=['POST'])
def summarize_text():
    """Generate summary of transcribed text"""
    try:
        data = request.get_json()
        text_file = data.get('text_file')
        
        if not text_file or not os.path.exists(text_file):
            return jsonify({'error': 'Text file not found'}), 400
        
        # Read text file
        with open(text_file, 'r', encoding="utf-8") as file:
            content = file.read()
        
        # Create summarization prompt
        prompt_template = PromptTemplate.from_template("""
        Summarize the following content in clear and simple language:

        {context}

        Summary:
        """)
        
        final_prompt = prompt_template.format(context=content)
        
        # Get summary from LLM
        response = llm.invoke(final_prompt)
        
        return jsonify({
            'success': True,
            'summary': str(response)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    initialize_models()
    app.run(debug=True, host='0.0.0.0', port=5000) 