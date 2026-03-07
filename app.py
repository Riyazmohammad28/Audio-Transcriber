from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import numpy as np
import faiss
import pickle
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
import glob
import subprocess
import shutil

app = Flask(__name__)
CORS(app)

# Directory to store downloaded and generated user files (use app dir for predictable path)
DOWNLOADS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "downloads")
os.makedirs(DOWNLOADS_DIR, exist_ok=True)

# Global variables to store models and data
whisper_model = None
embedding_model = None
faiss_index = None
chunks = []
llm = None

def _find_ffmpeg():
    """Return path to ffmpeg executable or None."""
    if shutil.which('ffmpeg'):
        return 'ffmpeg'
    for path in [
        r'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
        r'C:\ffmpeg\bin\ffmpeg.exe',
        os.path.expanduser(r'~\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-7.1.1-full_build\bin\ffmpeg.exe'),
    ]:
        if os.path.isfile(path):
            return path
    return None

def _preprocess_audio_for_whisper(audio_path):
    """Convert audio to 16kHz mono WAV only when needed. Skip for mp3/m4a/webm (Whisper handles these well)."""
    ext = (audio_path.rsplit('.', 1)[-1] or '').lower()
    # Skip preprocessing for formats that faster-whisper handles well (avoids long conversion for YouTube etc.)
    if ext in ('mp3', 'm4a', 'webm'):
        return audio_path
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        return audio_path
    if ext == 'wav':
        try:
            import wave
            with wave.open(audio_path, 'rb') as w:
                if w.getnchannels() == 1 and w.getframerate() == 16000:
                    return audio_path
        except Exception:
            pass
    out_path = os.path.join(tempfile.gettempdir(), f"whisper_{uuid.uuid4().hex}.wav")
    try:
        result = subprocess.run(
            [ffmpeg, '-y', '-i', audio_path, '-ar', '16000', '-ac', '1', '-vn', out_path],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0 and os.path.isfile(out_path):
            print(f"Preprocessed audio to 16kHz mono: {out_path}")
            return out_path
    except Exception as e:
        print(f"Preprocessing failed ({e}), using original file")
    return audio_path

def _get_whisper_device():
    """Use GPU (CUDA) if available for much faster transcription, else CPU."""
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"  # GPU: float16 is faster than int8
    except Exception:
        pass
    return "cpu", "int8"

def initialize_models():
    """Initialize all models on startup"""
    global whisper_model, embedding_model, llm
    
    device, compute_type = _get_whisper_device()
    print(f"Loading Whisper model (device={device}, compute_type={compute_type})...")
    try:
        whisper_model = WhisperModel("base", device=device, compute_type=compute_type)
        print(f"Loaded Faster Whisper base model on {device}")
    except Exception as e:
        print(f"Failed to load base on {device}: {e}")
        try:
            whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
            print("Loaded Faster Whisper tiny model (CPU fallback)")
        except Exception as e2:
            print(f"Failed to load tiny model: {e2}")
            whisper_model = WhisperModel("medium", device="cpu", compute_type="int8")
            print("Loaded Faster Whisper medium model (CPU fallback)")
    
    print("Loading sentence transformer model...")
    embedding_model = SentenceTransformer('all-mpnet-base-v2')
    
    print("Initializing Gemini model...")
    os.environ["GOOGLE_API_KEY"] = ""
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
            'noplaylist': True,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
        }
        
        # Try to find FFmpeg in common locations (needed for MP3 conversion)
        ffmpeg_found = False
        if shutil.which('ffmpeg'):
            ffmpeg_found = True
            print("FFmpeg found in PATH")
        else:
            for path in [
                r'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
                r'C:\ffmpeg\bin\ffmpeg.exe',
                os.path.expanduser(r'~\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-7.1.1-full_build\bin\ffmpeg.exe'),
            ]:
                if os.path.isfile(path):
                    ydl_opts['ffmpeg_location'] = os.path.dirname(path)
                    ffmpeg_found = True
                    print(f"FFmpeg found at: {path}")
                    break
        
        if not ffmpeg_found:
            print("FFmpeg not found, will download audio without conversion")
            ydl_opts['postprocessors'] = []  # Remove postprocessors if FFmpeg not available
        
        # Use browser cookies to bypass YouTube bot check. Try Firefox first - Chrome/Edge lock
        # their cookie DB on Windows when open (yt-dlp#7271), but Firefox works.
        browsers_to_try = [('firefox',), ('chrome',), ('edge',), ('brave',)]
        info = None
        last_error = None
        for browser in browsers_to_try:
            try:
                opts = {**ydl_opts, 'cookiesfrombrowser': browser}
                with yt_dlp.YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(youtube_url, download=True)
                print(f"Used cookies from {browser[0]}")
                break
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                if any(x in err_str for x in ['bot', 'sign in', 'cookies', 'cookie', 'confirm', 'copy chrome', 'permission denied']):
                    print(f"Browser {browser[0]} cookies unavailable, trying next...")
                    continue
                raise
        if info is None and last_error:
            msg = str(last_error)
            err_lower = msg.lower()
            if 'cookie' in err_lower and ('copy' in err_lower or 'chrome' in err_lower):
                msg = "Chrome locks its cookies when open. Use Firefox (log into youtube.com in Firefox) and try again. Or close Chrome completely, then retry."
            elif 'bot' in err_lower or 'sign in' in err_lower:
                msg = "YouTube requires sign-in. Log into youtube.com in Firefox (recommended) or Chrome, then try again."
            raise RuntimeError(msg)
        
        video_title = info.get('title', 'Unknown')
        
        # Get filepath from yt-dlp's result if available
        audio_file = None
        for d in info.get('requested_downloads', []) or []:
            fp = d.get('filepath')
            if fp and os.path.exists(fp):
                audio_file = fp
                break
        # Fallback: search downloads folder by prefix (any audio extension)
        if not audio_file:
            for path in glob.glob(os.path.join(DOWNLOADS_DIR, f"audio_{file_id}.*")):
                ext = (path.rsplit('.', 1)[-1] or '').lower()
                if ext in {'mp3', 'webm', 'm4a', 'opus', 'ogg', 'weba', 'wav', 'aac'}:
                    audio_file = path
                    break
        
        if not audio_file:
            return jsonify({'error': 'Failed to download audio. Ensure FFmpeg is installed (or yt-dlp could not save the file). Try updating yt-dlp: pip install -U yt-dlp'}), 500
        
        return jsonify({
            'success': True,
            'audio_file': audio_file,
            'video_title': video_title,
            'file_id': file_id
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

ALLOWED_AUDIO_EXTENSIONS = {'mp3', 'wav', 'm4a', 'webm', 'ogg', 'flac', 'aac', 'wma', 'mp4'}

@app.route('/api/upload-audio', methods=['POST'])
def upload_audio():
    """Upload a local audio file for transcription"""
    try:
        if 'audio' not in request.files and 'file' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        file = request.files.get('audio') or request.files.get('file')
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        ext = (file.filename.rsplit('.', 1)[-1] or '').lower()
        if ext not in ALLOWED_AUDIO_EXTENSIONS:
            return jsonify({'error': f'Unsupported format. Allowed: {", ".join(sorted(ALLOWED_AUDIO_EXTENSIONS))}'}), 400
        file_id = str(uuid.uuid4())
        safe_name = f"audio_{file_id}.{ext}"
        save_path = os.path.join(DOWNLOADS_DIR, safe_name)
        file.save(save_path)
        return jsonify({
            'success': True,
            'audio_file': save_path,
            'file_name': file.filename,
            'file_id': file_id
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/process-text', methods=['POST'])
def process_text():
    """Process transcribed text file into chunks and create FAISS index"""
    try:
        data = request.get_json()
        text_file = data.get('text_file')
        if not text_file or not os.path.exists(text_file):
            return jsonify({'success': False, 'error': f'Text file not found: {text_file}'}), 400

        # Read text file
        with open(text_file, 'r', encoding="utf-8") as file:
            content = file.read()

        # Split text into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks_list = splitter.split_text(content)
        if not chunks_list:
            return jsonify({'success': False, 'error': 'No chunks created from text.'}), 400

        # Ensure embedding_model is initialized
        global embedding_model
        if embedding_model is None:
            return jsonify({'success': False, 'error': 'Embedding model not initialized.'}), 500

        # Create embeddings
        embeddings = embedding_model.encode(chunks_list, show_progress_bar=False)
        if len(embeddings) == 0:
            return jsonify({'success': False, 'error': 'Embeddings could not be created.'}), 500

        # Create FAISS index
        embedding_matrix = np.array(embeddings).astype("float32")
        dimension = embedding_matrix.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embedding_matrix)

        # Ensure downloads directory exists
        if not os.path.exists(DOWNLOADS_DIR):
            os.makedirs(DOWNLOADS_DIR)

        index_file = os.path.join(DOWNLOADS_DIR, f"faiss_index_{os.path.basename(text_file).replace('.txt', '.idx')}")
        chunks_file = os.path.join(DOWNLOADS_DIR, f"chunks_{os.path.basename(text_file).replace('.txt', '.pkl')}")

        faiss.write_index(index, index_file)
        with open(chunks_file, "wb") as f:
            pickle.dump(chunks_list, f)

        return jsonify({
            'success': True,
            'num_chunks': len(chunks_list),
            'index_file': index_file,
            'chunks_file': chunks_file
        })
    except Exception as e:
        import traceback
        print('Error in /api/process-text:', traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/transcribe', methods=['POST'])
def transcribe_audio():
    """Transcribe audio file using Whisper"""
    try:
        data = request.get_json()
        audio_file = data.get('audio_file')
        task = data.get('task', 'translate')  # 'translate' or 'transcribe'
        language = data.get('language')  # e.g. 'te' for Telugu, 'hi' for Hindi, None for auto-detect
        
        print(f"Transcribing file: {audio_file}, task={task}, language={language}")
        print(f"File exists: {os.path.exists(audio_file)}")
        print(f"Current directory: {os.getcwd()}")
        
        if not audio_file or not os.path.exists(audio_file):
            return jsonify({'error': f'Audio file not found: {audio_file}'}), 400
        
        # Preprocess only when needed (skip for mp3/m4a/webm to avoid long conversion)
        audio_to_transcribe = _preprocess_audio_for_whisper(audio_file)
        if audio_to_transcribe != audio_file:
            print("Preprocessed audio; starting Whisper...")
        else:
            print("Starting Whisper transcription (no preprocessing)...")
        
        try:
            
            # Build transcribe kwargs (gentle VAD to avoid filtering out speech in uploaded files)
            transcribe_kwargs = {
                "task": task,
                "beam_size": 1,
                "vad_filter": True,
                "vad_parameters": dict(threshold=0.35, min_silence_duration_ms=400),
                "condition_on_previous_text": False,
            }
            if language:
                transcribe_kwargs["language"] = language
            if language == "te":
                transcribe_kwargs["initial_prompt"] = "ఇది తెలుగు భాషలో ఉంది."
            
            segments, info = whisper_model.transcribe(audio_to_transcribe, **transcribe_kwargs)
            
            # Combine all segments into one text
            transcribed_text = " ".join([segment.text for segment in segments])
            print(f"Transcription completed. Text length: {len(transcribed_text)}")
            
        except Exception as e:
            print(f"Faster Whisper transcription failed: {e}")
            # Fallback: try with different settings
            try:
                print("Trying with fallback settings...")
                fallback_kwargs = {
                    "task": task, "language": language or None,
                    "beam_size": 1, "vad_filter": True,
                }
                segments, info = whisper_model.transcribe(audio_to_transcribe, **fallback_kwargs)
                transcribed_text = " ".join([segment.text for segment in segments])
                print(f"Transcription completed with different settings. Text length: {len(transcribed_text)}")
            except Exception as e2:
                print(f"All transcription methods failed: {e2}")
                raise e
        finally:
            # Clean up temp preprocessed file if we created one
            if audio_to_transcribe != audio_file and os.path.isfile(audio_to_transcribe):
                try:
                    os.remove(audio_to_transcribe)
                except Exception:
                    pass
        
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