"""
KnowledgeFlow - Python Processing Engine
========================================
Deploy this to Render, Railway, or any VPS with Python support.

This server handles:
1. Video downloading via yt-dlp (TikTok, YouTube, Instagram)
2. Audio transcription via Google Gemini
3. Knowledge extraction via Gemini 1.5 Pro
4. Pushing results back to your v0 app via webhooks

ENVIRONMENT VARIABLES REQUIRED:
- GOOGLE_API_KEY: Your Google AI API key (get from https://aistudio.google.com/app/apikey)
- SUPABASE_URL: Your Supabase project URL
- SUPABASE_SERVICE_KEY: Your Supabase service role key (not anon key!)
- WEBHOOK_SECRET: A secret key to authenticate webhook calls
- V0_WEBHOOK_URL: Your deployed v0 app webhook URL (e.g., https://your-app.vercel.app/api/webhooks/processing)
"""

import os
import json
import asyncio
import hashlib
import hmac
import base64
from datetime import datetime
from typing import Optional
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yt_dlp
import google.generativeai as genai
from supabase import create_client, Client
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "your-webhook-secret-here")
V0_WEBHOOK_URL = os.getenv("V0_WEBHOOK_URL", "")

# Initialize Google AI
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# FastAPI app
app = FastAPI(
    title="KnowledgeFlow Processing Engine",
    description="Video processing, transcription, and knowledge extraction API (powered by Google AI)",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# MODELS
# ============================================================================

class VideoProcessRequest(BaseModel):
    video_url: str
    user_id: str
    source_id: Optional[str] = None
    queue_id: Optional[str] = None
    priority: int = 5

class ProfileSyncRequest(BaseModel):
    username: str
    platform: str  # "tiktok", "youtube", "instagram"
    user_id: str
    source_id: str
    max_videos: int = 2000

class WebhookPayload(BaseModel):
    event: str
    data: dict
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    version: str
    services: dict

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def generate_webhook_signature(payload: str) -> str:
    """Generate HMAC signature for webhook authentication"""
    return hmac.new(
        WEBHOOK_SECRET.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest()

async def send_webhook(event: str, data: dict):
    """Send webhook to v0 app with results"""
    if not V0_WEBHOOK_URL:
        print("Warning: V0_WEBHOOK_URL not configured, skipping webhook")
        return
    
    payload = {
        "event": event,
        "data": data,
        "timestamp": datetime.utcnow().isoformat()
    }
    payload_str = json.dumps(payload)
    signature = generate_webhook_signature(payload_str)
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                V0_WEBHOOK_URL,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "X-Webhook-Signature": signature
                },
                timeout=30.0
            )
            print(f"Webhook sent: {event} - Status: {response.status_code}")
        except Exception as e:
            print(f"Webhook failed: {e}")

def update_queue_status(queue_id: str, status: str, result: dict = None, error: str = None):
    """Update processing queue status in Supabase"""
    if not queue_id:
        return
    
    update_data = {
        "status": status,
        "updated_at": datetime.utcnow().isoformat()
    }
    
    if status == "processing":
        update_data["started_at"] = datetime.utcnow().isoformat()
    elif status in ["completed", "failed"]:
        update_data["completed_at"] = datetime.utcnow().isoformat()
    
    if result:
        update_data["result"] = result
    if error:
        update_data["error_message"] = error
        try:
            current = supabase.table("processing_queue").select("attempts").eq("id", queue_id).execute()
            if current.data:
                update_data["attempts"] = current.data[0]["attempts"] + 1
        except:
            pass
    
    try:
        supabase.table("processing_queue").update(update_data).eq("id", queue_id).execute()
    except Exception as e:
        print(f"Failed to update queue: {e}")

# ============================================================================
# VIDEO PROCESSING
# ============================================================================

def download_video(video_url: str) -> tuple[str, dict]:
    """
    Download video audio using yt-dlp
    Returns: (file_path, metadata)
    """
    os.makedirs("downloads", exist_ok=True)
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'downloads/%(id)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            
            video_id = info.get('id', 'unknown')
            file_path = f"downloads/{video_id}.mp3"
            
            metadata = {
                'video_id': video_id,
                'title': info.get('title', 'Untitled'),
                'description': info.get('description', ''),
                'duration': info.get('duration', 0),
                'thumbnail': info.get('thumbnail', ''),
                'view_count': info.get('view_count', 0),
                'like_count': info.get('like_count', 0),
                'uploader': info.get('uploader', ''),
                'upload_date': info.get('upload_date', ''),
                'platform': detect_platform(video_url),
                'original_url': video_url
            }
            
            return file_path, metadata
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Download failed: {str(e)}")

def detect_platform(url: str) -> str:
    """Detect platform from URL"""
    url_lower = url.lower()
    if 'tiktok' in url_lower:
        return 'tiktok'
    elif 'youtube' in url_lower or 'youtu.be' in url_lower:
        return 'youtube'
    elif 'instagram' in url_lower:
        return 'instagram'
    return 'unknown'

def transcribe_audio(file_path: str) -> dict:
    """
    Transcribe audio using Google Gemini 1.5 Pro
    Gemini can process audio directly!
    Returns: {text, segments with timestamps}
    """
    try:
        # Upload the audio file to Gemini
        audio_file = genai.upload_file(file_path, mime_type="audio/mp3")
        
        # Use Gemini 1.5 Pro which has native audio understanding
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        prompt = """Listen to this audio and provide a complete transcription.
        
Return your response as JSON with this exact format:
{
    "text": "The complete transcription of the audio",
    "segments": [
        {"start": 0, "end": 10, "text": "First segment text"},
        {"start": 10, "end": 20, "text": "Second segment text"}
    ],
    "language": "en",
    "duration": 60
}

Break the transcription into logical segments of roughly 10-30 seconds each.
Estimate the timestamps based on speech patterns and pauses.
Be accurate with the transcription - capture every word spoken."""

        response = model.generate_content(
            [audio_file, prompt],
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.1
            )
        )
        
        # Parse the JSON response
        result = json.loads(response.text)
        
        # Clean up the uploaded file
        audio_file.delete()
        
        return {
            'text': result.get('text', ''),
            'segments': result.get('segments', []),
            'language': result.get('language', 'en'),
            'duration': result.get('duration', 0)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

def extract_knowledge_atoms(transcript: str, video_metadata: dict) -> list[dict]:
    """
    Extract knowledge atoms from transcript using Gemini 1.5 Pro
    Returns: list of knowledge atoms
    """
    extraction_prompt = f"""Analyze this video transcript and extract distinct "Knowledge Atoms" - discrete, learnable pieces of information.

VIDEO TITLE: {video_metadata.get('title', 'Unknown')}
VIDEO DESCRIPTION: {video_metadata.get('description', '')[:500]}

TRANSCRIPT:
{transcript[:12000]}

For each knowledge atom, provide:
1. type: One of ["concept", "technique", "principle", "fact", "quote", "action_item"]
2. title: A clear, concise title (max 60 chars)
3. content: The full explanation (2-4 sentences)
4. summary: One-sentence summary
5. difficulty_level: 1-5 (1=beginner, 5=expert)
6. tags: Array of relevant tags (max 5)
7. timestamp_hint: Approximate position in video (early/middle/late)

Return JSON format:
{{
    "atoms": [
        {{
            "type": "concept",
            "title": "Example Title",
            "content": "Full explanation here...",
            "summary": "One sentence summary",
            "difficulty_level": 2,
            "tags": ["tag1", "tag2"],
            "timestamp_hint": "early"
        }}
    ],
    "video_summary": "Brief 2-sentence summary of entire video",
    "main_topics": ["topic1", "topic2", "topic3"]
}}

Extract 3-8 knowledge atoms depending on content density. Focus on actionable, memorable insights."""

    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        response = model.generate_content(
            extraction_prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.3,
                max_output_tokens=4000
            )
        )
        
        result = json.loads(response.text)
        return result.get('atoms', []), result.get('video_summary', ''), result.get('main_topics', [])
        
    except Exception as e:
        print(f"Knowledge extraction failed: {e}")
        return [], "", []

def generate_quiz_questions(atom: dict) -> list[dict]:
    """Generate quiz questions for a knowledge atom using Gemini"""
    quiz_prompt = f"""Create 2 multiple-choice quiz questions to test understanding of this knowledge:

TITLE: {atom.get('title')}
CONTENT: {atom.get('content')}

Return JSON format:
{{
    "questions": [
        {{
            "question": "The question text?",
            "options": ["Option A", "Option B", "Option C", "Option D"],
            "correct_index": 0,
            "explanation": "Why this is correct..."
        }}
    ]
}}"""

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')  # Use Flash for speed on simple tasks
        
        response = model.generate_content(
            quiz_prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.5,
                max_output_tokens=1000
            )
        )
        
        result = json.loads(response.text)
        return result.get('questions', [])
    except:
        return []

# ============================================================================
# BACKGROUND PROCESSING
# ============================================================================

async def process_video_task(
    video_url: str,
    user_id: str,
    source_id: Optional[str],
    queue_id: Optional[str]
):
    """Background task to process a video"""
    file_path = None
    video_db_id = None
    
    try:
        # Update status: downloading
        update_queue_status(queue_id, "processing")
        await send_webhook("processing_started", {
            "queue_id": queue_id,
            "video_url": video_url,
            "stage": "downloading"
        })
        
        # 1. Download video
        print(f"Downloading: {video_url}")
        file_path, metadata = download_video(video_url)
        
        await send_webhook("processing_progress", {
            "queue_id": queue_id,
            "stage": "transcribing",
            "progress": 25
        })
        
        # 2. Transcribe audio
        print(f"Transcribing: {file_path}")
        transcript_data = transcribe_audio(file_path)
        
        await send_webhook("processing_progress", {
            "queue_id": queue_id,
            "stage": "extracting",
            "progress": 50
        })
        
        # 3. Save video to database
        video_data = {
            "user_id": user_id,
            "source_id": source_id,
            "platform": metadata['platform'],
            "video_id": metadata['video_id'],
            "video_url": video_url,
            "title": metadata['title'],
            "description": metadata['description'][:1000] if metadata['description'] else None,
            "thumbnail_url": metadata['thumbnail'],
            "duration_seconds": metadata['duration'],
            "view_count": metadata['view_count'],
            "like_count": metadata['like_count'],
            "transcript_raw": transcript_data['text'],
            "transcript_formatted": json.dumps(transcript_data['segments']),
            "language": transcript_data.get('language', 'en'),
            "processing_status": "extracting"
        }
        
        # Insert or update video
        try:
            video_result = supabase.table("videos").upsert(
                video_data,
                on_conflict="user_id,video_id"
            ).execute()
            video_db_id = video_result.data[0]['id']
        except Exception as e:
            existing = supabase.table("videos").select("id").eq("user_id", user_id).eq("video_id", metadata['video_id']).execute()
            if existing.data:
                video_db_id = existing.data[0]['id']
                supabase.table("videos").update(video_data).eq("id", video_db_id).execute()
            else:
                raise e
        
        # 4. Extract knowledge atoms
        print(f"Extracting knowledge from: {metadata['title']}")
        atoms, video_summary, main_topics = extract_knowledge_atoms(
            transcript_data['text'],
            metadata
        )
        
        await send_webhook("processing_progress", {
            "queue_id": queue_id,
            "stage": "saving",
            "progress": 75
        })
        
        # 5. Save atoms to database
        saved_atoms = []
        for i, atom in enumerate(atoms):
            duration = metadata.get('duration', 60)
            hint = atom.get('timestamp_hint', 'middle')
            if hint == 'early':
                ts_start = int(duration * 0.1)
            elif hint == 'late':
                ts_start = int(duration * 0.7)
            else:
                ts_start = int(duration * 0.4)
            
            atom_data = {
                "user_id": user_id,
                "video_id": video_db_id,
                "atom_type": atom.get('type', 'concept'),
                "title": atom.get('title', f'Insight {i+1}'),
                "content": atom.get('content', ''),
                "summary": atom.get('summary'),
                "difficulty_level": atom.get('difficulty_level', 2),
                "tags": atom.get('tags', []),
                "timestamp_start": ts_start,
                "timestamp_end": ts_start + 30,
                "mastery_score": 0,
                "times_reviewed": 0
            }
            
            result = supabase.table("knowledge_atoms").insert(atom_data).execute()
            if result.data:
                saved_atoms.append(result.data[0])
        
        # 6. Update video status to completed
        supabase.table("videos").update({
            "processing_status": "completed",
            "processed_at": datetime.utcnow().isoformat()
        }).eq("id", video_db_id).execute()
        
        # 7. Update queue status
        result_data = {
            "video_id": video_db_id,
            "atoms_created": len(saved_atoms),
            "video_summary": video_summary,
            "main_topics": main_topics
        }
        update_queue_status(queue_id, "completed", result=result_data)
        
        # 8. Send completion webhook
        await send_webhook("processing_completed", {
            "queue_id": queue_id,
            "video_id": video_db_id,
            "video_url": video_url,
            "title": metadata['title'],
            "atoms_created": len(saved_atoms),
            "video_summary": video_summary,
            "user_id": user_id
        })
        
        print(f"Successfully processed: {metadata['title']} - {len(saved_atoms)} atoms created")
        
    except Exception as e:
        error_msg = str(e)
        print(f"Processing failed: {error_msg}")
        
        if video_db_id:
            supabase.table("videos").update({
                "processing_status": "failed",
                "processing_error": error_msg
            }).eq("id", video_db_id).execute()
        
        update_queue_status(queue_id, "failed", error=error_msg)
        
        await send_webhook("processing_failed", {
            "queue_id": queue_id,
            "video_url": video_url,
            "error": error_msg,
            "user_id": user_id
        })
        
    finally:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            print(f"Cleaned up: {file_path}")

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "services": {
            "google_ai": bool(GOOGLE_API_KEY),
            "supabase": bool(SUPABASE_URL and SUPABASE_SERVICE_KEY),
            "webhook": bool(V0_WEBHOOK_URL)
        }
    }

@app.get("/health")
async def health():
    """Simple health endpoint for uptime monitoring"""
    return {"status": "ok"}

@app.post("/process")
async def process_video(request: VideoProcessRequest, background_tasks: BackgroundTasks):
    """
    Queue a video for processing
    This immediately returns and processes in the background
    """
    if not request.video_url:
        raise HTTPException(status_code=400, detail="video_url is required")
    
    queue_id = request.queue_id
    if not queue_id:
        queue_entry = supabase.table("processing_queue").insert({
            "user_id": request.user_id,
            "job_type": "download_video",
            "status": "pending",
            "priority": request.priority,
            "payload": {
                "video_url": request.video_url,
                "source_id": request.source_id
            }
        }).execute()
        queue_id = queue_entry.data[0]['id']
    
    background_tasks.add_task(
        process_video_task,
        request.video_url,
        request.user_id,
        request.source_id,
        queue_id
    )
    
    return {
        "status": "queued",
        "queue_id": queue_id,
        "message": "Video processing started"
    }

@app.post("/sync-profile")
async def sync_profile(request: ProfileSyncRequest, background_tasks: BackgroundTasks):
    """
    Sync videos from a social media profile
    """
    platform_urls = {
        "tiktok": f"https://www.tiktok.com/@{request.username}",
        "youtube": f"https://www.youtube.com/@{request.username}/videos",
        "instagram": f"https://www.instagram.com/{request.username}/"
    }
    
    profile_url = platform_urls.get(request.platform)
    if not profile_url:
        raise HTTPException(status_code=400, detail=f"Unsupported platform: {request.platform}")
    
    # Fetch video list from profile
    ydl_opts = {
        'extract_flat': True,
        'quiet': True,
        'no_warnings': True,
        'playlistend': request.max_videos,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(profile_url, download=False)
            
            videos = []
            entries = info.get('entries', [])[:request.max_videos]
            
            for entry in entries:
                video_url = entry.get('url') or entry.get('webpage_url')
                if video_url:
                    videos.append({
                        "url": video_url,
                        "title": entry.get('title', 'Unknown'),
                        "id": entry.get('id')
                    })
            
            # Queue each video for processing
            queued = []
            for video in videos:
                queue_entry = supabase.table("processing_queue").insert({
                    "user_id": request.user_id,
                    "job_type": "download_video",
                    "status": "pending",
                    "priority": 5,
                    "payload": {
                        "video_url": video['url'],
                        "source_id": request.source_id
                    }
                }).execute()
                
                background_tasks.add_task(
                    process_video_task,
                    video['url'],
                    request.user_id,
                    request.source_id,
                    queue_entry.data[0]['id']
                )
                
                queued.append({
                    "queue_id": queue_entry.data[0]['id'],
                    "video_url": video['url'],
                    "title": video['title']
                })
            
            return {
                "status": "syncing",
                "videos_found": len(videos),
                "videos_queued": len(queued),
                "queued": queued
            }
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Profile sync failed: {str(e)}")

@app.post("/webhook/test")
async def test_webhook():
    """Test the webhook connection"""
    await send_webhook("test", {
        "message": "Webhook connection successful",
        "timestamp": datetime.utcnow().isoformat()
    })
    return {"status": "sent", "url": V0_WEBHOOK_URL}

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("Starting KnowledgeFlow Processing Engine (Google AI)...")
    print(f"Google AI configured: {bool(GOOGLE_API_KEY)}")
    print(f"Supabase configured: {bool(SUPABASE_URL and SUPABASE_SERVICE_KEY)}")
    print(f"Webhook URL: {V0_WEBHOOK_URL or 'Not configured'}")
    
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
