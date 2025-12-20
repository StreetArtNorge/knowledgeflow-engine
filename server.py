"""
KnowledgeFlow - Python Processing Engine
========================================
Deploy this to Render, Railway, or any VPS with Python support.

This server handles:
1. Video downloading via yt-dlp (TikTok, YouTube, Instagram)
2. Audio transcription via OpenAI Whisper
3. Knowledge extraction via GPT-4o
4. Pushing results back to your v0 app via webhooks

ENVIRONMENT VARIABLES REQUIRED:
- OPENAI_API_KEY: Your OpenAI API key
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
from datetime import datetime
from typing import Optional
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yt_dlp
from openai import OpenAI
from supabase import create_client, Client
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "your-webhook-secret-here")
V0_WEBHOOK_URL = os.getenv("V0_WEBHOOK_URL", "")

# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# FastAPI app
app = FastAPI(
    title="KnowledgeFlow Processing Engine",
    description="Video processing, transcription, and knowledge extraction API",
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
    max_videos: int = 10

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
        update_data["attempts"] = supabase.table("processing_queue").select("attempts").eq("id", queue_id).execute().data[0]["attempts"] + 1
    
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
    # Create downloads directory
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
        # TikTok specific options
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            
            # Get the output file path
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
    Transcribe audio using OpenAI Whisper
    Returns: {text, segments with timestamps}
    """
    try:
        with open(file_path, "rb") as audio_file:
            # Use verbose_json for timestamps
            transcript = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )
            
            return {
                'text': transcript.text,
                'segments': [
                    {
                        'start': seg.start,
                        'end': seg.end,
                        'text': seg.text
                    }
                    for seg in (transcript.segments or [])
                ],
                'language': transcript.language,
                'duration': transcript.duration
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

def extract_knowledge_atoms(transcript: str, video_metadata: dict) -> list[dict]:
    """
    Extract knowledge atoms from transcript using GPT-4o
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
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert educational content analyst. Extract precise, learnable knowledge from video transcripts. Always respond with valid JSON."
                },
                {"role": "user", "content": extraction_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=4000
        )
        
        result = json.loads(response.choices[0].message.content)
        return result.get('atoms', []), result.get('video_summary', ''), result.get('main_topics', [])
        
    except Exception as e:
        print(f"Knowledge extraction failed: {e}")
        return [], "", []

def generate_quiz_questions(atom: dict) -> list[dict]:
    """Generate quiz questions for a knowledge atom"""
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
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": quiz_prompt}],
            response_format={"type": "json_object"},
            temperature=0.5,
            max_tokens=1000
        )
        result = json.loads(response.choices[0].message.content)
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
            # If upsert fails, try to get existing
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
            # Estimate timestamp based on hint
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
        
        # Update video status if we have an ID
        if video_db_id:
            supabase.table("videos").update({
                "processing_status": "failed",
                "processing_error": error_msg
            }).eq("id", video_db_id).execute()
        
        # Update queue status
        update_queue_status(queue_id, "failed", error=error_msg)
        
        # Send failure webhook
        await send_webhook("processing_failed", {
            "queue_id": queue_id,
            "video_url": video_url,
            "error": error_msg,
            "user_id": user_id
        })
        
    finally:
        # Cleanup downloaded file
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
            "openai": bool(OPENAI_API_KEY),
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
    # Validate URL
    if not request.video_url:
        raise HTTPException(status_code=400, detail="video_url is required")
    
    # Create queue entry if not provided
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
    
    # Add to background tasks
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
    Sync latest videos from a TikTok/YouTube profile
    """
    # This would use yt-dlp to get latest videos from channel
    # Then queue each one for processing
    
    ydl_opts = {
        'quiet': True,
        'extract_flat': True,
        'playlistend': request.max_videos
    }
    
    try:
        # Build profile URL based on platform
        if request.platform == 'tiktok':
            profile_url = f"https://www.tiktok.com/@{request.username}"
        elif request.platform == 'youtube':
            profile_url = f"https://www.youtube.com/@{request.username}/videos"
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported platform: {request.platform}")
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(profile_url, download=False)
            
            videos = []
            for entry in info.get('entries', [])[:request.max_videos]:
                if entry:
                    video_url = entry.get('url') or entry.get('webpage_url')
                    if video_url:
                        videos.append({
                            'url': video_url,
                            'title': entry.get('title', 'Unknown'),
                            'id': entry.get('id')
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
                queued.append(queue_entry.data[0]['id'])
            
            # Update source last_synced_at
            supabase.table("sources").update({
                "last_synced_at": datetime.utcnow().isoformat(),
                "video_count": len(videos)
            }).eq("id", request.source_id).execute()
            
            return {
                "status": "success",
                "videos_found": len(videos),
                "videos_queued": len(queued),
                "queue_ids": queued
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Profile sync failed: {str(e)}")

@app.post("/generate-quiz/{atom_id}")
async def generate_quiz(atom_id: str):
    """Generate quiz questions for a specific knowledge atom"""
    # Fetch atom from database
    atom_result = supabase.table("knowledge_atoms").select("*").eq("id", atom_id).execute()
    
    if not atom_result.data:
        raise HTTPException(status_code=404, detail="Atom not found")
    
    atom = atom_result.data[0]
    questions = generate_quiz_questions(atom)
    
    return {
        "atom_id": atom_id,
        "questions": questions
    }

@app.post("/webhook/test")
async def test_webhook():
    """Test webhook connectivity"""
    await send_webhook("test", {"message": "Webhook test successful", "timestamp": datetime.utcnow().isoformat()})
    return {"status": "sent", "target": V0_WEBHOOK_URL}

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("KnowledgeFlow Processing Engine")
    print("=" * 60)
    print(f"OpenAI API Key: {'Configured' if OPENAI_API_KEY else 'MISSING'}")
    print(f"Supabase URL: {'Configured' if SUPABASE_URL else 'MISSING'}")
    print(f"Supabase Key: {'Configured' if SUPABASE_SERVICE_KEY else 'MISSING'}")
    print(f"Webhook URL: {V0_WEBHOOK_URL or 'Not configured'}")
    print("=" * 60)
    
    os.makedirs("downloads", exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)
