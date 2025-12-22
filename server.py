"""
KnowledgeFlow - Python Processing Engine
========================================
Deploy this to Render, Railway, or any VPS with Python support.

This server handles:
1. Video downloading via yt-dlp (TikTok, YouTube, Instagram)
2. Audio transcription via Google Gemini (FREE with API key)
3. Knowledge extraction via Gemini
4. Pushing results back to your v0 app via webhooks

ENVIRONMENT VARIABLES REQUIRED:
- GOOGLE_API_KEY: Your Google AI API key (get from https://aistudio.google.com/app/apikey)
- SUPABASE_URL: Your Supabase project URL
- SUPABASE_SERVICE_KEY: Your Supabase service role key (not anon key!)
- WEBHOOK_SECRET: A secret key to authenticate webhook calls
- V0_WEBHOOK_URL: Your deployed v0 app webhook URL
"""

import os
import json
import asyncio
import hashlib
import hmac
import tempfile
from datetime import datetime
from typing import Optional, List
import uvicorn
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yt_dlp
import google.generativeai as genai
from supabase import create_client, Client
import httpx
from dotenv import load_dotenv

load_dotenv()

# Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "your-webhook-secret-here")
V0_WEBHOOK_URL = os.getenv("V0_WEBHOOK_URL", "")

# Initialize Google AI
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Supabase
supabase: Client = None
if SUPABASE_URL and SUPABASE_SERVICE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

app = FastAPI(
    title="KnowledgeFlow Processing Engine",
    description="Video processing, transcription, and knowledge extraction API",
    version="1.3.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class VideoProcessRequest(BaseModel):
    video_url: str
    user_id: str
    source_id: Optional[str] = None
    queue_id: Optional[str] = None

class ProfileSyncRequest(BaseModel):
    username: str
    platform: str
    user_id: str
    source_id: str
    max_videos: int = 50

class ListVideosRequest(BaseModel):
    username: str
    platform: str
    max_videos: int = 100

class VideoInfo(BaseModel):
    url: str
    title: str
    description: str = ""
    thumbnail: str = ""
    duration: int = 0
    view_count: int = 0
    like_count: int = 0

class ProcessSelectedRequest(BaseModel):
    videos: List[VideoInfo]
    source_id: str
    user_id: str

# Utility Functions
def generate_webhook_signature(payload: str) -> str:
    return hmac.new(WEBHOOK_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()

async def send_webhook(event: str, data: dict):
    if not V0_WEBHOOK_URL:
        return
    payload = {"event": event, "data": data, "timestamp": datetime.utcnow().isoformat()}
    payload_str = json.dumps(payload, separators=(',', ':'), sort_keys=True)
    signature = generate_webhook_signature(payload_str)
    
    async with httpx.AsyncClient() as client:
        try:
            await client.post(V0_WEBHOOK_URL, json=payload, headers={
                "Content-Type": "application/json",
                "X-Webhook-Signature": signature
            }, timeout=30.0)
        except Exception as e:
            print(f"Webhook failed: {e}")

def update_queue_status(queue_id: str, status: str, result: dict = None, error: str = None):
    if not queue_id or not supabase:
        return
    update_data = {"status": status, "updated_at": datetime.utcnow().isoformat()}
    if status == "processing":
        update_data["started_at"] = datetime.utcnow().isoformat()
    elif status in ["completed", "failed"]:
        update_data["completed_at"] = datetime.utcnow().isoformat()
    if result:
        update_data["result"] = result
    if error:
        update_data["error_message"] = error
    try:
        supabase.table("processing_queue").update(update_data).eq("id", queue_id).execute()
    except Exception as e:
        print(f"Queue update failed: {e}")

def detect_platform(url: str) -> str:
    url_lower = url.lower()
    if 'tiktok' in url_lower:
        return 'tiktok'
    elif 'youtube' in url_lower or 'youtu.be' in url_lower:
        return 'youtube'
    elif 'instagram' in url_lower:
        return 'instagram'
    return 'unknown'

def download_video(video_url: str) -> tuple[str, dict]:
    """Download video audio using yt-dlp"""
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
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    }
    
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
            'platform': detect_platform(video_url),
            'original_url': video_url
        }
        return file_path, metadata

def transcribe_audio(file_path: str) -> dict:
    """Transcribe audio using Google Gemini - FREE with API key"""
    try:
        if not os.path.exists(file_path):
            print(f"Audio file not found: {file_path}")
            return {'text': '', 'language': 'en'}
        
        print(f"Uploading audio for transcription: {file_path}")
        audio_file = genai.upload_file(file_path, mime_type="audio/mp3")
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = """Transcribe this audio completely and accurately. 
Return ONLY a JSON object with this exact format:
{
    "text": "The complete transcription of all spoken words",
    "language": "en"
}

Important: Include ALL spoken words in the transcription. Do not summarize."""
        
        response = model.generate_content(
            [audio_file, prompt],
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.1
            )
        )
        
        result = json.loads(response.text)
        
        # Clean up uploaded file
        try:
            audio_file.delete()
        except:
            pass
        
        transcription = result.get('text', '')
        print(f"Transcription complete: {len(transcription)} characters")
        
        return {'text': transcription, 'language': result.get('language', 'en')}
        
    except Exception as e:
        print(f"Transcription error: {e}")
        return {'text': '', 'language': 'en'}

def extract_knowledge_atoms(transcript: str, video_metadata: dict) -> tuple:
    """Extract knowledge atoms from transcript using Gemini"""
    if not transcript or len(transcript) < 50:
        print(f"Transcript too short for extraction: {len(transcript) if transcript else 0} chars")
        return [], "", []
    
    prompt = f"""Analyze this video transcript and extract Knowledge Atoms - key pieces of information that someone could learn from.

VIDEO TITLE: {video_metadata.get('title', 'Unknown')}
VIDEO DESCRIPTION: {video_metadata.get('description', '')[:500]}
TRANSCRIPT: {transcript[:6000]}

Return a JSON object with this exact format:
{{
    "atoms": [
        {{
            "type": "concept",
            "title": "Clear, concise title for this knowledge",
            "content": "2-3 sentence explanation of the concept or insight",
            "summary": "One sentence summary",
            "difficulty_level": 2,
            "tags": ["relevant", "tags"]
        }}
    ],
    "video_summary": "2-3 sentence summary of the entire video",
    "main_topics": ["topic1", "topic2", "topic3"]
}}

Extract 3-8 meaningful knowledge atoms. Focus on:
- Key concepts explained
- Insights and wisdom shared  
- Practical advice given
- Important facts mentioned"""

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.3
            )
        )
        result = json.loads(response.text)
        atoms = result.get('atoms', [])
        print(f"Extracted {len(atoms)} knowledge atoms")
        return atoms, result.get('video_summary', ''), result.get('main_topics', [])
    except Exception as e:
        print(f"Extraction error: {e}")
        return [], "", []

def process_single_video(video_url: str, user_id: str, source_id: str, queue_id: str = None, metadata_override: dict = None) -> dict:
    """Process a single video synchronously - download, transcribe, extract, save"""
    file_path = None
    video_db_id = None
    
    try:
        update_queue_status(queue_id, "processing")
        print(f"\n{'='*50}")
        print(f"Processing: {video_url}")
        
        # 1. Download
        file_path, metadata = download_video(video_url)
        
        # Override metadata if provided
        if metadata_override:
            metadata['title'] = metadata_override.get('title') or metadata['title']
            metadata['description'] = metadata_override.get('description') or metadata['description']
            metadata['thumbnail'] = metadata_override.get('thumbnail') or metadata['thumbnail']
            metadata['duration'] = metadata_override.get('duration') or metadata['duration']
            metadata['view_count'] = metadata_override.get('view_count') or metadata['view_count']
            metadata['like_count'] = metadata_override.get('like_count') or metadata['like_count']
        
        print(f"Downloaded: {metadata['title']}")
        
        # 2. Transcribe using Gemini
        transcript_data = transcribe_audio(file_path)
        transcription_text = transcript_data.get('text', '')
        print(f"Transcribed: {len(transcription_text)} characters")
        
        # 3. Save video to database with transcription
        video_data = {
            "user_id": user_id,
            "source_id": source_id,
            "platform": metadata['platform'],
            "video_id": metadata['video_id'],
            "video_url": video_url,
            "title": metadata['title'],
            "description": (metadata.get('description') or '')[:2000],
            "thumbnail_url": metadata['thumbnail'],
            "duration_seconds": metadata['duration'],
            "view_count": metadata.get('view_count', 0),
            "like_count": metadata.get('like_count', 0),
            # Save transcription to BOTH columns for compatibility
            "transcription": transcription_text,
            "transcript_raw": transcription_text,
            "language": transcript_data.get('language', 'en'),
            "processing_status": "extracting"
        }
        
        # Check if video already exists
        existing = supabase.table("videos").select("id").eq("user_id", user_id).eq("video_id", metadata['video_id']).execute()
        
        if existing.data:
            video_db_id = existing.data[0]['id']
            supabase.table("videos").update(video_data).eq("id", video_db_id).execute()
            print(f"Updated existing video: {video_db_id}")
        else:
            result = supabase.table("videos").insert(video_data).execute()
            video_db_id = result.data[0]['id']
            print(f"Created new video: {video_db_id}")
        
        # 4. Extract knowledge atoms
        atoms, summary, topics = extract_knowledge_atoms(transcription_text, metadata)
        print(f"Extracted {len(atoms)} atoms")
        
        # 5. Save atoms to database
        saved_count = 0
        for atom in atoms:
            atom_data = {
                "user_id": user_id,
                "video_id": video_db_id,
                "atom_type": atom.get('type', 'concept'),
                "title": atom.get('title', 'Insight'),
                "content": atom.get('content', ''),
                "summary": atom.get('summary'),
                "difficulty_level": atom.get('difficulty_level', 2),
                "tags": atom.get('tags', []),
                "mastery_score": 0,
                "times_reviewed": 0
            }
            try:
                supabase.table("knowledge_atoms").insert(atom_data).execute()
                saved_count += 1
            except Exception as e:
                print(f"Atom save error: {e}")
        
        # 6. Update video status to completed
        supabase.table("videos").update({
            "processing_status": "completed",
            "processed_at": datetime.utcnow().isoformat()
        }).eq("id", video_db_id).execute()
        
        update_queue_status(queue_id, "completed", result={"video_id": video_db_id, "atoms_created": saved_count})
        
        print(f"SUCCESS: {metadata['title']} - {saved_count} atoms created")
        print(f"{'='*50}\n")
        
        return {"success": True, "video_id": video_db_id, "atoms": saved_count, "title": metadata['title']}
        
    except Exception as e:
        error_msg = str(e)
        print(f"ERROR processing video: {error_msg}")
        update_queue_status(queue_id, "failed", error=error_msg)
        
        if video_db_id:
            try:
                supabase.table("videos").update({"processing_status": "failed"}).eq("id", video_db_id).execute()
            except:
                pass
        
        return {"success": False, "error": error_msg}
    
    finally:
        # Clean up downloaded file
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass


# API Endpoints

@app.get("/")
def root():
    return {"status": "ok", "service": "KnowledgeFlow Engine", "version": "1.3.0"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "google_ai": "connected" if GOOGLE_API_KEY else "missing",
        "supabase": "connected" if supabase else "missing",
        "webhook_url": V0_WEBHOOK_URL[:50] + "..." if V0_WEBHOOK_URL else "not set"
    }

@app.post("/sync-profile")
async def sync_profile(request: ProfileSyncRequest):
    """Sync a TikTok/YouTube profile - fetch videos and process first 3"""
    try:
        print(f"\nSyncing profile: @{request.username} on {request.platform}")
        
        # Build profile URL
        if request.platform == "tiktok":
            profile_url = f"https://www.tiktok.com/@{request.username}"
        elif request.platform == "youtube":
            profile_url = f"https://www.youtube.com/@{request.username}/videos"
        else:
            raise HTTPException(400, f"Unsupported platform: {request.platform}")
        
        # Fetch video list
        ydl_opts = {
            'extract_flat': True,
            'quiet': True,
            'no_warnings': True,
            'playlist_items': f'1-{request.max_videos}'
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(profile_url, download=False)
            entries = info.get('entries', [])
        
        print(f"Found {len(entries)} videos")
        
        # Process first 3 videos synchronously
        processed = []
        for i, entry in enumerate(entries[:3]):
            video_url = entry.get('url') or entry.get('webpage_url')
            if video_url:
                result = process_single_video(
                    video_url=video_url,
                    user_id=request.user_id,
                    source_id=request.source_id
                )
                processed.append(result)
        
        # Queue remaining videos
        queued = 0
        for entry in entries[3:]:
            video_url = entry.get('url') or entry.get('webpage_url')
            if video_url and supabase:
                try:
                    supabase.table("processing_queue").insert({
                        "user_id": request.user_id,
                        "source_id": request.source_id,
                        "job_type": "process_video",
                        "status": "pending",
                        "priority": 5,
                        "payload": {
                            "video_url": video_url,
                            "title": entry.get('title', 'Unknown')
                        }
                    }).execute()
                    queued += 1
                except Exception as e:
                    print(f"Queue error: {e}")
        
        # Update source
        if supabase:
            supabase.table("sources").update({
                "video_count": len(entries),
                "last_synced_at": datetime.utcnow().isoformat()
            }).eq("id", request.source_id).execute()
        
        return {
            "success": True,
            "total_videos": len(entries),
            "processed": len([p for p in processed if p.get('success')]),
            "queued": queued
        }
        
    except Exception as e:
        print(f"Sync error: {e}")
        raise HTTPException(500, str(e))

@app.post("/process-pending")
async def process_pending(limit: int = 5):
    """Process pending videos from the queue"""
    if not supabase:
        raise HTTPException(500, "Database not connected")
    
    # Get pending items
    result = supabase.table("processing_queue")\
        .select("*")\
        .eq("status", "pending")\
        .order("priority", desc=True)\
        .order("created_at")\
        .limit(limit)\
        .execute()
    
    pending = result.data or []
    print(f"\nProcessing {len(pending)} pending videos")
    
    processed = []
    for item in pending:
        payload = item.get('payload', {})
        video_url = payload.get('video_url')
        
        if video_url:
            result = process_single_video(
                video_url=video_url,
                user_id=item['user_id'],
                source_id=item.get('source_id'),
                queue_id=item['id']
            )
            processed.append(result)
    
    return {
        "success": True,
        "processed": len(processed),
        "results": processed
    }

@app.post("/list-videos")
async def list_videos(request: ListVideosRequest):
    """List available videos from a profile without downloading"""
    try:
        if request.platform == "tiktok":
            url = f"https://www.tiktok.com/@{request.username}"
        elif request.platform == "youtube":
            url = f"https://www.youtube.com/@{request.username}/videos"
        else:
            raise HTTPException(400, f"Unsupported platform: {request.platform}")
        
        ydl_opts = {
            'extract_flat': 'in_playlist',
            'quiet': True,
            'no_warnings': True,
            'playlist_items': f'1-{request.max_videos}'
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            entries = info.get('entries', [])
        
        videos = []
        for entry in entries:
            videos.append({
                "url": entry.get('url') or entry.get('webpage_url', ''),
                "title": entry.get('title', 'Untitled'),
                "description": entry.get('description', ''),
                "thumbnail": entry.get('thumbnail', ''),
                "duration": entry.get('duration', 0),
                "view_count": entry.get('view_count', 0),
                "like_count": entry.get('like_count', 0)
            })
        
        return {"success": True, "videos": videos, "total": len(videos)}
        
    except Exception as e:
        print(f"List videos error: {e}")
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
