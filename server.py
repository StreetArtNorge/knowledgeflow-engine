"""
KnowledgeFlow - Python Processing Engine
========================================
Deploy this to Render, Railway, or any VPS with Python support.

This server handles:
1. Video downloading via yt-dlp (TikTok, YouTube, Instagram)
2. Audio transcription via Google Gemini
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
    version="1.2.0"
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
    update_data = {"status": status}
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
    """Transcribe audio using Google Gemini"""
    try:
        audio_file = genai.upload_file(file_path, mime_type="audio/mp3")
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = """Transcribe this audio completely. Return JSON:
{
    "text": "Full transcription",
    "language": "en",
    "duration": 60
}"""
        
        response = model.generate_content(
            [audio_file, prompt],
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.1
            )
        )
        
        result = json.loads(response.text)
        audio_file.delete()
        return {'text': result.get('text', ''), 'language': result.get('language', 'en')}
    except Exception as e:
        print(f"Transcription error: {e}")
        return {'text': '', 'language': 'en'}

def extract_knowledge_atoms(transcript: str, video_metadata: dict) -> tuple:
    """Extract knowledge atoms from transcript"""
    if not transcript or len(transcript) < 50:
        return [], "", []
    
    prompt = f"""Analyze this transcript and extract Knowledge Atoms.

TITLE: {video_metadata.get('title', 'Unknown')}
TRANSCRIPT: {transcript[:6000]}

Return JSON:
{{
    "atoms": [
        {{
            "type": "concept",
            "title": "Clear title",
            "content": "2-3 sentence explanation",
            "summary": "One sentence",
            "difficulty_level": 2,
            "tags": ["tag1", "tag2"]
        }}
    ],
    "video_summary": "2 sentence summary",
    "main_topics": ["topic1", "topic2"]
}}

Extract 3-6 knowledge atoms."""

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
        return result.get('atoms', []), result.get('video_summary', ''), result.get('main_topics', [])
    except Exception as e:
        print(f"Extraction error: {e}")
        return [], "", []

def process_single_video(video_url: str, user_id: str, source_id: str, queue_id: str = None, metadata_override: dict = None) -> dict:
    """Process a single video synchronously - download, transcribe, extract, save"""
    file_path = None
    video_db_id = None
    
    try:
        update_queue_status(queue_id, "processing")
        print(f"Processing: {video_url}")
        
        # 1. Download
        file_path, metadata = download_video(video_url)
        
        # Override metadata if provided (from list-videos)
        if metadata_override:
            metadata['title'] = metadata_override.get('title') or metadata['title']
            metadata['description'] = metadata_override.get('description') or metadata['description']
            metadata['thumbnail'] = metadata_override.get('thumbnail') or metadata['thumbnail']
            metadata['duration'] = metadata_override.get('duration') or metadata['duration']
            metadata['view_count'] = metadata_override.get('view_count') or metadata['view_count']
            metadata['like_count'] = metadata_override.get('like_count') or metadata['like_count']
        
        print(f"Downloaded: {metadata['title']}")
        
        # 2. Transcribe
        transcript_data = transcribe_audio(file_path)
        print(f"Transcribed: {len(transcript_data['text'])} chars")
        
        # 3. Save video to database
        video_data = {
            "user_id": user_id,
            "source_id": source_id,
            "platform": metadata['platform'],
            "video_id": metadata['video_id'],
            "video_url": video_url,
            "title": metadata['title'],
            "description": (metadata.get('description') or '')[:2000],  # Allow longer descriptions
            "thumbnail_url": metadata['thumbnail'],
            "duration_seconds": metadata['duration'],
            "view_count": metadata.get('view_count', 0),
            "like_count": metadata.get('like_count', 0),
            "transcript_raw": transcript_data['text'],
            "language": transcript_data.get('language', 'en'),
            "processing_status": "extracting"
        }
        
        # Check if video already exists
        existing = supabase.table("videos").select("id").eq("user_id", user_id).eq("video_id", metadata['video_id']).execute()
        
        if existing.data:
            video_db_id = existing.data[0]['id']
            supabase.table("videos").update(video_data).eq("id", video_db_id).execute()
        else:
            result = supabase.table("videos").insert(video_data).execute()
            video_db_id = result.data[0]['id']
        
        print(f"Saved video: {video_db_id}")
        
        # 4. Extract knowledge atoms
        atoms, summary, topics = extract_knowledge_atoms(transcript_data['text'], metadata)
        print(f"Extracted {len(atoms)} atoms")
        
        # 5. Save atoms
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
        
        # 6. Update video status
        supabase.table("videos").update({
            "processing_status": "completed",
            "processed_at": datetime.utcnow().isoformat()
        }).eq("id", video_db_id).execute()
        
        update_queue_status(queue_id, "completed", result={"video_id": video_db_id, "atoms": saved_count})
        
        print(f"SUCCESS: {metadata['title']} - {saved_count} atoms")
        return {"success": True, "video_id": video_db_id, "atoms": saved_count, "title": metadata['title']}
        
    except Exception as e:
        error_msg = str(e)
        print(f"FAILED: {error_msg}")
        
        if video_db_id:
            try:
                supabase.table("videos").update({
                    "processing_status": "failed",
                    "processing_error": error_msg
                }).eq("id", video_db_id).execute()
            except:
                pass
        
        update_queue_status(queue_id, "failed", error=error_msg)
        return {"success": False, "error": error_msg}
        
    finally:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass

# API Endpoints
@app.get("/")
async def root():
    return {
        "status": "healthy",
        "version": "1.2.0",
        "services": {
            "google_ai": bool(GOOGLE_API_KEY),
            "supabase": bool(supabase),
            "webhook": bool(V0_WEBHOOK_URL)
        }
    }

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

@app.post("/list-videos")
async def list_videos(request: ListVideosRequest):
    """List all videos from a profile without downloading - returns metadata only"""
    print(f"Listing videos for @{request.username} on {request.platform}")
    
    # Build profile URL
    if request.platform == "tiktok":
        profile_url = f"https://www.tiktok.com/@{request.username}"
    elif request.platform == "youtube":
        profile_url = f"https://www.youtube.com/@{request.username}/videos"
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported platform: {request.platform}")
    
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': 'in_playlist',
        'playlistend': request.max_videos,
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(profile_url, download=False)
            
            videos = []
            if 'entries' in info:
                for entry in info['entries']:
                    if entry:
                        video_id = entry.get('id', '')
                        
                        # Build video URL
                        if request.platform == "tiktok":
                            video_url = f"https://www.tiktok.com/@{request.username}/video/{video_id}"
                        else:
                            video_url = entry.get('url') or f"https://www.youtube.com/watch?v={video_id}"
                        
                        videos.append({
                            'id': video_id,
                            'url': video_url,
                            'title': entry.get('title', 'Untitled'),
                            'description': entry.get('description', '') or '',
                            'thumbnail': entry.get('thumbnail', '') or '',
                            'duration': entry.get('duration', 0) or 0,
                            'view_count': entry.get('view_count', 0) or 0,
                            'like_count': entry.get('like_count', 0) or 0,
                        })
            
            print(f"Found {len(videos)} videos")
            return {"videos": videos, "total": len(videos)}
            
    except Exception as e:
        print(f"List videos failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-selected")
async def process_selected(request: ProcessSelectedRequest):
    """Process a list of selected videos"""
    print(f"Processing {len(request.videos)} selected videos")
    
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    
    # Process first 3 immediately, queue the rest
    results = []
    queued = 0
    
    for i, video in enumerate(request.videos):
        # Create queue entry
        entry = supabase.table("processing_queue").insert({
            "user_id": request.user_id,
            "source_id": request.source_id,
            "job_type": "download_video",
            "status": "pending",
            "payload": {
                "video_url": video.url,
                "title": video.title,
                "description": video.description,
                "thumbnail": video.thumbnail,
                "duration": video.duration,
                "view_count": video.view_count,
                "like_count": video.like_count,
            }
        }).execute()
        queue_id = entry.data[0]['id']
        
        if i < 3:
            # Process immediately
            metadata_override = {
                'title': video.title,
                'description': video.description,
                'thumbnail': video.thumbnail,
                'duration': video.duration,
                'view_count': video.view_count,
                'like_count': video.like_count,
            }
            result = process_single_video(video.url, request.user_id, request.source_id, queue_id, metadata_override)
            results.append(result)
        else:
            queued += 1
    
    # Update source last_synced
    supabase.table("sources").update({
        "last_synced_at": datetime.utcnow().isoformat()
    }).eq("id", request.source_id).execute()
    
    successful = len([r for r in results if r.get("success")])
    
    return {
        "status": "completed",
        "processed": successful,
        "queued": queued,
        "results": results
    }

@app.post("/process")
async def process_video(request: VideoProcessRequest):
    """Process a single video synchronously"""
    if not request.video_url:
        raise HTTPException(status_code=400, detail="video_url is required")
    
    # Create queue entry
    queue_id = request.queue_id
    if not queue_id and supabase:
        entry = supabase.table("processing_queue").insert({
            "user_id": request.user_id,
            "job_type": "process_video",
            "status": "pending",
            "payload": {"video_url": request.video_url}
        }).execute()
        queue_id = entry.data[0]['id']
    
    # Process synchronously
    result = process_single_video(request.video_url, request.user_id, request.source_id, queue_id)
    
    if result["success"]:
        await send_webhook("processing_completed", {
            "video_id": result["video_id"],
            "atoms_created": result["atoms"],
            "user_id": request.user_id
        })
    
    return result

@app.post("/sync-profile")
async def sync_profile(request: ProfileSyncRequest):
    """Sync videos from a profile - processes first 3 immediately, queues rest"""
    print(f"Syncing: @{request.username} on {request.platform}")
    
    # Build profile URL
    if request.platform == "tiktok":
        profile_url = f"https://www.tiktok.com/@{request.username}"
    elif request.platform == "youtube":
        profile_url = f"https://www.youtube.com/@{request.username}/videos"
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported platform: {request.platform}")
    
    # Get video list
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': True,
        'playlistend': min(request.max_videos, 50),
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(profile_url, download=False)
            
            videos = []
            if 'entries' in info:
                for entry in info['entries']:
                    if entry:
                        video_url = entry.get('url') or entry.get('webpage_url')
                        video_id = entry.get('id')
                        
                        if not video_url and video_id:
                            if request.platform == "tiktok":
                                video_url = f"https://www.tiktok.com/@{request.username}/video/{video_id}"
                            else:
                                video_url = f"https://www.youtube.com/watch?v={video_id}"
                        
                        if video_url:
                            videos.append({
                                'url': video_url,
                                'id': video_id,
                                'title': entry.get('title', 'Unknown'),
                                'description': entry.get('description', ''),
                            })
            
            print(f"Found {len(videos)} videos")
            
            # Process first 3 videos immediately
            processed = []
            for video in videos[:3]:
                # Create queue entry
                entry = supabase.table("processing_queue").insert({
                    "user_id": request.user_id,
                    "source_id": request.source_id,
                    "job_type": "download_video",
                    "status": "pending",
                    "payload": {"video_url": video['url'], "title": video['title'], "description": video.get('description', '')}
                }).execute()
                queue_id = entry.data[0]['id']
                
                # Process immediately
                result = process_single_video(video['url'], request.user_id, request.source_id, queue_id)
                processed.append(result)
            
            # Queue remaining videos (just create entries, don't process)
            queued = 0
            for video in videos[3:]:
                try:
                    supabase.table("processing_queue").insert({
                        "user_id": request.user_id,
                        "source_id": request.source_id,
                        "job_type": "download_video",
                        "status": "pending",
                        "payload": {"video_url": video['url'], "title": video['title'], "description": video.get('description', '')}
                    }).execute()
                    queued += 1
                except:
                    pass
            
            # Update source last_synced
            supabase.table("sources").update({
                "last_synced_at": datetime.utcnow().isoformat()
            }).eq("id", request.source_id).execute()
            
            successful = [p for p in processed if p.get("success")]
            
            return {
                "status": "completed",
                "found": len(videos),
                "processed_immediately": len(successful),
                "queued_for_later": queued,
                "results": processed
            }
            
    except Exception as e:
        print(f"Sync failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-pending")
async def process_pending():
    """Process pending queue items - call this periodically"""
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    
    # Get pending items
    pending = supabase.table("processing_queue").select("*").eq("status", "pending").limit(5).execute()
    
    results = []
    for item in pending.data:
        payload = item.get("payload", {})
        video_url = payload.get("video_url")
        
        if video_url:
            metadata_override = {
                'title': payload.get('title'),
                'description': payload.get('description'),
                'thumbnail': payload.get('thumbnail'),
                'duration': payload.get('duration'),
                'view_count': payload.get('view_count'),
                'like_count': payload.get('like_count'),
            }
            result = process_single_video(
                video_url,
                item["user_id"],
                item.get("source_id"),
                item["id"],
                metadata_override
            )
            results.append(result)
    
    return {"processed": len(results), "results": results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
