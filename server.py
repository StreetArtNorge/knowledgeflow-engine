"""
KnowledgeFlow - Python Processing Engine v2.1
=============================================
Deploy this to Render with FFmpeg support.

RENDER SETUP:
1. Create a new Web Service from your GitHub repo
2. Set Build Command: pip install -r requirements.txt
3. Set Start Command: uvicorn server:app --host 0.0.0.0 --port $PORT
4. Add these Environment Variables:
   - GOOGLE_API_KEY: Your Google AI API key
   - SUPABASE_URL: Your Supabase project URL
   - SUPABASE_SERVICE_KEY: Your Supabase service role key
   - WEBHOOK_SECRET: A secret key (any random string)
   - V0_WEBHOOK_URL: https://your-v0-app.vercel.app/api/webhooks/processing

IMPORTANT: Add FFmpeg support in Render:
- Go to Settings > Build & Deploy
- Add to Build Command: apt-get update && apt-get install -y ffmpeg && pip install -r requirements.txt
"""

import os
import json
import hashlib
import hmac
from datetime import datetime
from typing import Optional, List
import uvicorn
from fastapi import FastAPI, HTTPException, Query
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

# Initialize Google GenAI
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Supabase
supabase: Client = None
if SUPABASE_URL and SUPABASE_SERVICE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

app = FastAPI(
    title="KnowledgeFlow Processing Engine",
    description="Video processing, transcription, and knowledge extraction API",
    version="2.1.0"
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
            print(f"[WEBHOOK] Failed: {e}")


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
        print(f"[QUEUE] Update failed: {e}")


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
    
    print(f"[DOWNLOAD] Starting: {video_url}")
    
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
        
        print(f"[DOWNLOAD] Success: {metadata['title']}")
        return file_path, metadata


def get_user_ai_settings(user_id: str) -> dict:
    """Fetch user's AI settings from the database"""
    default_settings = {
        "transcription_prompt": "Transcribe this video accurately. Include speaker emotions and tone indicators in [brackets]. Preserve the natural flow and pauses.",
        "extraction_prompt": """Extract knowledge atoms from this transcription. Focus on: key insights, practical techniques, memorable quotes, and actionable advice.

For each piece of knowledge, create a structured entry with:
- A clear, concise title
- The main content/insight  
- A brief summary suitable for a flashcard
- Difficulty level (1-5)
- Relevant tags

Format as JSON array.""",
        "model": "gemini-1.5-flash",
        "temperature": 0.3,
        "max_tokens": 8192,
        "extraction_categories": ["insights", "techniques", "quotes", "advice"]
    }
    
    try:
        if not supabase:
            return default_settings
            
        result = supabase.table("ai_settings").select("*").eq("user_id", user_id).execute()
        
        if result.data and len(result.data) > 0:
            settings = result.data[0]
            return {
                "transcription_prompt": settings.get("transcription_prompt") or default_settings["transcription_prompt"],
                "extraction_prompt": settings.get("extraction_prompt") or default_settings["extraction_prompt"],
                "model": settings.get("model") or default_settings["model"],
                "temperature": float(settings.get("temperature") or default_settings["temperature"]),
                "max_tokens": int(settings.get("max_tokens") or default_settings["max_tokens"]),
                "extraction_categories": settings.get("extraction_categories") or default_settings["extraction_categories"]
            }
    except Exception as e:
        print(f"[SETTINGS] Error fetching settings: {e}")
    
    return default_settings


def transcribe_audio_with_settings(file_path: str, settings: dict) -> dict:
    """Transcribe audio using Google Gemini with user's custom prompt"""
    try:
        if not os.path.exists(file_path):
            print(f"[TRANSCRIBE] File not found: {file_path}")
            return {'text': '', 'language': 'en'}
        
        if not GOOGLE_API_KEY:
            print("[TRANSCRIBE] No Google API key configured")
            return {'text': '', 'language': 'en'}
        
        file_size = os.path.getsize(file_path)
        print(f"[TRANSCRIBE] Uploading file: {file_path} ({file_size} bytes)")
        
        # Upload the audio file
        uploaded_file = genai.upload_file(file_path)
        print(f"[TRANSCRIBE] Uploaded: {uploaded_file.name}")
        
        # Wait for file to be processed
        import time
        while uploaded_file.state.name == "PROCESSING":
            print("[TRANSCRIBE] Waiting for file processing...")
            time.sleep(2)
            uploaded_file = genai.get_file(uploaded_file.name)
        
        if uploaded_file.state.name == "FAILED":
            print("[TRANSCRIBE] File processing failed")
            return {'text': '', 'language': 'en'}
        
        print("[TRANSCRIBE] Generating transcript...")
        
        # Use user's custom prompt and model
        model_name = settings.get("model", "gemini-1.5-flash")
        custom_prompt = settings.get("transcription_prompt", "Transcribe this video accurately.")
        
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            [uploaded_file, f"{custom_prompt}\n\nReturn ONLY the transcript text, nothing else. If no speech is detected, return 'No speech detected'."],
            generation_config=genai.GenerationConfig(temperature=0.1)
        )
        
        transcription = response.text.strip() if response.text else ""
        
        if "no speech" in transcription.lower() or len(transcription) < 10:
            print("[TRANSCRIBE] No speech detected or very short response")
            transcription = ""
        
        print(f"[TRANSCRIBE] Success: {len(transcription)} characters")
        
        # Clean up uploaded file
        try:
            genai.delete_file(uploaded_file.name)
        except:
            pass
        
        return {'text': transcription, 'language': 'en'}
        
    except Exception as e:
        print(f"[TRANSCRIBE] Error: {e}")
        import traceback
        traceback.print_exc()
        return {'text': '', 'language': 'en'}


def extract_knowledge_atoms_with_settings(transcription: str, metadata: dict, settings: dict) -> tuple:
    """Extract knowledge atoms using user's custom prompt"""
    try:
        if not GOOGLE_API_KEY or not transcription:
            return [], "", []
        
        model_name = settings.get("model", "gemini-1.5-flash")
        custom_prompt = settings.get("extraction_prompt", "Extract knowledge atoms from this transcription.")
        temperature = settings.get("temperature", 0.3)
        categories = settings.get("extraction_categories", ["insights", "techniques", "quotes", "advice"])
        
        model = genai.GenerativeModel(model_name)
        
        full_prompt = f"""{custom_prompt}

Categories to extract: {', '.join(categories)}

Video Title: {metadata.get('title', 'Unknown')}
Video Description: {metadata.get('description', '')[:500]}

Transcription:
{transcription}

Return a JSON array of knowledge atoms. Each atom should have:
- type: one of {categories}
- title: brief title (max 100 chars)
- content: the full insight/knowledge (max 500 chars)
- summary: one sentence summary for flashcard
- difficulty_level: 1-5 (1=beginner, 5=expert)
- tags: array of relevant tags

Return ONLY valid JSON array, no other text."""

        response = model.generate_content(
            full_prompt,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                response_mime_type="application/json"
            )
        )
        
        # Parse JSON response
        atoms = []
        try:
            result_text = response.text.strip()
            if result_text.startswith('['):
                atoms = json.loads(result_text)
            else:
                # Try to extract JSON array
                import re
                json_match = re.search(r'\[[\s\S]*\]', result_text)
                if json_match:
                    atoms = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            print(f"[EXTRACT] JSON parse error: {e}")
        
        # Generate summary
        summary = f"Extracted {len(atoms)} knowledge atoms from {metadata.get('title', 'video')}"
        
        # Extract topics from atoms
        topics = list(set([atom.get('type', 'insight') for atom in atoms]))
        
        print(f"[EXTRACT] Found {len(atoms)} atoms, topics: {topics}")
        return atoms, summary, topics
        
    except Exception as e:
        print(f"[EXTRACT] Error: {e}")
        import traceback
        traceback.print_exc()
        return [], "", []


def process_single_video(video_url: str, user_id: str, source_id: str, queue_id: str = None, metadata_override: dict = None) -> dict:
    """Process a single video synchronously - download, transcribe, extract, save"""
    file_path = None
    video_db_id = None
    
    try:
        update_queue_status(queue_id, "processing")
        print(f"\n{'='*60}")
        print(f"[PROCESS] Starting: {video_url}")
        
        # 1. Download
        print("[PROCESS] Step 1: Downloading...")
        file_path, metadata = download_video(video_url)
        
        # Override metadata if provided
        if metadata_override:
            metadata['title'] = metadata_override.get('title') or metadata['title']
            metadata['description'] = metadata_override.get('description') or metadata['description']
            metadata['thumbnail'] = metadata_override.get('thumbnail') or metadata['thumbnail']
            metadata['duration'] = metadata_override.get('duration') or metadata['duration']
            metadata['view_count'] = metadata_override.get('view_count') or metadata['view_count']
            metadata['like_count'] = metadata_override.get('like_count') or metadata['like_count']
        
        print(f"[PROCESS] Downloaded: {metadata['title']}")
        
        # Fetch user's AI settings
        settings = get_user_ai_settings(user_id)
        
        # 2. Transcribe using Gemini
        print("[PROCESS] Step 2: Transcribing with Gemini...")
        transcript_data = transcribe_audio_with_settings(file_path, settings)
        transcription_text = transcript_data.get('text', '')
        print(f"[PROCESS] Transcribed: {len(transcription_text)} characters")
        
        # 3. Save video to database with transcription
        print("[PROCESS] Step 3: Saving to database...")
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
            "transcription": transcription_text,
            "language": transcript_data.get('language', 'en'),
            "processing_status": "extracting" if transcription_text else "completed"
        }
        
        # Check if video already exists
        existing = supabase.table("videos").select("id").eq("user_id", user_id).eq("video_id", metadata['video_id']).execute()
        
        if existing.data:
            video_db_id = existing.data[0]['id']
            supabase.table("videos").update(video_data).eq("id", video_db_id).execute()
            print(f"[PROCESS] Updated existing video: {video_db_id}")
        else:
            result = supabase.table("videos").insert(video_data).execute()
            video_db_id = result.data[0]['id']
            print(f"[PROCESS] Created new video: {video_db_id}")
        
        # 4. Extract knowledge atoms (only if we have transcription)
        atoms_saved = 0
        if transcription_text and len(transcription_text) >= 50:
            print("[PROCESS] Step 4: Extracting knowledge atoms...")
            atoms, summary, topics = extract_knowledge_atoms_with_settings(transcription_text, metadata, settings)
            print(f"[PROCESS] Extracted {len(atoms)} atoms")
            
            # 5. Save atoms to database
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
                    atoms_saved += 1
                except Exception as e:
                    print(f"[PROCESS] Atom save error: {e}")
        else:
            print("[PROCESS] Step 4: Skipping atom extraction (no transcription)")
        
        # 6. Update video status to completed
        supabase.table("videos").update({
            "processing_status": "completed",
            "processed_at": datetime.utcnow().isoformat()
        }).eq("id", video_db_id).execute()
        
        update_queue_status(queue_id, "completed", result={"video_id": video_db_id, "atoms_created": atoms_saved})
        
        print(f"[PROCESS] SUCCESS: {metadata['title']} - {atoms_saved} atoms created")
        print(f"{'='*60}\n")
        
        return {"success": True, "video_id": video_db_id, "atoms": atoms_saved, "title": metadata['title']}
        
    except Exception as e:
        error_msg = str(e)
        print(f"[PROCESS] ERROR: {error_msg}")
        import traceback
        traceback.print_exc()
        
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
    return {"status": "ok", "service": "KnowledgeFlow Engine", "version": "2.1.0"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "google_ai": "connected" if GOOGLE_API_KEY else "missing GOOGLE_API_KEY",
        "supabase": "connected" if supabase else "missing credentials",
        "webhook_url": V0_WEBHOOK_URL[:50] + "..." if V0_WEBHOOK_URL else "not set"
    }


@app.post("/sync-profile")
async def sync_profile(request: ProfileSyncRequest):
    """Sync a TikTok/YouTube profile - fetch videos and process first 3"""
    try:
        print(f"\n[SYNC] Starting sync: @{request.username} on {request.platform}")
        
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
        
        print(f"[SYNC] Found {len(entries)} videos")
        
        # Process first 3 videos synchronously
        processed = []
        for i, entry in enumerate(entries[:3]):
            video_url = entry.get('url') or entry.get('webpage_url')
            if video_url:
                print(f"[SYNC] Processing video {i+1}/3: {entry.get('title', 'Unknown')}")
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
                    print(f"[SYNC] Queue error: {e}")
        
        # Update source
        if supabase:
            supabase.table("sources").update({
                "video_count": len(entries),
                "last_synced_at": datetime.utcnow().isoformat()
            }).eq("id", request.source_id).execute()
        
        success_count = len([p for p in processed if p.get('success')])
        print(f"[SYNC] Complete: {success_count} processed, {queued} queued")
        
        return {
            "success": True,
            "total_videos": len(entries),
            "processed": success_count,
            "queued": queued
        }
        
    except Exception as e:
        print(f"[SYNC] Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.post("/process-pending")
async def process_pending(limit: int = Query(default=5, ge=1, le=20)):
    """Process pending videos from the queue"""
    if not supabase:
        raise HTTPException(500, "Database not connected")
    
    print(f"\n[PENDING] Processing up to {limit} pending videos...")
    
    # Get pending items
    result = supabase.table("processing_queue")\
        .select("*")\
        .eq("status", "pending")\
        .order("priority", desc=True)\
        .order("created_at")\
        .limit(limit)\
        .execute()
    
    pending = result.data or []
    print(f"[PENDING] Found {len(pending)} pending items")
    
    processed = []
    for i, item in enumerate(pending):
        payload = item.get('payload', {})
        video_url = payload.get('video_url')
        
        if video_url:
            print(f"[PENDING] Processing {i+1}/{len(pending)}: {payload.get('title', 'Unknown')}")
            result = process_single_video(
                video_url=video_url,
                user_id=item['user_id'],
                source_id=item.get('source_id'),
                queue_id=item['id']
            )
            processed.append(result)
    
    success_count = len([p for p in processed if p.get('success')])
    print(f"[PENDING] Complete: {success_count}/{len(processed)} successful")
    
    return {
        "success": True,
        "processed": len(processed),
        "successful": success_count,
        "results": processed
    }


@app.post("/list-videos")
async def list_videos(request: ListVideosRequest):
    """List videos from a profile without downloading"""
    try:
        if request.platform == "tiktok":
            profile_url = f"https://www.tiktok.com/@{request.username}"
        elif request.platform == "youtube":
            profile_url = f"https://www.youtube.com/@{request.username}/videos"
        else:
            raise HTTPException(400, f"Unsupported platform: {request.platform}")
        
        ydl_opts = {
            'extract_flat': True,
            'quiet': True,
            'no_warnings': True,
            'playlist_items': f'1-{request.max_videos}'
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(profile_url, download=False)
            entries = info.get('entries', [])
        
        videos = []
        for entry in entries:
            videos.append({
                "url": entry.get('url') or entry.get('webpage_url'),
                "title": entry.get('title', 'Untitled'),
                "description": entry.get('description', ''),
                "thumbnail": entry.get('thumbnail', ''),
                "duration": entry.get('duration', 0),
                "view_count": entry.get('view_count', 0),
                "like_count": entry.get('like_count', 0)
            })
        
        return {"success": True, "videos": videos, "total": len(videos)}
        
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/process-selected")
async def process_selected(request: ProcessSelectedRequest):
    """Process specific selected videos"""
    try:
        processed = []
        for video in request.videos[:10]:  # Limit to 10 at a time
            result = process_single_video(
                video_url=video.url,
                user_id=request.user_id,
                source_id=request.source_id,
                metadata_override={
                    "title": video.title,
                    "description": video.description,
                    "thumbnail": video.thumbnail,
                    "duration": video.duration,
                    "view_count": video.view_count,
                    "like_count": video.like_count
                }
            )
            processed.append(result)
        
        success_count = len([p for p in processed if p.get('success')])
        return {"success": True, "processed": len(processed), "successful": success_count}
        
    except Exception as e:
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
