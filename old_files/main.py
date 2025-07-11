
import os
import whisper
from pathlib import Path
from langchain_community.document_loaders.generic import GenericLoader, FileSystemBlobLoader

from utils import split_in_chunks

# Configuration
VIDEOS_DIRECTORY = "yt_videos/italian"
OUTPUT_DIRECTORY = "transcriptions"
SUPPORTED_FORMATS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v', '.webm', '.mp3', '.wav', '.m4a'}

def setup_directories():
    """Create output directory if it doesn't exist."""
    Path(OUTPUT_DIRECTORY).mkdir(exist_ok=True)
    print(f"Output directory: {OUTPUT_DIRECTORY}")

def get_video_files(directory):
    """Get list of video files from the specified directory."""
    video_files = []
    
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return video_files
    
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            file_ext = Path(file).suffix.lower()
            if file_ext in SUPPORTED_FORMATS:
                video_files.append(file_path)
    
    return video_files

def transcribe_video(model, video_path, output_dir):
    """Transcribe a single video and save the result to a text file."""
    try:
        print(f"Transcribing: {video_path}")
        
        # Transcribe the video
        result = model.transcribe(video_path)
        
        # Create output filename
        video_name = Path(video_path).stem
        output_file = os.path.join(output_dir, f"{video_name}_transcription.txt")
        
        # Save transcription to text file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Transcription for: {video_path}\n")
            f.write("=" * 50 + "\n\n")
            f.write(result["text"])
            
            # Optionally, save segments with timestamps
            f.write("\n\n" + "=" * 50 + "\n")
            f.write("DETAILED SEGMENTS:\n")
            f.write("=" * 50 + "\n\n")
            
            for segment in result["segments"]:
                start_time = segment["start"]
                end_time = segment["end"]
                text = segment["text"]
                f.write(f"[{start_time:.2f}s - {end_time:.2f}s]: {text}\n")
        
        print(f"✓ Transcription saved: {output_file}")
        return True
        
    except Exception as e:
        print(f"✗ Error transcribing {video_path}: {str(e)}")
        return False



def main():
    """Main function to process all videos in the directory."""
    print("Video Transcription Script")
    print("=" * 30)
    
    # Setup
    setup_directories()
    
    # Load Whisper model
    print("Loading Whisper model...")
    try:
        model = whisper.load_model("turbo")
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")
        return
    
    # Get video files
    video_files = get_video_files(VIDEOS_DIRECTORY)
    if not video_files:
        print(f"No video files found in '{VIDEOS_DIRECTORY}'")
        print(f"Supported formats: {', '.join(SUPPORTED_FORMATS)}")
        return
    
    print(f"Found {len(video_files)} video file(s):")
    for video in video_files:
        print(f"  - {video}")
    print()
    
    # Process each video
    successful = 0
    failed = 0
    
    for video_path in video_files:
        if transcribe_video(model, video_path, OUTPUT_DIRECTORY):
            successful += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "=" * 30)
    print("TRANSCRIPTION SUMMARY")
    print("=" * 30)
    print(f"Total videos processed: {len(video_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Transcriptions saved in: {OUTPUT_DIRECTORY}")

if __name__ == "__main__":
    main()
    split_in_chunks()