import streamlit as st
import yt_dlp
from pydub import AudioSegment
from model import process_audio
from io import BytesIO
import requests
import os
import tempfile

def get_audio_url(url):
    try:
        ydl_opts = {
            'format': 'bestaudio/best',  # Select best audio quality
            'quiet': True,
            'force_generic_extractor': True,  # Ensure generic extractor for URL
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            # Extract the best audio format URL and extension
            if 'url' in info:
                return info['url'], info.get('ext', 'wav')
            else:
                # Fallback to first available audio format
                for f in info.get('formats', []):
                    if f.get('acodec') != 'none' and f.get('vcodec') == 'none':
                        return f['url'], f.get('ext', 'wav')
                st.error("No audio-only format found.")
                return None, None
    except Exception as e:
        st.error(f"Error extracting audio URL: {str(e)}")
        return None, None

st.title("YouTube Audio Downloader & Segment Explorer")
url = st.text_input("Enter YouTube URL:")

if st.button("Process Audio"):
    if url:
        if 'audio_chunks' in st.session_state:
            del st.session_state['audio_chunks']
        
        with st.spinner("Processing..."):
            audio_url, ext = get_audio_url(url)
            
            if audio_url and ext:
                try:
                    response = requests.get(audio_url)
                    response.raise_for_status()  # Check for HTTP errors
                    audio_data = BytesIO(response.content)
                    # Load audio using the correct format
                    audio = AudioSegment.from_file(audio_data, format=ext)
                    
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                        audio.export(tmpfile.name, format="wav")
                        chunks = process_audio(tmpfile.name)
                    
                    if chunks:
                        st.session_state['audio_chunks'] = chunks
                        st.success("Audio processed into segments!")
                    else:
                        st.error("Failed to process audio into segments.")
                except Exception as e:
                    st.error(f"Processing error: {str(e)}")
            else:
                st.error("Invalid audio URL or format.")
    else:
        st.warning("Please enter a YouTube URL.")

if 'audio_chunks' in st.session_state:
    st.subheader("Audio Segments")
    for idx, chunk in enumerate(st.session_state['audio_chunks']):
        with st.expander(f"Segment {idx + 1} (00:{chunk['start']:.2f} - 00:{chunk['end']:.2f})"):
            st.write(f"**Transcript:** {chunk['text']}")
            audio_buffer = BytesIO()
            chunk['audio'].export(audio_buffer, format="wav")
            st.audio(audio_buffer.getvalue(), format="audio/wav")

if st.button("Clear All Segments"):
    if 'audio_chunks' in st.session_state:
        del st.session_state['audio_chunks']
