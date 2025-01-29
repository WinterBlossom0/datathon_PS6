import os
import re
import json
import logging
from typing import List, Dict, Any
from pathlib import Path
from tqdm import tqdm
import numpy as np
from pydub import AudioSegment
import whisper
from sklearn.cluster import DBSCAN
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize
import torch
print(torch.classes)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK resources
nltk.download('punkt', quiet=True)

class EnhancedSemanticChunker:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {
            'embedding_model': 'all-mpnet-base-v2',
            'coref_model': 'coref-roberta-large',
            'min_cluster_size': 5,
            'cluster_selection_epsilon': 0.4,
            'min_chunk_duration': 10.0,
            'max_chunk_duration': 60.0,
            'target_embedding_batch_size': 16,
            'discourse_markers': [
                'however', 'but', 'on the other hand', 'meanwhile', 
                'moving on', 'next', 'finally', 'in conclusion',
                'additionally', 'firstly', 'secondly', 'therefore',
                'moreover', 'as a result', 'in summary', 'consequently',
                'furthermore', 'nevertheless', 'that said', 'similarly',
                'conversely', 'specifically', 'for example', 'in contrast'
            ]
        }
        
        self.embedder = SentenceTransformer(
            self.config['embedding_model'],
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        self.coref_model = self._init_coref_model()
        self.discourse_pattern = self._build_discourse_regex()
        
    def _init_coref_model(self):
        try:
            return pipeline(
                "coreference-resolution",
                model=self.config['coref_model'],
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            logger.warning(f"Coreference model failed to load: {e}")
            return None
            
    def _build_discourse_regex(self):
        markers = [re.escape(marker) for marker in self.config['discourse_markers']]
        pattern = fr"\b({'|'.join(markers)})\b"
        return re.compile(pattern, re.IGNORECASE)
    
    def _batch_process(self, items, batch_size, process_fn, desc):
        results = []
        for i in tqdm(range(0, len(items), batch_size), 
                    desc=desc, unit='batch'):
            batch = items[i:i+batch_size]
            results.extend(process_fn(batch))
        return results
    
    def resolve_coreferences(self, text: str) -> str:
        if not self.coref_model:
            return text
            
        try:
            sentences = sent_tokenize(text)
            resolved_text = []
            buffer = []
            
            for sent in sentences:
                buffer.append(sent)
                if len(' '.join(buffer)) > 1000:
                    chunk = ' '.join(buffer[:-1])
                    resolved = self.coref_model(chunk)
                    resolved_text.append(resolved['resolved_text'])
                    buffer = [buffer[-1]]
            
            if buffer:
                chunk = ' '.join(buffer)
                resolved = self.coref_model(chunk)
                resolved_text.append(resolved['resolved_text'])
                
            return ' '.join(resolved_text)
            
        except Exception as e:
            logger.error(f"Coreference resolution failed: {e}")
            return text
    
    def process_segments(self, segments: List[Dict]) -> List[Dict]:
        full_text = ' '.join(seg['text'] for seg in segments)
        resolved_text = self.resolve_coreferences(full_text)
        
        # Realign with original timing using ratio-based approach
        orig_words = full_text.split()
        resolved_words = resolved_text.split()
        alignment_ratio = len(resolved_words) / len(orig_words) if orig_words else 1
        
        processed_segments = []
        current_resolved_idx = 0
        
        for seg in segments:
            orig_word_count = len(seg['text'].split())
            resolved_word_count = int(orig_word_count * alignment_ratio)
            chunk_words = resolved_words[current_resolved_idx:current_resolved_idx+resolved_word_count]
            
            processed_segments.append({
                'start': seg['start'],
                'end': seg['end'],
                'text': ' '.join(chunk_words).strip()
            })
            current_resolved_idx += resolved_word_count
            
        return processed_segments
    
    def cluster_segments(self, embeddings: np.ndarray, segments: List[Dict]) -> List[List[Dict]]:
        clusterer = DBSCAN(
            eps=self.config['cluster_selection_epsilon'],
            min_samples=self.config['min_cluster_size'],
            metric='cosine'
        )
        
        clusters = clusterer.fit_predict(embeddings)
        
        clustered_segments = {}
        for idx, cluster_id in enumerate(clusters):
            if cluster_id == -1:
                cluster_id = f"noise_{idx}"
            clustered_segments.setdefault(cluster_id, []).append(segments[idx])
            
        return sorted(clustered_segments.values(), 
                     key=lambda x: x[0]['start'])
    
    def split_cluster(self, cluster: List[Dict]) -> List[List[Dict]]:
        sub_clusters = []
        current_cluster = []
        split_points = []
        
        for idx, seg in enumerate(cluster):
            if self.discourse_pattern.search(seg['text']):
                split_points.append(idx)
        
        if not split_points:
            return [cluster]
            
        split_ranges = zip([0] + split_points, split_points + [len(cluster)])
        for start, end in split_ranges:
            if end - start > 0:
                sub_clusters.append(cluster[start:end])
                
        return sub_clusters
    
    def create_chunks(self, segments: List[Dict]) -> List[List[Dict]]:
        processed_segments = self.process_segments(segments)
        
        texts = [seg['text'] for seg in processed_segments]
        embeddings = self.embedder.encode(
            texts,
            batch_size=self.config['target_embedding_batch_size'],
            show_progress_bar=True,
            convert_to_tensor=True,
            normalize_embeddings=True
        ).cpu().numpy()
        
        clusters = self.cluster_segments(embeddings, processed_segments)
        
        final_chunks = []
        for cluster in clusters:
            sub_clusters = self.split_cluster(cluster)
            
            for sub in sub_clusters:
                duration = sub[-1]['end'] - sub[0]['start']
                if duration < self.config['min_chunk_duration']:
                    if final_chunks:
                        final_chunks[-1].extend(sub)
                    else:
                        final_chunks.append(sub)
                elif duration > self.config['max_chunk_duration']:
                    midpoint = len(sub) // 2
                    final_chunks.extend([sub[:midpoint], sub[midpoint:]])
                else:
                    final_chunks.append(sub)
        
        return final_chunks

class AudioProcessor:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {
            'fade_duration': 100,
            'target_sample_rate': 16000,
            'normalize_db': -20.0,
            'crossfade_duration': 200
        }
        
    def process_chunk(self, chunk: List[Dict], audio: AudioSegment) -> Dict:
        start = chunk[0]['start']
        end = chunk[-1]['end']
        text = ' '.join(seg['text'] for seg in chunk)
        
        audio_chunk = audio[start*1000 : end*1000]
        
        # Audio processing
        audio_chunk = audio_chunk.set_frame_rate(self.config['target_sample_rate'])
        audio_chunk = audio_chunk.normalize().apply_gain(self.config['normalize_db'])
        
        # Apply crossfade between chunks
        if self.config['crossfade_duration'] > 0:
            audio_chunk = audio_chunk.fade_in(self.config['fade_duration'])\
                                    .fade_out(self.config['fade_duration'])
        
        return {
            'audio': audio_chunk,
            'text': text,
            'start': start,
            'end': end,
            'duration': end - start
        }
    
    def export_chunks(self, chunks: List[Dict], output_dir: str = "chunks"):
        Path(output_dir).mkdir(exist_ok=True)
        
        manifest = []
        for idx, chunk in enumerate(chunks):
            filename = f"chunk_{idx:04d}.wav"
            metadata = {
                'id': idx,
                'filename': filename,
                'start': chunk['start'],
                'end': chunk['end'],
                'duration': chunk['duration'],
                'text': chunk['text']
            }
            
            chunk['audio'].export(str(Path(output_dir)/filename), format="wav")
            manifest.append(metadata)
        
        with open(Path(output_dir)/'manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)
            
        return manifest

def process_audio(
    audio_path: str,
    whisper_model: str = 'tiny.en',
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> List[Dict]:
    logger.info("Initializing pipeline...")
    
    # Load models
    logger.info("Loading Whisper model...")
    model = whisper.load_model(whisper_model, device=device)
    
    chunker = EnhancedSemanticChunker()
    audio_processor = AudioProcessor()
    
    # Transcribe audio
    logger.info("Transcribing audio...")
    result = model.transcribe(
        audio_path,
        verbose=False,
        word_timestamps=True
    )
    
    # Load and preprocess audio
    logger.info("Processing audio file...")
    audio = AudioSegment.from_file(audio_path).set_channels(1)
    
    # Create chunks
    logger.info("Creating semantic chunks...")
    chunks = chunker.create_chunks(result['segments'])
    
    # Process audio chunks
    logger.info("Processing audio chunks...")
    processed_chunks = [
        audio_processor.process_chunk(chunk, audio)
        for chunk in tqdm(chunks, desc="Processing chunks")
    ]
    
    # Filter and merge chunks
    final_chunks = []
    for chunk in processed_chunks:
        if chunk['duration'] < chunker.config['min_chunk_duration']:
            if final_chunks:
                final_chunks[-1] = merge_chunks(final_chunks[-1], chunk)
            else:
                final_chunks.append(chunk)
        else:
            final_chunks.append(chunk)
    
    # Export results
    logger.info("Exporting chunks...")
    manifest = audio_processor.export_chunks(final_chunks)
    
    # Print summary
    logger.info(f"\nGenerated {len(final_chunks)} contextual chunks:")
    for idx, chunk in enumerate(final_chunks):
        print(f"\n[Chunk {idx}] ({chunk['duration']:.1f}s)")
        print(f"From {chunk['start']:.1f}s to {chunk['end']:.1f}s")
        print(f"Text: {chunk['text']}")
        print("-" * 80)
    
    return final_chunks

def merge_chunks(chunk1: Dict, chunk2: Dict) -> Dict:
    merged_audio = chunk1['audio'] + chunk2['audio']
    return {
        'audio': merged_audio,
        'text': chunk1['text'] + " " + chunk2['text'],
        'start': chunk1['start'],
        'end': chunk2['end'],
        'duration': chunk2['end'] - chunk1['start']
    }

    
