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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
nltk.download('punkt', quiet=True)

class EnhancedSemanticChunker:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {
            'embedding_model': 'all-mpnet-base-v2',
            'coref_model': 'coref-roberta-large',
            'min_cluster_size': 5,
            'cluster_selection_epsilon': 0.4,
            'min_chunk_duration': 5.0,
            'max_chunk_duration': 15.0,
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
        for i in tqdm(range(0, len(items), batch_size), desc=desc, unit='batch'):
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
            
        return sorted(clustered_segments.values(), key=lambda x: x[0]['start'])
    
    def split_cluster(self, cluster: List[Dict]) -> List[List[Dict]]:
        sub_clusters = []
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
    
    def split_by_duration(self, cluster: List[Dict], max_duration: float) -> List[List[Dict]]:
        chunks = []
        current_chunk = []
        current_start = cluster[0]['start'] if cluster else 0
        
        for seg in cluster:
            potential_end = seg['end']
            if potential_end - current_start > max_duration:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = [seg]
                    current_start = seg['start']
                else:
                    chunks.append([seg])
                    current_start = seg['end']
            else:
                current_chunk.append(seg)
        if current_chunk:
            chunks.append(current_chunk)
        return chunks
    
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
                else:
                    split_subclusters = self.split_by_duration(sub, self.config['max_chunk_duration'])
                    final_chunks.extend(split_subclusters)
        
        return [chunk for chunk in final_chunks if chunk]

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
        audio_chunk = audio_chunk.set_frame_rate(self.config['target_sample_rate'])
        audio_chunk = audio_chunk.normalize().apply_gain(self.config['normalize_db'])
        
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

def split_long_segments(segments: List[Dict], max_duration: float = 15.0) -> List[Dict]:
    split_segments = []
    for seg in segments:
        seg_duration = seg['end'] - seg['start']
        if seg_duration <= max_duration:
            split_segments.append(seg)
            continue
        words = seg.get('words', [])
        if not words:
            split_segments.append(seg)
            continue
        current_start = seg['start']
        current_words = []
        
        for word in words:
            word_duration = word['end'] - word['start']
            if word_duration > max_duration:
                # Split the word into multiple max_duration segments
                start = word['start']
                end = word['end']
                while start < end:
                    split_end = min(start + max_duration, end)
                    split_word = {
                        'word': word['word'],
                        'start': start,
                        'end': split_end
                    }
                    current_words.append(split_word)
                    split_segments.append({
                        'start': current_start,
                        'end': split_end,
                        'text': word['word'],
                        'words': current_words.copy()
                    })
                    current_start = split_end
                    start = split_end
                    current_words = []
                continue
                
            current_words.append(word)
            current_end = word['end']
            if current_end - current_start > max_duration:
                if len(current_words) > 1:
                    split_words = current_words[:-1]
                    split_end = split_words[-1]['end']
                    split_segments.append({
                        'start': current_start,
                        'end': split_end,
                        'text': ' '.join(w['word'] for w in split_words),
                        'words': split_words
                    })
                    current_start = split_end
                    current_words = [current_words[-1]]
                else:
                    split_segments.append({
                        'start': current_start,
                        'end': current_end,
                        'text': word['word'],
                        'words': [word]
                    })
                    current_start = current_end
                    current_words = []
        if current_words:
            split_segments.append({
                'start': current_start,
                'end': current_words[-1]['end'],
                'text': ' '.join(w['word'] for w in current_words),
                'words': current_words
            })
    return split_segments

def split_by_duration(self, cluster: List[Dict], max_duration: float) -> List[List[Dict]]:
    chunks = []
    current_chunk = []
    current_start = cluster[0]['start'] if cluster else 0
    
    for seg in cluster:
        potential_end = seg['end']
        if potential_end - current_start > max_duration:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = [seg]
                current_start = seg['start']
            else:
                chunks.append([seg])
                current_start = seg['end']
        else:
            current_chunk.append(seg)
    # Handle remaining current_chunk
    if current_chunk:
        chunk_duration = current_chunk[-1]['end'] - current_chunk[0]['start']
        if chunks and chunk_duration < self.config['min_chunk_duration']:
            # Merge with last chunk if total duration doesn't exceed max
            last_chunk = chunks[-1]
            merged_duration = current_chunk[-1]['end'] - last_chunk[0]['start']
            if merged_duration <= self.config['max_chunk_duration']:
                chunks[-1].extend(current_chunk)
            else:
                chunks.append(current_chunk)
        else:
            chunks.append(current_chunk)
    return chunks

def process_audio(
    audio_path: str,
    whisper_model: str = 'tiny.en',
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> List[Dict]:
    logger.info("Initializing pipeline...")
    model = whisper.load_model(whisper_model, device=device)
    chunker = EnhancedSemanticChunker()
    audio_processor = AudioProcessor()
    
    logger.info("Transcribing audio...")
    result = model.transcribe(audio_path, verbose=False, word_timestamps=True)
    
    logger.info("Splitting long segments...")
    segments = split_long_segments(result['segments'], max_duration=chunker.config['max_chunk_duration'])
    
    logger.info("Creating semantic chunks...")
    chunks = chunker.create_chunks(segments)
    
    logger.info("Processing audio file...")
    audio = AudioSegment.from_file(audio_path).set_channels(1)
    
    logger.info("Processing audio chunks...")
    processed_chunks = []
    for chunk in chunks:
        processed = audio_processor.process_chunk(chunk, audio)
        if processed['duration'] < chunker.config['min_chunk_duration']:
            if processed_chunks:
                last = processed_chunks.pop()
                merged_duration = processed['end'] - last['start']
                if merged_duration <= chunker.config['max_chunk_duration']:
                    merged = merge_chunks(last, processed)
                    processed_chunks.append(merged)
                else:
                    # Re-add the last chunk and current processed chunk
                    processed_chunks.append(last)
                    processed_chunks.append(processed)
            else:
                processed_chunks.append(processed)
        else:
            processed_chunks.append(processed)
    
    # Final check to ensure all chunks are within duration limits
    valid_chunks = []
    for chunk in processed_chunks:
        if chunk['duration'] >= chunker.config['min_chunk_duration'] and chunk['duration'] <= chunker.config['max_chunk_duration']:
            valid_chunks.append(chunk)
        else:
            # Handle invalid chunk (shouldn't occur if above logic works)
            logger.warning(f"Invalid chunk duration: {chunk['duration']}s. Text: {chunk['text']}")
    
    logger.info("Exporting chunks...")
    manifest = audio_processor.export_chunks(valid_chunks)
    
    logger.info(f"\nGenerated {len(valid_chunks)} contextual chunks:")
    for idx, chunk in enumerate(valid_chunks):
        print(f"\n[Chunk {idx}] ({chunk['duration']:.1f}s)")
        print(f"From {chunk['start']:.1f}s to {chunk['end']:.1f}s")
        print(f"Text: {chunk['text']}")
        print("-" * 80)
    
    return valid_chunks
