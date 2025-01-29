import logging
import re
import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import torch
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from transformers import pipeline
import whisper
from pydub import AudioSegment
from tqdm import tqdm

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
    
    def _split_into_max_duration(self, segments: List[Dict]) -> List[List[Dict]]:
        max_duration = self.config['max_chunk_duration']
        min_duration = self.config['min_chunk_duration']
        parts = []
        current_part = []
        current_start = None
        current_end = None

        for seg in segments:
            seg_start = seg['start']
            seg_end = seg['end']
            
            if not current_part:
                current_start = seg_start
                current_end = seg_end
                current_part.append(seg)
                continue
            
            potential_end = seg_end
            potential_duration = potential_end - current_start
            
            if potential_duration <= max_duration:
                current_part.append(seg)
                current_end = seg_end
            else:
                parts.append(current_part)
                current_part = [seg]
                current_start = seg_start
                current_end = seg_end
        
        if current_part:
            parts.append(current_part)
        
        merged_parts = []
        for part in parts:
            part_duration = part[-1]['end'] - part[0]['start']
            if part_duration < min_duration:
                if merged_parts:
                    last_part = merged_parts[-1]
                    combined = last_part + part
                    combined_duration = combined[-1]['end'] - combined[0]['start']
                    if combined_duration <= max_duration:
                        merged_parts[-1] = combined
                    else:
                        merged_parts.append(part)
                else:
                    merged_parts.append(part)
            else:
                merged_parts.append(part)
        
        final_parts = []
        for part in merged_parts:
            part_duration = part[-1]['end'] - part[0]['start']
            if part_duration > max_duration:
                split_subparts = self._split_into_max_duration(part)
                final_parts.extend(split_subparts)
            else:
                final_parts.append(part)
        
        return final_parts
    
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
                        merged = final_chunks[-1] + sub
                        merged_duration = merged[-1]['end'] - merged[0]['start']
                        if merged_duration > self.config['max_chunk_duration']:
                            split_parts = self._split_into_max_duration(merged)
                            final_chunks[-1] = split_parts[0]
                            final_chunks.extend(split_parts[1:])
                        else:
                            final_chunks[-1] = merged
                    else:
                        final_chunks.append(sub)
                elif duration > self.config['max_chunk_duration']:
                    split_parts = self._split_into_max_duration(sub)
                    final_chunks.extend(split_parts)
                else:
                    final_chunks.append(sub)
        
        post_processed = []
        i = 0
        while i < len(final_chunks):
            chunk = final_chunks[i]
            duration = chunk[-1]['end'] - chunk[0]['start']
            
            if duration < self.config['min_chunk_duration']:
                if i < len(final_chunks) - 1:
                    next_chunk = final_chunks[i+1]
                    merged = chunk + next_chunk
                    merged_duration = merged[-1]['end'] - merged[0]['start']
                    if merged_duration > self.config['max_chunk_duration']:
                        split_parts = self._split_into_max_duration(merged)
                        post_processed.extend(split_parts)
                        i += 2
                    else:
                        post_processed.append(merged)
                        i += 2
                else:
                    if post_processed:
                        merged = post_processed[-1] + chunk
                        merged_duration = merged[-1]['end'] - merged[0]['start']
                        if merged_duration > self.config['max_chunk_duration']:
                            split_parts = self._split_into_max_duration(merged)
                            post_processed[-1] = split_parts[0]
                            post_processed.extend(split_parts[1:])
                        else:
                            post_processed[-1] = merged
                    else:
                        post_processed.append(chunk)
                    i += 1
            else:
                post_processed.append(chunk)
                i += 1
        
        final_checked = []
        for chunk in post_processed:
            duration = chunk[-1]['end'] - chunk[0]['start']
            if duration > self.config['max_chunk_duration']:
                split_parts = self._split_into_max_duration(chunk)
                final_checked.extend(split_parts)
            else:
                final_checked.append(chunk)
        
        return final_checked

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
        
        # Validate times
        audio_duration = len(audio) / 1000  # Convert ms to seconds
        start = max(0.0, start)
        end = min(audio_duration, end)
        
        if start >= end:
            logger.warning(f"Invalid chunk times: start={start}, end={end}. Skipping.")
            return None
        
        text = ' '.join(seg['text'] for seg in chunk)
        
        # Convert to milliseconds for audio slicing
        start_ms = int(start * 1000)
        end_ms = int(end * 1000)
        audio_chunk = audio[start_ms:end_ms]
        
        # Handle empty audio (shouldn't occur after validation)
        if len(audio_chunk) == 0:
            logger.warning(f"Empty audio chunk: start={start}s, end={end}s. Skipping.")
            return None
        
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
            if chunk is None:
                continue  # Skip invalid chunks
                
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
    
    model = whisper.load_model(whisper_model, device=device)
    chunker = EnhancedSemanticChunker()
    audio_processor = AudioProcessor()
    
    logger.info("Transcribing audio...")
    result = model.transcribe(
        audio_path,
        verbose=False,
        word_timestamps=True
    )
    
    logger.info("Processing audio file...")
    audio = AudioSegment.from_file(audio_path).set_channels(1)
    
    logger.info("Creating semantic chunks...")
    chunks = chunker.create_chunks(result['segments'])
    
    logger.info("Processing audio chunks...")
    processed_chunks = []
    for chunk in tqdm(chunks, desc="Processing chunks"):
        processed = audio_processor.process_chunk(chunk, audio)
        if processed:  # Only keep valid chunks
            processed_chunks.append(processed)
    
    logger.info("Exporting chunks...")
    manifest = audio_processor.export_chunks(processed_chunks)
    
    logger.info(f"\nGenerated {len(processed_chunks)} contextual chunks:")
    for idx, chunk in enumerate(processed_chunks):
        print(f"\n[Chunk {idx}] ({chunk['duration']:.1f}s)")
        print(f"From {chunk['start']:.1f}s to {chunk['end']:.1f}s")
        print(f"Text: {chunk['text']}")
        print("-" * 80)
    
    return processed_chunks
