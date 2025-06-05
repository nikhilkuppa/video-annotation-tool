#!/usr/bin/env python3
"""
Video Annotation Tool for Content and Format Verification
A Flask web application for annotating videos with content and format labels.
"""

import os
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for
import re
from datetime import datetime
import json

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'

class VideoAnnotationTool:
    def __init__(self):
        self.data_dir = './data'
        self.metadata_file = os.path.join(self.data_dir, 'metadata_sample2.tsv')
        self.content_annotations_file = os.path.join(self.data_dir, 'content_annotations.tsv')
        self.format_annotations_file = os.path.join(self.data_dir, 'format_annotations.tsv')
        self.progress_file = os.path.join(self.data_dir, 'annotation_progress.json')
        
        self.load_data()
        self.load_progress()
    
    def load_data(self):
        """Load metadata and label mappings"""
        try:
            # Load metadata
            self.metadata = pd.read_csv(self.metadata_file, sep='\t')
            print(f"Loaded {len(self.metadata)} videos from metadata")
            
            # Load label mappings
            content_map = pd.read_csv('./data/content-map.tsv', sep='\t')
            format_map = pd.read_csv('./data/format-map.tsv', sep='\t')
            
            self.content_labels = sorted(content_map['map2'].unique())
            self.format_labels = sorted(format_map['map1'].unique())
            
            print(f"Loaded {len(self.content_labels)} content labels and {len(self.format_labels)} format labels")
            
            # Initialize annotation files if they don't exist
            self.init_annotation_files()
            
        except Exception as e:
            print(f"Error loading data: {e}")
            # Create dummy data for testing if files don't exist
            self.metadata = pd.DataFrame({
                'Video': ['test1', 'test2'],
                'title': ['Test Video 1', 'Test Video 2'],
                'url': ['https://www.youtube.com/watch?v=dQw4w9WgXcQ', 'https://www.youtube.com/watch?v=dQw4w9WgXcQ']
            })
            self.content_labels = ['comedy', 'education', 'music', 'news', 'sports']
            self.format_labels = ['animation', 'live-action', 'documentary', 'tutorial', 'vlog']
    
    def init_annotation_files(self):
        """Initialize CSV files for annotations if they don't exist"""
        content_columns = ['Video', 'title', 'url', 'content_label_1', 'content_label_2', 'annotated_at', 'annotator_id']
        format_columns = ['Video', 'title', 'url', 'format_label_1', 'format_label_2', 'annotated_at', 'annotator_id']
        
        if not os.path.exists(self.content_annotations_file):
            pd.DataFrame(columns=content_columns).to_csv(self.content_annotations_file, sep='\t', index=False)
        
        if not os.path.exists(self.format_annotations_file):
            pd.DataFrame(columns=format_columns).to_csv(self.format_annotations_file, sep='\t', index=False)
    
    def load_progress(self):
        """Load annotation progress"""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                self.progress = json.load(f)
        else:
            self.progress = {'annotated_videos': [], 'current_index': 0}
    
    def save_progress(self):
        """Save annotation progress"""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f)
    
    def get_youtube_embed_url(self, url):
        """Convert YouTube URL to embed format"""
        video_id_match = re.search(r'(?:v=|youtu\.be/|embed/)([^&\n?#]+)', url)
        if video_id_match:
            video_id = video_id_match.group(1)
            return f"https://www.youtube.com/embed/{video_id}"
        return url
    
    def get_current_video(self):
        """Get the current video to annotate"""
        if self.progress['current_index'] >= len(self.metadata):
            return None
        
        video_data = self.metadata.iloc[self.progress['current_index']].to_dict()
        video_data['embed_url'] = self.get_youtube_embed_url(video_data['url'])
        video_data['progress'] = {
            'current': self.progress['current_index'] + 1,
            'total': len(self.metadata),
            'percentage': round((self.progress['current_index'] / len(self.metadata)) * 100, 1)
        }
        
        return video_data
    
    def get_video_by_index(self, index):
        """Get a specific video by index"""
        if 0 <= index < len(self.metadata):
            video_data = self.metadata.iloc[index].to_dict()
            video_data['embed_url'] = self.get_youtube_embed_url(video_data['url'])
            video_data['progress'] = {
                'current': index + 1,
                'total': len(self.metadata),
                'percentage': round((index / len(self.metadata)) * 100, 1)
            }
            return video_data
        return None
    
    def save_annotations(self, video_id, content_labels, format_labels, annotator_id='default'):
        """Save annotations to TSV files"""
        timestamp = datetime.now().isoformat()
        
        # Get video metadata
        video_row = self.metadata[self.metadata['Video'] == video_id].iloc[0]
        
        # Prepare content annotation
        content_annotation = {
            'Video': video_id,
            'title': video_row['title'],
            'url': video_row['url'],
            'content_label_1': content_labels[0] if len(content_labels) > 0 else '',
            'content_label_2': content_labels[1] if len(content_labels) > 1 else '',
            'annotated_at': timestamp,
            'annotator_id': annotator_id
        }
        
        # Prepare format annotation
        format_annotation = {
            'Video': video_id,
            'title': video_row['title'],
            'url': video_row['url'],
            'format_label_1': format_labels[0] if len(format_labels) > 0 else '',
            'format_label_2': format_labels[1] if len(format_labels) > 1 else '',
            'annotated_at': timestamp,
            'annotator_id': annotator_id
        }
        
        # Save to files
        pd.DataFrame([content_annotation]).to_csv(
            self.content_annotations_file, sep='\t', mode='a', header=False, index=False
        )
        pd.DataFrame([format_annotation]).to_csv(
            self.format_annotations_file, sep='\t', mode='a', header=False, index=False
        )
        
        # Update progress
        if video_id not in self.progress['annotated_videos']:
            self.progress['annotated_videos'].append(video_id)
            self.progress['current_index'] += 1
            self.save_progress()
        
        return True

# Initialize the annotation tool
annotation_tool = VideoAnnotationTool()

@app.route('/')
def index():
    """Main annotation interface"""
    video = annotation_tool.get_current_video()
    if video is None:
        return render_template('completed.html')
    
    return render_template('annotate.html', 
                         video=video,
                         content_labels=annotation_tool.content_labels,
                         format_labels=annotation_tool.format_labels)

@app.route('/video/<int:index>')
def video_by_index(index):
    """Navigate to specific video by index"""
    video = annotation_tool.get_video_by_index(index)
    if video is None:
        return redirect(url_for('index'))
    
    # Update current index
    annotation_tool.progress['current_index'] = index
    annotation_tool.save_progress()
    
    return render_template('annotate.html',
                         video=video,
                         content_labels=annotation_tool.content_labels,
                         format_labels=annotation_tool.format_labels)

@app.route('/submit', methods=['POST'])
def submit_annotation():
    """Handle annotation submission"""
    try:
        data = request.get_json()
        video_id = data.get('video_id')
        content_labels = data.get('content_labels', [])
        format_labels = data.get('format_labels', [])
        
        # Validation
        if len(content_labels) == 0 or len(format_labels) == 0:
            return jsonify({'success': False, 'error': 'Please select at least one label for both content and format'})
        
        if len(content_labels) > 2 or len(format_labels) > 2:
            return jsonify({'success': False, 'error': 'Please select at most 2 labels for each category'})
        
        # Save annotations
        success = annotation_tool.save_annotations(video_id, content_labels, format_labels)
        
        if success:
            return jsonify({'success': True, 'message': 'Annotation saved successfully'})
        else:
            return jsonify({'success': False, 'error': 'Failed to save annotation'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/progress')
def get_progress():
    """Get current progress"""
    return jsonify({
        'current': annotation_tool.progress['current_index'] + 1,
        'total': len(annotation_tool.metadata),
        'percentage': round((annotation_tool.progress['current_index'] / len(annotation_tool.metadata)) * 100, 1)
    })

if __name__ == '__main__':
    # Create data directory if it doesn't exist
    os.makedirs('./data', exist_ok=True)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)