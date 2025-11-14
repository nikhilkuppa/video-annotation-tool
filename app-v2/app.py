"""
Enhanced Video Annotation Tool with Annotation Review Mode
A Flask web application for reviewing and annotating videos with content and format labels.
"""

import os
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for
import re
from datetime import datetime
import json
from collections import defaultdict

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'

class VideoAnnotationTool:
    def __init__(self):
        self.data_dir = './data'
        self.metadata_file = os.path.join(self.data_dir, 'videos_annotate_clear.tsv')
        self.content_annotations_file = os.path.join(self.data_dir, 'content_annotations.tsv')
        self.format_annotations_file = os.path.join(self.data_dir, 'format_annotations.tsv')
        
        # Annotation review file
        self.annotation_review_file = os.path.join(self.data_dir, 'annotation_reviews2.tsv')
        
        self.progress_file = os.path.join(self.data_dir, 'annotation_progress.json')
        
        # Definition files
        self.content_definitions_file = os.path.join(self.data_dir, 'content_definitions.tsv')
        self.format_definitions_file = os.path.join(self.data_dir, 'format_definitions.tsv')
        
        self.load_data()
        self.load_definitions()
        self.load_progress()    

    def load_data(self):
        """Load metadata and create hierarchical label mappings"""
        try:
            # Load metadata
            self.metadata = pd.read_csv(self.metadata_file, sep='\t')
            print(f"Loaded {len(self.metadata)} videos from metadata")
            
            # Create hierarchical mappings for content and format
            self.content_hierarchies = self.build_hierarchies('content')
            self.format_hierarchies = self.build_hierarchies('format')
            
            print(f"Loaded content hierarchies: {list(self.content_hierarchies.keys())}")
            print(f"Loaded format hierarchies: {list(self.format_hierarchies.keys())}")
            
            # Initialize annotation files if they don't exist
            self.init_annotation_files()
            
        except Exception as e:
            print(f"Error loading data: {e}")
            # Create dummy data for testing if files don't exist
            self.metadata = pd.DataFrame({
                'Video': ['test1', 'test2'],
                'title': ['Test Video 1', 'Test Video 2'],
                'url': ['https://www.youtube.com/watch?v=dQw4w9WgXcQ', 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'],
                'content_map_hierarchy1': ['arts/culture', 'lifesciences'],
                'content_map1': ['arts', 'medicine'],
                'format_map_hierarchy1': ['Talking Head', 'Writing Board'],
                'format_map1': ['Talking Head - Center', 'Chalkboard']
            })
            self.content_hierarchies = self.build_hierarchies('content')
            self.format_hierarchies = self.build_hierarchies('format')
    
    def load_definitions(self):
        """Load definition files for content and format labels"""
        self.content_definitions = {}
        self.format_definitions = {}
        
        # Load content definitions
        try:
            if os.path.exists(self.content_definitions_file):
                content_def_df = pd.read_csv(self.content_definitions_file, sep='\t')
                for _, row in content_def_df.iterrows():
                    label = row['mapAnnotation']
                    hierarchy = row['mapAnnotationHierarchy']
                    label_def = row['mapAnnotation_Definition']
                    hierarchy_def = row['mapAnnotationHierarchy_Definition']
                    
                    self.content_definitions[label] = {
                        'label_definition': label_def,
                        'hierarchy': hierarchy,
                        'hierarchy_definition': hierarchy_def
                    }
                print(f"Loaded {len(self.content_definitions)} content definitions")
            else:
                print("Content definitions file not found")
        except Exception as e:
            print(f"Error loading content definitions: {e}")
        
        # Load format definitions
        try:
            if os.path.exists(self.format_definitions_file):
                format_def_df = pd.read_csv(self.format_definitions_file, sep='\t')
                for _, row in format_def_df.iterrows():
                    label = row['mapAnnotation']
                    hierarchy = row['mapAnnotationHierarchy']
                    label_def = row['mapAnnotation_Definition']
                    hierarchy_def = row['mapAnnotationHierarchy_Definition']
                    
                    self.format_definitions[label] = {
                        'label_definition': label_def,
                        'hierarchy': hierarchy,
                        'hierarchy_definition': hierarchy_def
                    }
                print(f"Loaded {len(self.format_definitions)} format definitions")
            else:
                print("Format definitions file not found")
        except Exception as e:
            print(f"Error loading format definitions: {e}")
    
    def build_hierarchies(self, category_type):
        """Build hierarchical structure from metadata"""
        if category_type == 'content':
            # Build triple hierarchy for content: Upper Hierarchy > Hierarchy > Label
            domains = {}
            
            for index in [1, 2]:  # Handle both option 1 and option 2
                upper_hierarchy_col = f'{category_type}_map_hierarchy_upper{index}'
                hierarchy_col = f'{category_type}_map_hierarchy{index}'
                label_col = f'{category_type}_map{index}'
                
                if (upper_hierarchy_col in self.metadata.columns and 
                    hierarchy_col in self.metadata.columns and 
                    label_col in self.metadata.columns):
                    
                    # Group labels by their hierarchies and upper hierarchies (domains)
                    for _, row in self.metadata.iterrows():
                        upper_hierarchy = row.get(upper_hierarchy_col)
                        hierarchy = row.get(hierarchy_col)
                        label = row.get(label_col)
                        
                        if pd.notna(upper_hierarchy) and pd.notna(hierarchy) and pd.notna(label):
                            # Initialize nested structure
                            if upper_hierarchy not in domains:
                                domains[upper_hierarchy] = {}
                            if hierarchy not in domains[upper_hierarchy]:
                                domains[upper_hierarchy][hierarchy] = set()
                                
                            domains[upper_hierarchy][hierarchy].add(label)
            
            # Convert sets to sorted lists and sort hierarchies within domains
            for domain in domains:
                for hierarchy in domains[domain]:
                    domains[domain][hierarchy] = sorted(list(domains[domain][hierarchy]))
                domains[domain] = dict(sorted(domains[domain].items()))
            
            # Sort domains
            return dict(sorted(domains.items()))
        
        else:
            # Keep original logic for format
            hierarchies = defaultdict(set)
            
            for index in [1, 2]:  # Handle both option 1 and option 2
                hierarchy_col = f'{category_type}_map_hierarchy{index}'
                label_col = f'{category_type}_map{index}'
                
                if hierarchy_col in self.metadata.columns and label_col in self.metadata.columns:
                    # Group labels by their hierarchies
                    for _, row in self.metadata.iterrows():
                        hierarchy = row.get(hierarchy_col)
                        label = row.get(label_col)
                        
                        if pd.notna(hierarchy) and pd.notna(label):
                            hierarchies[hierarchy].add(label)
            
            # Convert sets to sorted lists for consistent ordering
            return {hierarchy: sorted(list(labels)) for hierarchy, labels in hierarchies.items()}
    
    def init_annotation_files(self):
        """Initialize CSV files for annotations if they don't exist"""
        content_columns = [
            'Video', 'title', 'url', 
            'content_label_1', 'content_hierarchy_1',
            'content_label_2', 'content_hierarchy_2',
            'gt_content_label_1', 'gt_content_hierarchy_1',
            'gt_content_label_2', 'gt_content_hierarchy_2',
            'annotated_at', 'annotator_id'
        ]
        format_columns = [
            'Video', 'title', 'url',
            'format_label_1', 'format_hierarchy_1', 
            'format_label_2', 'format_hierarchy_2',
            'gt_format_label_1', 'gt_format_hierarchy_1',
            'gt_format_label_2', 'gt_format_hierarchy_2',
            'annotated_at', 'annotator_id'
        ]
        
        # Annotation review columns
        annotation_review_columns = [
            'Video', 'title', 'url',
            'annotation_rating', 'comments',
            'reviewed_at', 'reviewer_id'
        ]
        
        if not os.path.exists(self.content_annotations_file):
            pd.DataFrame(columns=content_columns).to_csv(self.content_annotations_file, sep='\t', index=False)
        
        if not os.path.exists(self.format_annotations_file):
            pd.DataFrame(columns=format_columns).to_csv(self.format_annotations_file, sep='\t', index=False)
        
        if not os.path.exists(self.annotation_review_file):
            pd.DataFrame(columns=annotation_review_columns).to_csv(self.annotation_review_file, sep='\t', index=False)
        
    
    def update_existing_files_headers(self):
        """Update existing annotation files to include hierarchy and ground truth columns if missing"""
        try:
            # Update content annotations file
            if os.path.exists(self.content_annotations_file):
                content_df = pd.read_csv(self.content_annotations_file, sep='\t')
                
                # Add missing columns
                missing_cols = []
                expected_cols = [
                    'Video', 'title', 'url', 
                    'content_label_1', 'content_hierarchy_1',
                    'content_label_2', 'content_hierarchy_2',
                    'gt_content_label_1', 'gt_content_hierarchy_1',
                    'gt_content_label_2', 'gt_content_hierarchy_2',
                    'annotated_at', 'annotator_id'
                ]
                
                for col in expected_cols:
                    if col not in content_df.columns:
                        missing_cols.append(col)
                        content_df[col] = ''
                
                if missing_cols:
                    print(f"Adding missing columns to content annotations: {missing_cols}")
                    content_df = content_df.reindex(columns=expected_cols, fill_value='')
                    content_df.to_csv(self.content_annotations_file, sep='\t', index=False)
            
            # Update format annotations file
            if os.path.exists(self.format_annotations_file):
                format_df = pd.read_csv(self.format_annotations_file, sep='\t')
                
                # Add missing columns
                missing_cols = []
                expected_cols = [
                    'Video', 'title', 'url',
                    'format_label_1', 'format_hierarchy_1', 
                    'format_label_2', 'format_hierarchy_2',
                    'gt_format_label_1', 'gt_format_hierarchy_1',
                    'gt_format_label_2', 'gt_format_hierarchy_2',
                    'annotated_at', 'annotator_id'
                ]
                
                for col in expected_cols:
                    if col not in format_df.columns:
                        missing_cols.append(col)
                        format_df[col] = ''
                
                if missing_cols:
                    print(f"Adding missing columns to format annotations: {missing_cols}")
                    format_df = format_df.reindex(columns=expected_cols, fill_value='')
                    format_df.to_csv(self.format_annotations_file, sep='\t', index=False)
            
            # Update annotation review file
            if os.path.exists(self.annotation_review_file):
                review_df = pd.read_csv(self.annotation_review_file, sep='\t')
                
                # Add missing columns
                missing_cols = []
                expected_cols = [
                    'Video', 'title', 'url',
                    'annotation_rating', 'comments',
                    'reviewed_at', 'reviewer_id'
                ]
                
                for col in expected_cols:
                    if col not in review_df.columns:
                        missing_cols.append(col)
                        review_df[col] = ''
                
                if missing_cols:
                    print(f"Adding missing columns to annotation reviews: {missing_cols}")
                    review_df = review_df.reindex(columns=expected_cols, fill_value='')
                    review_df.to_csv(self.annotation_review_file, sep='\t', index=False)
                    
        except Exception as e:
            print(f"Error updating file headers: {e}")
    
    def load_progress(self):
        """Load annotation progress"""
        try:
            if os.path.exists(self.progress_file):
                with open(self.progress_file, 'r') as f:
                    self.progress = json.load(f)
            else:
                self.progress = {'annotated_videos': [], 'current_index': 0, 'reviewed_videos': []}
        except Exception as e:
            print(f"Error loading progress: {e}")
            # Initialize with default progress if there's any error
            self.progress = {'annotated_videos': [], 'current_index': 0, 'reviewed_videos': []}
    
    def save_progress(self):
        """Save annotation progress"""
        try:
            # Ensure progress exists before saving
            if not hasattr(self, 'progress') or self.progress is None:
                self.progress = {'annotated_videos': [], 'current_index': 0, 'reviewed_videos': []}
            
            os.makedirs(os.path.dirname(self.progress_file), exist_ok=True)
            with open(self.progress_file, 'w') as f:
                json.dump(self.progress, f)
        except Exception as e:
            print(f"Error saving progress: {e}")
    
    def get_youtube_embed_url(self, url):
        """Convert YouTube URL to embed format"""
        video_id_match = re.search(r'(?:v=|youtu\.be/|embed/)([^&\n?#]+)', url)
        if video_id_match:
            video_id = video_id_match.group(1)
            return f"https://www.youtube.com/embed/{video_id}"
        return url
    
    def get_ground_truth_labels(self, video_row):
        """Extract ground truth labels from metadata"""
        ground_truth = {
            'content': [],
            'format': []
        }
        
        # Extract content ground truth
        for i in [1, 2]:
            content_label = video_row.get(f'content_map{i}')
            content_hierarchy = video_row.get(f'content_map_hierarchy{i}')
            if pd.notna(content_label) and pd.notna(content_hierarchy):
                ground_truth['content'].append({
                    'label': content_label,
                    'hierarchy': content_hierarchy
                })
        
        # Extract format ground truth
        for i in [1, 2]:
            format_label = video_row.get(f'format_map{i}')
            format_hierarchy = video_row.get(f'format_map_hierarchy{i}')
            if pd.notna(format_label) and pd.notna(format_hierarchy):
                ground_truth['format'].append({
                    'label': format_label,
                    'hierarchy': format_hierarchy
                })
        
        return ground_truth
    
    def get_current_video(self):
        """Get the current video to annotate"""
        # Ensure progress exists
        if not hasattr(self, 'progress') or self.progress is None:
            self.progress = {'annotated_videos': [], 'current_index': 0, 'reviewed_videos': []}
            
        if self.progress['current_index'] >= len(self.metadata):
            return None
        
        video_data = self.metadata.iloc[self.progress['current_index']].to_dict()
        video_data['embed_url'] = self.get_youtube_embed_url(video_data['url'])
        video_data['progress'] = {
            'current': self.progress['current_index'] + 1,
            'total': len(self.metadata),
            'percentage': round((self.progress['current_index'] / len(self.metadata)) * 100, 1)
        }
        
        # Add ground truth labels
        video_data['ground_truth'] = self.get_ground_truth_labels(video_data)
        
        return video_data
    
    def get_video_by_index(self, index):
        """Get a specific video by index"""
        # Ensure progress exists
        if not hasattr(self, 'progress') or self.progress is None:
            self.progress = {'annotated_videos': [], 'current_index': 0, 'reviewed_videos': []}
            
        if 0 <= index < len(self.metadata):
            video_data = self.metadata.iloc[index].to_dict()
            video_data['embed_url'] = self.get_youtube_embed_url(video_data['url'])
            video_data['progress'] = {
                'current': index + 1,
                'total': len(self.metadata),
                'percentage': round((index / len(self.metadata)) * 100, 1)
            }
            
            # Add ground truth labels
            video_data['ground_truth'] = self.get_ground_truth_labels(video_data)
            
            return video_data
        return None
    
    def save_annotation_review(self, video_id, annotation_rating, comments='', reviewer_id='default'):
        """Save annotation review to TSV file"""
        timestamp = datetime.now().isoformat()
        
        # Ensure progress exists
        if not hasattr(self, 'progress') or self.progress is None:
            self.progress = {'annotated_videos': [], 'current_index': 0, 'reviewed_videos': []}
        
        # Get video metadata
        video_row = self.metadata[self.metadata['Video'] == video_id].iloc[0]
        
        # Check if this video was already reviewed
        already_reviewed = video_id in self.progress.get('reviewed_videos', [])
        
        # Prepare annotation review record
        annotation_review = {
            'Video': video_id,
            'title': video_row['title'],
            'url': video_row['url'],
            'annotation_rating': annotation_rating,
            'comments': comments,
            'reviewed_at': timestamp,
            'reviewer_id': reviewer_id
        }
        
        if already_reviewed:
            # Update existing annotation review by removing old one and adding new one
            try:
                # Read existing file
                review_df = pd.read_csv(self.annotation_review_file, sep='\t')
                
                # Remove existing entry for this video
                review_df = review_df[review_df['Video'] != video_id]
                
                # Add new annotation review
                review_df = pd.concat([review_df, pd.DataFrame([annotation_review])], ignore_index=True)
                
                # Save updated file
                review_df.to_csv(self.annotation_review_file, sep='\t', index=False)
                
            except Exception as e:
                print(f"Error updating annotation review: {e}")
                # Fallback to append mode
                pd.DataFrame([annotation_review]).to_csv(
                    self.annotation_review_file, sep='\t', mode='a', header=False, index=False
                )
                return False

        else:
            # Append new annotation review
            pd.DataFrame([annotation_review]).to_csv(
                self.annotation_review_file, sep='\t', mode='a', header=False, index=False
            )
            
            # Update progress only for new annotation reviews
            if 'reviewed_videos' not in self.progress:
                self.progress['reviewed_videos'] = []
            self.progress['reviewed_videos'].append(video_id)
            self.save_progress()

        return True


    
    def get_existing_annotation_review(self, video_id):
        """Get existing annotation review for a video if it exists"""
        try:
            if os.path.exists(self.annotation_review_file):
                review_df = pd.read_csv(self.annotation_review_file, sep='\t')
                review_row = review_df[review_df['Video'] == video_id]
                if not review_row.empty:
                    latest_review = review_row.iloc[-1]  # Get most recent review
                    return {
                        'rating': latest_review.get('annotation_rating', ''),
                        'comments': latest_review.get('comments', '')
                    }
        except Exception as e:
            print(f"Error loading existing annotation review: {e}")
        
        return None
    
    def save_annotations(self, video_id, content_selections, format_selections, annotator_id='default'):
        """Save annotations with hierarchies and ground truth to TSV files"""
        timestamp = datetime.now().isoformat()
        
        # Ensure progress exists
        if not hasattr(self, 'progress') or self.progress is None:
            self.progress = {'annotated_videos': [], 'current_index': 0, 'reviewed_videos': []}
        
        # Get video metadata
        video_row = self.metadata[self.metadata['Video'] == video_id].iloc[0]
        ground_truth = self.get_ground_truth_labels(video_row)
        
        # Check if this video was already annotated
        already_annotated = video_id in self.progress['annotated_videos']
        
        # Prepare content annotation
        content_annotation = {
            'Video': video_id,
            'title': video_row['title'],
            'url': video_row['url'],
            'content_label_1': content_selections[0]['label'] if len(content_selections) > 0 else '',
            'content_hierarchy_1': content_selections[0]['hierarchy'] if len(content_selections) > 0 else '',
            'content_label_2': content_selections[1]['label'] if len(content_selections) > 1 else '',
            'content_hierarchy_2': content_selections[1]['hierarchy'] if len(content_selections) > 1 else '',
            'gt_content_label_1': ground_truth['content'][0]['label'] if len(ground_truth['content']) > 0 else '',
            'gt_content_hierarchy_1': ground_truth['content'][0]['hierarchy'] if len(ground_truth['content']) > 0 else '',
            'gt_content_label_2': ground_truth['content'][1]['label'] if len(ground_truth['content']) > 1 else '',
            'gt_content_hierarchy_2': ground_truth['content'][1]['hierarchy'] if len(ground_truth['content']) > 1 else '',
            'annotated_at': timestamp,
            'annotator_id': annotator_id
        }
        
        # Prepare format annotation
        format_annotation = {
            'Video': video_id,
            'title': video_row['title'],
            'url': video_row['url'],
            'format_label_1': format_selections[0]['label'] if len(format_selections) > 0 else '',
            'format_hierarchy_1': format_selections[0]['hierarchy'] if len(format_selections) > 0 else '',
            'format_label_2': format_selections[1]['label'] if len(format_selections) > 1 else '',
            'format_hierarchy_2': format_selections[1]['hierarchy'] if len(format_selections) > 1 else '',
            'gt_format_label_1': ground_truth['format'][0]['label'] if len(ground_truth['format']) > 0 else '',
            'gt_format_hierarchy_1': ground_truth['format'][0]['hierarchy'] if len(ground_truth['format']) > 0 else '',
            'gt_format_label_2': ground_truth['format'][1]['label'] if len(ground_truth['format']) > 1 else '',
            'gt_format_hierarchy_2': ground_truth['format'][1]['hierarchy'] if len(ground_truth['format']) > 1 else '',
            'annotated_at': timestamp,
            'annotator_id': annotator_id
        }
        
        if already_annotated:
            # Update existing annotations by removing old ones and adding new ones
            try:
                # Read existing files
                content_df = pd.read_csv(self.content_annotations_file, sep='\t')
                format_df = pd.read_csv(self.format_annotations_file, sep='\t')
                
                # Remove existing entries for this video
                content_df = content_df[content_df['Video'] != video_id]
                format_df = format_df[format_df['Video'] != video_id]
                
                # Add new annotations
                content_df = pd.concat([content_df, pd.DataFrame([content_annotation])], ignore_index=True)
                format_df = pd.concat([format_df, pd.DataFrame([format_annotation])], ignore_index=True)
                
                # Save updated files
                content_df.to_csv(self.content_annotations_file, sep='\t', index=False)
                format_df.to_csv(self.format_annotations_file, sep='\t', index=False)
                
            except Exception as e:
                print(f"Error updating annotations: {e}")
                # Fallback to append mode
                pd.DataFrame([content_annotation]).to_csv(
                    self.content_annotations_file, sep='\t', mode='a', header=False, index=False
                )
                pd.DataFrame([format_annotation]).to_csv(
                    self.format_annotations_file, sep='\t', mode='a', header=False, index=False
                )
        else:
            # Append new annotations
            pd.DataFrame([content_annotation]).to_csv(
                self.content_annotations_file, sep='\t', mode='a', header=False, index=False
            )
            pd.DataFrame([format_annotation]).to_csv(
                self.format_annotations_file, sep='\t', mode='a', header=False, index=False
            )
            
            # Update progress only for new annotations
            self.progress['annotated_videos'].append(video_id)
            self.save_progress()

        return True
        
    def get_existing_annotations(self, video_id):
        """Get existing annotations for a video if they exist"""
        existing_annotations = {
            'content': {'label_1': '', 'hierarchy_1': '', 'label_2': '', 'hierarchy_2': ''},
            'format': {'label_1': '', 'hierarchy_1': '', 'label_2': '', 'hierarchy_2': ''}
        }
        
        try:
            # Check content annotations
            if os.path.exists(self.content_annotations_file):
                content_df = pd.read_csv(self.content_annotations_file, sep='\t')
                content_row = content_df[content_df['Video'] == video_id]
                if not content_row.empty:
                    latest_content = content_row.iloc[-1]  # Get most recent annotation
                    existing_annotations['content'] = {
                        'label_1': latest_content.get('content_label_1', ''),
                        'hierarchy_1': latest_content.get('content_hierarchy_1', ''),
                        'label_2': latest_content.get('content_label_2', ''),
                        'hierarchy_2': latest_content.get('content_hierarchy_2', '')
                    }
            
            # Check format annotations
            if os.path.exists(self.format_annotations_file):
                format_df = pd.read_csv(self.format_annotations_file, sep='\t')
                format_row = format_df[format_df['Video'] == video_id]
                if not format_row.empty:
                    latest_format = format_row.iloc[-1]  # Get most recent annotation
                    existing_annotations['format'] = {
                        'label_1': latest_format.get('format_label_1', ''),
                        'hierarchy_1': latest_format.get('format_hierarchy_1', ''),
                        'label_2': latest_format.get('format_label_2', ''),
                        'hierarchy_2': latest_format.get('format_hierarchy_2', '')
                    }
        except Exception as e:
            print(f"Error loading existing annotations: {e}")
        
        return existing_annotations

# Initialize the annotation tool
annotation_tool = VideoAnnotationTool()

# Annotation Review Routes
@app.route('/annotation_review')
def annotation_review_index():
    """Main annotation review interface"""
    video = annotation_tool.get_current_video()
    if video is None:
        return render_template('completed.html')
    
    # Get existing annotation review for this video
    existing_annotation_review = annotation_tool.get_existing_annotation_review(video['Video'])
    
    return render_template('annotation_review.html', 
                         video=video,
                         existing_annotation_review=existing_annotation_review)

@app.route('/annotation_review/<int:index>')
def annotation_review_by_index(index):
    """Navigate to specific video for annotation review by index"""
    video = annotation_tool.get_video_by_index(index)
    if video is None:
        return redirect(url_for('annotation_review_index'))
    
    # Ensure progress exists and update current index
    if not hasattr(annotation_tool, 'progress') or annotation_tool.progress is None:
        annotation_tool.progress = {'annotated_videos': [], 'current_index': 0, 'reviewed_videos': []}
    
    annotation_tool.progress['current_index'] = index
    annotation_tool.save_progress()
    
    # Get existing annotation review for this video
    existing_annotation_review = annotation_tool.get_existing_annotation_review(video['Video'])
    
    return render_template('annotation_review.html',
                         video=video,
                         existing_annotation_review=existing_annotation_review)

@app.route('/submit_annotation_review', methods=['POST'])
def submit_annotation_review():
    """Handle annotation review submission"""
    try:
        data = request.get_json()
        video_id = data.get('video_id')
        annotation_rating = data.get('annotation_rating')
        comments = data.get('comments', '')
        
        # Validation
        if not annotation_rating or annotation_rating not in ['good', 'bad']:
            return jsonify({'success': False, 'error': 'Please select a valid annotation assessment'})
        
        # Save annotation review
        success = annotation_tool.save_annotation_review(video_id, annotation_rating, comments)
        
        if success:
            return jsonify({'success': True, 'message': 'Annotation review saved successfully'})
        else:
            # print('Failed to save annotation review')
            return jsonify({'success': False, 'error': 'Failed to save annotation review'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Original Annotation Routes
@app.route('/')
def index():
    """Main annotation interface"""
    video = annotation_tool.get_current_video()
    if video is None:
        return render_template('completed.html')
    
    # Get existing annotations for this video
    existing_annotations = annotation_tool.get_existing_annotations(video['Video'])
    
    return render_template('annotate_with_definitions.html', 
                         video=video,
                         content_hierarchies=annotation_tool.content_hierarchies,
                         format_hierarchies=annotation_tool.format_hierarchies,
                         content_definitions=annotation_tool.content_definitions,
                         format_definitions=annotation_tool.format_definitions,
                         existing_annotations=existing_annotations)

@app.route('/video/<int:index>')
def video_by_index(index):
    """Navigate to specific video by index"""
    video = annotation_tool.get_video_by_index(index)
    if video is None:
        return redirect(url_for('index'))
    
    # Ensure progress exists and update current index
    if not hasattr(annotation_tool, 'progress') or annotation_tool.progress is None:
        annotation_tool.progress = {'annotated_videos': [], 'current_index': 0, 'reviewed_videos': []}
    
    annotation_tool.progress['current_index'] = index
    annotation_tool.save_progress()
    
    # Get existing annotations for this video
    existing_annotations = annotation_tool.get_existing_annotations(video['Video'])
    
    return render_template('annotate_with_definitions.html',
                         video=video,
                         content_hierarchies=annotation_tool.content_hierarchies,
                         format_hierarchies=annotation_tool.format_hierarchies,
                         content_definitions=annotation_tool.content_definitions,
                         format_definitions=annotation_tool.format_definitions,
                         existing_annotations=existing_annotations)

@app.route('/api/definition/<category>/<label>')
def get_definition(category, label):
    """Get definition for a specific label"""
    if category == 'content':
        definition_data = annotation_tool.content_definitions.get(label, {})
    elif category == 'format':
        definition_data = annotation_tool.format_definitions.get(label, {})
    else:
        return jsonify({'error': 'Invalid category'})
    
    return jsonify(definition_data)

@app.route('/submit', methods=['POST'])
def submit_annotation():
    """Handle annotation submission"""
    try:
        data = request.get_json()
        video_id = data.get('video_id')
        content_selections = data.get('content_selections', [])
        format_selections = data.get('format_selections', [])
        
        # Validation
        if len(content_selections) == 0 or len(format_selections) == 0:
            return jsonify({'success': False, 'error': 'Please select at least one label for both content and format'})
        
        if len(content_selections) > 2 or len(format_selections) > 2:
            return jsonify({'success': False, 'error': 'Please select at most 2 labels for each category'})
        
        # Save annotations
        success = annotation_tool.save_annotations(video_id, content_selections, format_selections)
        
        if success:
            return jsonify({'success': True, 'message': 'Annotation saved successfully'})
        else:
            return jsonify({'success': False, 'error': 'Failed to save annotation'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/progress')
def get_progress():
    """Get current progress"""
    # Ensure progress exists
    if not hasattr(annotation_tool, 'progress') or annotation_tool.progress is None:
        annotation_tool.progress = {'annotated_videos': [], 'current_index': 0, 'reviewed_videos': []}
        
    return jsonify({
        'current': annotation_tool.progress['current_index'] + 1,
        'total': len(annotation_tool.metadata),
        'percentage': round((annotation_tool.progress['current_index'] / len(annotation_tool.metadata)) * 100, 1),
        'reviewed': len(annotation_tool.progress.get('reviewed_videos', []))
    })

if __name__ == '__main__':
    # Create data directory if it doesn't exist
    os.makedirs('./data', exist_ok=True)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)