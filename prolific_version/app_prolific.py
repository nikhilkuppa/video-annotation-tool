#!/usr/bin/env python3
"""
Multi-User Video Annotation Tool with Prolific Integration
A Flask web application for annotating videos with content and format labels organized by hierarchy.

Features:
- Prolific integration with PID-based user management
- Random video subset assignment (30 videos per participant)
- Balanced coverage ensuring each video gets ≥3 annotations
- Triple hierarchy for content labels (Upper Hierarchy > Hierarchy > Label)
- Two-level hierarchy for format labels (Hierarchy > Label)
- Mandatory 2 selections each for content and format
- Automatic navigation and re-annotation capability
- Ground truth storage alongside annotations
"""

import os
import re
import json
import glob
import random
import pandas as pd
from datetime import datetime
from collections import defaultdict
from flask import Flask, render_template, request, jsonify, redirect, url_for

# =============================================================================
# FLASK APP CONFIGURATION
# =============================================================================

from config import ConfigVariables
app = Flask(__name__)
app.config.from_object(ConfigVariables)

# =============================================================================
# VIDEO ASSIGNMENT MANAGER
# =============================================================================

class VideoAssignmentManager:
    """
    Manages video assignments to ensure balanced coverage across participants.
    Each video should be annotated by at least 3 people.
    """
    
    def __init__(self, data_dir='./data'):
        self.data_dir = data_dir
        self.assignment_file = os.path.join(data_dir, 'video_assignments.json')
        self.videos_per_participant = 30
        self.min_annotations_per_video = 3
        self.max_participants = 25
        
    def create_master_assignment(self, total_videos=250):
        """
        Create master assignment plan ensuring balanced coverage.
        250 videos × 3 annotations = 750 total assignments
        25 participants × 30 videos = 750 total capacity
        """
        # Create assignment matrix
        assignments = {}
        video_indices = list(range(total_videos))
        
        # Calculate how many times each video should be assigned
        total_assignments = self.max_participants * self.videos_per_participant
        assignments_per_video = total_assignments // total_videos  # Should be 3
        
        # Create a list with each video repeated 3 times
        video_pool = []
        for video_idx in video_indices:
            for _ in range(assignments_per_video):
                video_pool.append(video_idx)
        
        # Shuffle the pool
        random.shuffle(video_pool)
        
        # Assign videos to participants
        for participant_idx in range(self.max_participants):
            start_idx = participant_idx * self.videos_per_participant
            end_idx = start_idx + self.videos_per_participant
            participant_videos = video_pool[start_idx:end_idx]
            
            # Shuffle each participant's videos
            random.shuffle(participant_videos)
            
            assignments[f"slot_{participant_idx}"] = {
                'videos': participant_videos,
                'assigned_to': None,
                'created_at': datetime.now().isoformat()
            }
        
        # Save master assignment
        master_data = {
            'assignments': assignments,
            'total_videos': total_videos,
            'videos_per_participant': self.videos_per_participant,
            'max_participants': self.max_participants,
            'created_at': datetime.now().isoformat()
        }
        
        os.makedirs(os.path.dirname(self.assignment_file), exist_ok=True)
        with open(self.assignment_file, 'w') as f:
            json.dump(master_data, f, indent=2)
        
        print(f"Created master assignment plan for {self.max_participants} participants")
        return master_data
    
    def get_assignment_for_participant(self, prolific_pid):
        """
        Get video assignment for a specific participant.
        If no assignment exists, assign the next available slot.
        """
        # Load existing assignments
        if os.path.exists(self.assignment_file):
            with open(self.assignment_file, 'r') as f:
                master_data = json.load(f)
        else:
            # Create master assignment if it doesn't exist
            master_data = self.create_master_assignment()
        
        assignments = master_data['assignments']
        
        # Check if this participant already has an assignment
        for slot_id, slot_data in assignments.items():
            if slot_data['assigned_to'] == prolific_pid:
                return slot_data['videos']
        
        # Find next available slot
        for slot_id, slot_data in assignments.items():
            if slot_data['assigned_to'] is None:
                # Assign this slot to the participant
                assignments[slot_id]['assigned_to'] = prolific_pid
                assignments[slot_id]['assigned_at'] = datetime.now().isoformat()
                
                # Save updated assignments
                with open(self.assignment_file, 'w') as f:
                    json.dump(master_data, f, indent=2)
                
                print(f"Assigned slot {slot_id} to participant {prolific_pid}")
                return slot_data['videos']
        
        # No slots available
        raise Exception("No available slots for new participants")

# Global assignment manager
assignment_manager = VideoAssignmentManager()

# =============================================================================
# VIDEO ANNOTATION TOOL CLASS
# =============================================================================

class VideoAnnotationTool:
    """
    Handles video annotation logic for a specific annotator.
    Now integrated with Prolific and video assignment system.
    """

    def __init__(self, prolific_pid):
        """Initialize annotation tool for Prolific participant"""
        self.data_dir = './data'
        self.prolific_pid = prolific_pid

        # Shared files (same for all users)
        self.metadata_file = os.path.join(self.data_dir, 'videos_annotate_clear.tsv')
        self.content_definitions_file = os.path.join(self.data_dir, 'content_definitions.tsv')
        self.format_definitions_file = os.path.join(self.data_dir, 'format_definitions.tsv')

        # Prolific-specific files
        self.user_dir = os.path.join(self.data_dir, 'prolific', prolific_pid)
        os.makedirs(self.user_dir, exist_ok=True)

        self.content_annotations_file = os.path.join(self.user_dir, 'content_annotations.tsv')
        self.format_annotations_file = os.path.join(self.user_dir, 'format_annotations.tsv')
        self.progress_file = os.path.join(self.user_dir, 'annotation_progress.json')
        self.participant_info_file = os.path.join(self.user_dir, 'participant_info.json')

        # Initialize the tool
        self.load_data()
        self.load_definitions()
        self.load_progress()
        self.get_assigned_videos()

    # -------------------------------------------------------------------------
    # DATA LOADING METHODS
    # -------------------------------------------------------------------------

    def load_data(self):
        """Load metadata and create hierarchical label mappings"""
        try:
            # Load metadata
            self.metadata = pd.read_csv(self.metadata_file, sep='\t')
            
            # Filter to keep only 250 videos (drop 6 specific ones as requested)
            self.metadata = self.filter_metadata_to_250()
            
            # Clean metadata columns
            for col in self.metadata.columns:
                if self.metadata[col].dtype == 'object':
                    self.metadata[col] = self.metadata[col].astype(str).str.strip()

            print(f"Loaded {len(self.metadata)} videos from metadata (filtered to 250)")

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
                'content_map_hierarchy_upper1': ['Educational', 'Educational'],
                'content_map_hierarchy1': ['sciences', 'mathematics'],
                'content_map1': ['physics', 'algebra'],
                'format_map_hierarchy1': ['Talking Head', 'Writing Board'],
                'format_map1': ['Talking Head - Center', 'Chalkboard']
            })
            self.content_hierarchies = self.build_hierarchies('content')
            self.format_hierarchies = self.build_hierarchies('format')

    def filter_metadata_to_250(self):
        """
        Filter metadata to exactly 250 videos by removing 6 specific videos
        as requested: drop 1 each from Physics, Movies/Theatre/Drama, News, Mental Health, Medicine, Personal Development
        where format label count > 5
        """
        # For now, just take first 250 videos
        # TODO: Implement specific filtering logic based on content and format criteria
        return self.metadata.head(250)

    def build_hierarchies(self, category_type):
        """Build hierarchical structure from metadata using new column structure"""
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

    def get_assigned_videos(self):
        """Get the assigned video indices for this participant"""
        try:
            self.assigned_video_indices = assignment_manager.get_assignment_for_participant(self.prolific_pid)
            print(f"Participant {self.prolific_pid} assigned {len(self.assigned_video_indices)} videos")
        except Exception as e:
            print(f"Error getting video assignment: {e}")
            # Fallback: assign first 30 videos
            self.assigned_video_indices = list(range(min(30, len(self.metadata))))

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

    def load_progress(self):
        """Load annotation progress"""
        try:
            if os.path.exists(self.progress_file):
                with open(self.progress_file, 'r') as f:
                    self.progress = json.load(f)
            else:
                self.progress = {'annotated_videos': [], 'current_index': 0}
        except Exception as e:
            print(f"Error loading progress: {e}")
            self.progress = {'annotated_videos': [], 'current_index': 0}

    # -------------------------------------------------------------------------
    # FILE MANAGEMENT METHODS
    # -------------------------------------------------------------------------

    def init_annotation_files(self):
        """Initialize CSV files for annotations if they don't exist"""
        content_columns = [
            'Video', 'title', 'url',
            'content_label_1', 'content_hierarchy_1', 'content_upper_hierarchy_1',
            'content_label_2', 'content_hierarchy_2', 'content_upper_hierarchy_2',
            'gt_content_label_1', 'gt_content_hierarchy_1', 'gt_content_upper_hierarchy_1',
            'gt_content_label_2', 'gt_content_hierarchy_2', 'gt_content_upper_hierarchy_2',
            'annotated_at', 'prolific_pid'
        ]
        format_columns = [
            'Video', 'title', 'url',
            'format_label_1', 'format_hierarchy_1',
            'format_label_2', 'format_hierarchy_2',
            'gt_format_label_1', 'gt_format_hierarchy_1',
            'gt_format_label_2', 'gt_format_hierarchy_2',
            'annotated_at', 'prolific_pid'
        ]

        if not os.path.exists(self.content_annotations_file):
            pd.DataFrame(columns=content_columns).to_csv(self.content_annotations_file, sep='\t', index=False)

        if not os.path.exists(self.format_annotations_file):
            pd.DataFrame(columns=format_columns).to_csv(self.format_annotations_file, sep='\t', index=False)

    def save_progress(self):
        """Save annotation progress"""
        try:
            if not hasattr(self, 'progress') or self.progress is None:
                self.progress = {'annotated_videos': [], 'current_index': 0}

            os.makedirs(os.path.dirname(self.progress_file), exist_ok=True)
            with open(self.progress_file, 'w') as f:
                json.dump(self.progress, f, indent=2)
        except Exception as e:
            print(f"Error saving progress: {e}")

    def save_participant_info(self, study_id, session_id):
        """Save Prolific participant information"""
        participant_data = {
            'prolific_pid': self.prolific_pid,
            'study_id': study_id,
            'session_id': session_id,
            'assigned_videos': self.assigned_video_indices,
            'total_videos': len(self.assigned_video_indices),
            'started_at': datetime.now().isoformat()
        }
        
        with open(self.participant_info_file, 'w') as f:
            json.dump(participant_data, f, indent=2)

    # -------------------------------------------------------------------------
    # VIDEO AND GROUND TRUTH METHODS
    # -------------------------------------------------------------------------

    def get_youtube_embed_url(self, url):
        """Convert YouTube URL to embed format"""
        video_id_match = re.search(r'(?:v=|youtu\.be/|embed/)([^&\n?#]+)', url)
        if video_id_match:
            video_id = video_id_match.group(1)
            return f"https://www.youtube.com/embed/{video_id}"
        return url

    def get_ground_truth_labels(self, video_row):
        """Extract ground truth labels from metadata"""
        ground_truth = {'content': [], 'format': []}

        # Extract content ground truth with upper hierarchy
        for i in [1, 2]:
            content_label = video_row.get(f'content_map{i}')
            content_hierarchy = video_row.get(f'content_map_hierarchy{i}')
            content_upper_hierarchy = video_row.get(f'content_map_hierarchy_upper{i}')
            
            if pd.notna(content_label) and pd.notna(content_hierarchy) and pd.notna(content_upper_hierarchy):
                ground_truth['content'].append({
                    'label': content_label,
                    'hierarchy': content_hierarchy,
                    'upper_hierarchy': content_upper_hierarchy
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
        """Get the current video to annotate from assigned subset"""
        if not hasattr(self, 'progress') or self.progress is None:
            self.progress = {'annotated_videos': [], 'current_index': 0}

        if self.progress['current_index'] >= len(self.assigned_video_indices):
            return None

        # Get the actual video index from assigned subset
        actual_index = self.assigned_video_indices[self.progress['current_index']]
        video_data = self.metadata.iloc[actual_index].to_dict()

        video_data['embed_url'] = self.get_youtube_embed_url(video_data['url'])
        video_data['progress'] = {
            'current': self.progress['current_index'] + 1,
            'total': len(self.assigned_video_indices),
            'percentage': round((self.progress['current_index'] / len(self.assigned_video_indices)) * 100, 1)
        }

        # Add ground truth labels
        video_data['ground_truth'] = self.get_ground_truth_labels(video_data)

        return video_data

    def get_video_by_index(self, index):
        """Get a specific video by index from assigned subset"""
        if not hasattr(self, 'progress') or self.progress is None:
            self.progress = {'annotated_videos': [], 'current_index': 0}

        if 0 <= index < len(self.assigned_video_indices):
            actual_index = self.assigned_video_indices[index]
            video_data = self.metadata.iloc[actual_index].to_dict()

            video_data['embed_url'] = self.get_youtube_embed_url(video_data['url'])
            video_data['progress'] = {
                'current': index + 1,
                'total': len(self.assigned_video_indices),
                'percentage': round((index / len(self.assigned_video_indices)) * 100, 1)
            }

            video_data['ground_truth'] = self.get_ground_truth_labels(video_data)
            return video_data
        return None

    # -------------------------------------------------------------------------
    # ANNOTATION METHODS
    # -------------------------------------------------------------------------

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
                    latest_content = content_row.iloc[-1]
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
                    latest_format = format_row.iloc[-1]
                    existing_annotations['format'] = {
                        'label_1': latest_format.get('format_label_1', ''),
                        'hierarchy_1': latest_format.get('format_hierarchy_1', ''),
                        'label_2': latest_format.get('format_label_2', ''),
                        'hierarchy_2': latest_format.get('format_hierarchy_2', '')
                    }
        except Exception as e:
            print(f"Error loading existing annotations: {e}")

        return existing_annotations

    def save_annotations(self, video_id, content_selections, format_selections):
        """Save annotations with hierarchies and ground truth to TSV files - SAVES IMMEDIATELY"""
        timestamp = datetime.now().isoformat()

        if not hasattr(self, 'progress') or self.progress is None:
            self.progress = {'annotated_videos': [], 'current_index': 0}

        # Get video metadata
        video_row = None
        for idx, assigned_idx in enumerate(self.assigned_video_indices):
            if self.metadata.iloc[assigned_idx]['Video'] == video_id:
                video_row = self.metadata.iloc[assigned_idx]
                break
        
        if video_row is None:
            raise Exception(f"Video {video_id} not found in assigned videos")

        ground_truth = self.get_ground_truth_labels(video_row)
        already_annotated = video_id in self.progress['annotated_videos']

        # Prepare content annotation with upper hierarchy
        content_annotation = {
            'Video': video_id,
            'title': video_row['title'],
            'url': video_row['url'],
            'content_label_1': content_selections[0]['label'] if len(content_selections) > 0 else '',
            'content_hierarchy_1': content_selections[0]['hierarchy'] if len(content_selections) > 0 else '',
            'content_upper_hierarchy_1': content_selections[0].get('upper_hierarchy', '') if len(content_selections) > 0 else '',
            'content_label_2': content_selections[1]['label'] if len(content_selections) > 1 else '',
            'content_hierarchy_2': content_selections[1]['hierarchy'] if len(content_selections) > 1 else '',
            'content_upper_hierarchy_2': content_selections[1].get('upper_hierarchy', '') if len(content_selections) > 1 else '',
            'gt_content_label_1': ground_truth['content'][0]['label'] if len(ground_truth['content']) > 0 else '',
            'gt_content_hierarchy_1': ground_truth['content'][0]['hierarchy'] if len(ground_truth['content']) > 0 else '',
            'gt_content_upper_hierarchy_1': ground_truth['content'][0]['upper_hierarchy'] if len(ground_truth['content']) > 0 else '',
            'gt_content_label_2': ground_truth['content'][1]['label'] if len(ground_truth['content']) > 1 else '',
            'gt_content_hierarchy_2': ground_truth['content'][1]['hierarchy'] if len(ground_truth['content']) > 1 else '',
            'gt_content_upper_hierarchy_2': ground_truth['content'][1]['upper_hierarchy'] if len(ground_truth['content']) > 1 else '',
            'annotated_at': timestamp,
            'prolific_pid': self.prolific_pid
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
            'prolific_pid': self.prolific_pid
        }

        try:
            # CRITICAL: Save immediately to files to prevent data loss
            if already_annotated:
                # Update existing annotations by replacing old entries
                print(f"Updating existing annotation for video {video_id} by participant {self.prolific_pid}")
                
                # Read existing files
                content_df = pd.read_csv(self.content_annotations_file, sep='\t') if os.path.exists(self.content_annotations_file) else pd.DataFrame()
                format_df = pd.read_csv(self.format_annotations_file, sep='\t') if os.path.exists(self.format_annotations_file) else pd.DataFrame()

                # Remove existing entries for this video
                content_df = content_df[content_df['Video'] != video_id]
                format_df = format_df[format_df['Video'] != video_id]

                # Add new annotations
                content_df = pd.concat([content_df, pd.DataFrame([content_annotation])], ignore_index=True)
                format_df = pd.concat([format_df, pd.DataFrame([format_annotation])], ignore_index=True)

                # Save immediately with error handling
                content_df.to_csv(self.content_annotations_file, sep='\t', index=False)
                format_df.to_csv(self.format_annotations_file, sep='\t', index=False)
                
                print(f"Successfully updated annotations for video {video_id}")

            else:
                # Append new annotations - FIRST TIME ANNOTATION
                print(f"Saving new annotation for video {video_id} by participant {self.prolific_pid}")
                
                # Create headers if files don't exist
                if not os.path.exists(self.content_annotations_file):
                    content_df = pd.DataFrame([content_annotation])
                    content_df.to_csv(self.content_annotations_file, sep='\t', index=False)
                else:
                    # Append to existing file
                    pd.DataFrame([content_annotation]).to_csv(
                        self.content_annotations_file, sep='\t', mode='a', header=False, index=False
                    )
                
                if not os.path.exists(self.format_annotations_file):
                    format_df = pd.DataFrame([format_annotation])
                    format_df.to_csv(self.format_annotations_file, sep='\t', index=False)
                else:
                    # Append to existing file
                    pd.DataFrame([format_annotation]).to_csv(
                        self.format_annotations_file, sep='\t', mode='a', header=False, index=False
                    )

                # Update progress tracking
                self.progress['annotated_videos'].append(video_id)
                self.progress['last_updated'] = timestamp
                self.save_progress()
                
                print(f"Successfully saved new annotations for video {video_id}")

            # VERIFY FILES WERE SAVED CORRECTLY
            if os.path.exists(self.content_annotations_file) and os.path.exists(self.format_annotations_file):
                content_size = os.path.getsize(self.content_annotations_file)
                format_size = os.path.getsize(self.format_annotations_file)
                print(f"File verification: content_annotations.tsv = {content_size} bytes, format_annotations.tsv = {format_size} bytes")
                
                if content_size == 0 or format_size == 0:
                    raise Exception("Annotation files are empty after save!")
            else:
                raise Exception("Annotation files do not exist after save!")

            return True

        except Exception as e:
            print(f"CRITICAL ERROR: Failed to save annotations for video {video_id}: {e}")
            # Emergency backup save attempt
            try:
                backup_content_file = self.content_annotations_file + f".backup_{timestamp.replace(':', '-')}"
                backup_format_file = self.format_annotations_file + f".backup_{timestamp.replace(':', '-')}"
                
                pd.DataFrame([content_annotation]).to_csv(backup_content_file, sep='\t', index=False)
                pd.DataFrame([format_annotation]).to_csv(backup_format_file, sep='\t', index=False)
                
                print(f"Emergency backup saved to {backup_content_file} and {backup_format_file}")
            except Exception as backup_error:
                print(f"Emergency backup also failed: {backup_error}")
            
            raise e

# =============================================================================
# FLASK ROUTES
# =============================================================================

# Global dictionary to store annotation tools for different users
annotation_tools = {}

def get_annotation_tool(prolific_pid):
    """Get or create annotation tool for specific Prolific participant"""
    if prolific_pid not in annotation_tools:
        annotation_tools[prolific_pid] = VideoAnnotationTool(prolific_pid)
    return annotation_tools[prolific_pid]

def validate_prolific_pid(prolific_pid):
    """Validate Prolific PID format"""
    return re.match(r'^[a-zA-Z0-9_-]+$', prolific_pid) is not None

# -------------------------------------------------------------------------
# MAIN ROUTES
# -------------------------------------------------------------------------

    #          content_definitions=annotation_tool.content_definitions,
    #                          format_definitions=annotation_tool.format_definitions,
    #                          existing_annotations=existing_annotations,
    #                          prolific_pid=prolific_pid)
    
    # except Exception as e:
    #     print(f"Error initializing annotation tool: {e}")
    #     return render_template('error.html', 
    #                          error_message=f"Error initializing annotation system: {str(e)}"), 500

@app.route('/')
def training_landing():
    """Training landing page - shows examples and instructions"""
    prolific_pid = request.args.get('PROLIFIC_PID')
    study_id = request.args.get('STUDY_ID')
    session_id = request.args.get('SESSION_ID')
    
    print(f"DEBUG: Received params - PID: {prolific_pid}, Study: {study_id}, Session: {session_id}")
    
    if not prolific_pid or not study_id or not session_id:
        print("DEBUG: Missing parameters, redirecting to error")
        return render_template('error.html', 
                             error_message="Missing required Prolific parameters"), 400
    
    if not validate_prolific_pid(prolific_pid):
        print("DEBUG: Invalid PID format")
        return render_template('error.html', 
                             error_message="Invalid Prolific PID format"), 400
    
    print("DEBUG: All parameters valid, rendering training page")
    return render_template('training_landing.html',
                         prolific_pid=prolific_pid,
                         study_id=study_id,
                         session_id=session_id)

@app.route('/video/<int:index>')
def video_by_index(index):
    """Navigate to specific video by index"""
    prolific_pid = request.args.get('PROLIFIC_PID')
    
    if not prolific_pid or not validate_prolific_pid(prolific_pid):
        return render_template('error.html', 
                             error_message="Invalid or missing Prolific PID"), 400
    
    try:
        annotation_tool = get_annotation_tool(prolific_pid)
        video = annotation_tool.get_video_by_index(index)
        
        if video is None:
            return redirect(f"/?PROLIFIC_PID={prolific_pid}")
        
        # Update current index
        if not hasattr(annotation_tool, 'progress') or annotation_tool.progress is None:
            annotation_tool.progress = {'annotated_videos': [], 'current_index': 0}
        
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
                             existing_annotations=existing_annotations,
                             prolific_pid=prolific_pid)
    
    except Exception as e:
        print(f"Error getting video by index: {e}")
        return render_template('error.html', 
                             error_message=f"Error loading video: {str(e)}"), 500

# -------------------------------------------------------------------------
# API ROUTES
# -------------------------------------------------------------------------

@app.route('/submit', methods=['POST'])
def submit_annotation():
    """Handle annotation submission"""
    try:
        data = request.get_json()
        prolific_pid = data.get('prolific_pid')
        
        if not prolific_pid or not validate_prolific_pid(prolific_pid):
            return jsonify({'success': False, 'error': 'Invalid or missing Prolific PID'})
        
        annotation_tool = get_annotation_tool(prolific_pid)
        
        video_id = data.get('video_id')
        content_selections = data.get('content_selections', [])
        format_selections = data.get('format_selections', [])
        
        # Validation - now require exactly 2 selections each
        if len(content_selections) != 2:
            return jsonify({'success': False, 'error': 'Please select exactly 2 content labels'})
        
        if len(format_selections) != 2:
            return jsonify({'success': False, 'error': 'Please select exactly 2 format labels'})
        
        # Save annotations
        success = annotation_tool.save_annotations(video_id, content_selections, format_selections)
        
        if success:
            return jsonify({'success': True, 'message': 'Annotation saved successfully'})
        else:
            return jsonify({'success': False, 'error': 'Failed to save annotation'})
    
    except Exception as e:
        print(f"Error in submit_annotation: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/annotate')
def start_annotation():
    """Main annotation interface - after training"""
    prolific_pid = request.args.get('PROLIFIC_PID')
    study_id = request.args.get('STUDY_ID')
    session_id = request.args.get('SESSION_ID')
    
    print(f"DEBUG /annotate: Received params - PID: {prolific_pid}, Study: {study_id}, Session: {session_id}")
    
    if not prolific_pid or not study_id or not session_id:
        print("DEBUG /annotate: Missing parameters")
        return render_template('error.html', 
                             error_message="Missing required Prolific parameters"), 400
    
    if not validate_prolific_pid(prolific_pid):
        print("DEBUG /annotate: Invalid PID")
        return render_template('error.html', 
                             error_message="Invalid Prolific PID format"), 400
    
    print("DEBUG /annotate: Parameters valid, initializing annotation tool")
    
    try:
        annotation_tool = get_annotation_tool(prolific_pid)
        print("DEBUG /annotate: Annotation tool created")
        
        annotation_tool.save_participant_info(study_id, session_id)
        print("DEBUG /annotate: Participant info saved")
        
        video = annotation_tool.get_current_video()
        print(f"DEBUG /annotate: Got video: {video is not None}")
        
        if video is None:
            print("DEBUG /annotate: No video, redirecting to completed")
            return render_template('completed.html', prolific_pid=prolific_pid)
        
        # Get existing annotations for this video
        existing_annotations = annotation_tool.get_existing_annotations(video['Video'])
        print("DEBUG /annotate: Got existing annotations")
        
        print("DEBUG /annotate: Rendering annotation template")
        return render_template('annotate_with_definitions.html',
                             video=video,
                             content_hierarchies=annotation_tool.content_hierarchies,
                             format_hierarchies=annotation_tool.format_hierarchies,
                             content_definitions=annotation_tool.content_definitions,
                             format_definitions=annotation_tool.format_definitions,
                             existing_annotations=existing_annotations,
                             prolific_pid=prolific_pid,
                             study_id=study_id,
                             session_id=session_id)
    
    except Exception as e:
        print(f"DEBUG /annotate: Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        return render_template('error.html', 
                             error_message=f"Error initializing annotation system: {str(e)}"), 500
    
@app.route('/api/progress')
def get_progress():
    """Get current progress"""
    prolific_pid = request.args.get('PROLIFIC_PID')
    
    if not prolific_pid or not validate_prolific_pid(prolific_pid):
        return jsonify({'error': 'Invalid or missing Prolific PID'})
    
    try:
        annotation_tool = get_annotation_tool(prolific_pid)
        
        if not hasattr(annotation_tool, 'progress') or annotation_tool.progress is None:
            annotation_tool.progress = {'annotated_videos': [], 'current_index': 0}
        
        return jsonify({
            'current': annotation_tool.progress['current_index'] + 1,
            'total': len(annotation_tool.assigned_video_indices),
            'percentage': round((annotation_tool.progress['current_index'] / len(annotation_tool.assigned_video_indices)) * 100, 1)
        })
    
    except Exception as e:
        print(f"Error getting progress: {e}")
        return jsonify({'error': str(e)})

@app.route('/api/definition/<category>/<label>')
def get_definition(category, label):
    """Get definition for a specific label"""
    prolific_pid = request.args.get('PROLIFIC_PID')
    
    if not prolific_pid:
        return jsonify({'error': 'Missing Prolific PID'})
    
    try:
        annotation_tool = get_annotation_tool(prolific_pid)
        
        if category == 'content':
            definition_data = annotation_tool.content_definitions.get(label, {})
        elif category == 'format':
            definition_data = annotation_tool.format_definitions.get(label, {})
        else:
            return jsonify({'error': 'Invalid category'})
        
        return jsonify(definition_data)
    
    except Exception as e:
        return jsonify({'error': str(e)})

# -------------------------------------------------------------------------
# ADMIN ROUTES
# -------------------------------------------------------------------------

@app.route('/admin/annotations')
def view_all_annotations():
    """Admin route to view all annotations from all Prolific participants"""
    try:
        data_dir = './data/prolific'
        
        if not os.path.exists(data_dir):
            return jsonify({'error': 'No Prolific annotations directory found'})
        
        pid_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        
        annotations_summary = {
            'content_files': [],
            'format_files': [],
            'total_annotations': 0,
            'total_participants': len(pid_dirs)
        }
        
        for prolific_pid in pid_dirs:
            pid_dir = os.path.join(data_dir, prolific_pid)
            content_file = os.path.join(pid_dir, 'content_annotations.tsv')
            format_file = os.path.join(pid_dir, 'format_annotations.tsv')
            
            # Process content file
            if os.path.exists(content_file):
                try:
                    df = pd.read_csv(content_file, sep='\t')
                    annotations_summary['content_files'].append({
                        'prolific_pid': prolific_pid,
                        'filename': 'content_annotations.tsv',
                        'count': len(df)
                    })
                    annotations_summary['total_annotations'] += len(df)
                except Exception as e:
                    annotations_summary['content_files'].append({
                        'prolific_pid': prolific_pid,
                        'filename': 'content_annotations.tsv',
                        'count': 0,
                        'error': str(e)
                    })
            
            # Process format file
            if os.path.exists(format_file):
                try:
                    df = pd.read_csv(format_file, sep='\t')
                    annotations_summary['format_files'].append({
                        'prolific_pid': prolific_pid,
                        'filename': 'format_annotations.tsv',
                        'count': len(df)
                    })
                except Exception as e:
                    annotations_summary['format_files'].append({
                        'prolific_pid': prolific_pid,
                        'filename': 'format_annotations.tsv',
                        'count': 0,
                        'error': str(e)
                    })
        
        return jsonify(annotations_summary)
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/admin/video_assignments')
def view_video_assignments():
    """Admin route to view video assignments"""
    try:
        assignment_file = './data/video_assignments.json'
        
        if not os.path.exists(assignment_file):
            return jsonify({'error': 'No video assignments file found'})
        
        with open(assignment_file, 'r') as f:
            assignment_data = json.load(f)
        
        return jsonify(assignment_data)
    
    except Exception as e:
        return jsonify({'error': str(e)})

# Add this route to your Flask app for local testing
@app.route('/test')
def test_local():
    """Local test route without Prolific requirements"""
    return "Flask app is running! This is a test page."

# =============================================================================
# ERROR HANDLING
# =============================================================================

@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', 
                         error_message="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', 
                         error_message="Internal server error"), 500

@app.route('/training_test')
def training_test():
    """Test training page locally"""
    return render_template('training_landing.html',
                         prolific_pid='TEST123',
                         study_id='TEST_STUDY',
                         session_id='TEST_SESSION')
# =============================================================================
# APPLICATION STARTUP
# =============================================================================

if __name__ == '__main__':
    # Create data directory if it doesn't exist
    os.makedirs('./data', exist_ok=True)
    os.makedirs('./data/prolific', exist_ok=True)
    
    # Initialize assignment manager
    print("Starting Video Annotation Tool with Prolific Integration...")
    print("Main URL: https://contentxformat.pythonanywhere.com/?PROLIFIC_PID={{%PROLIFIC_PID%}}&STUDY_ID={{%STUDY_ID%}}&SESSION_ID={{%SESSION_ID%}}")
    print("Admin dashboard: https://contentxformat.pythonanywhere.com/admin/annotations")
    print("Video assignments: https://contentxformat.pythonanywhere.com/admin/video_assignments")
    
    app.run(debug=True, host='0.0.0.0', port=5000)