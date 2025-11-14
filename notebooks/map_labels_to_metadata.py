import pandas as pd

content_map = pd.read_csv(r'D:\Users\Neuro\City College Dropbox\NIKHIL KUPPA\youtube_video_tagging\youtube_project\verify_annotations_tool\notebooks\new_content_labels.tsv', sep='\t')
content_map['tag'] = content_map['tag'].str.rstrip(':')
content_map['mapAnnotation'] = content_map['mapAnnotation'].str.rstrip(':')
content_map['mapAnnotationHierarchy'] = content_map['mapAnnotationHierarchy'].str.rstrip(':')
content_map['Hierarchy1'] = content_map['Hierarchy1'].str.rstrip(':')

content_map_annotate = content_map.set_index('tag')['mapAnnotation'].to_dict()
content_map_hierarchy = content_map.set_index('tag')['mapAnnotationHierarchy'].to_dict()
content_map_hierarchy_upper = content_map.set_index('tag')['Hierarchy1'].to_dict()

content_df = pd.read_csv(r"D:\Users\Neuro\City College Dropbox\NIKHIL KUPPA\youtube_video_tagging\_deprecated\results\transcript\transcript_classifications_r005_v3.tsv", sep='\t')

content_df_meta = pd.DataFrame()
content_df_meta['Video'] = content_df['Video'].str.removesuffix('_classification')
content_df_meta['content1'] = content_df.iloc[:, 2:].idxmax(axis=1)
content_df_meta['content_map1'] = content_df_meta['content1'].map(content_map_annotate)
content_df_meta['content_map_hierarchy1'] = content_df_meta['content1'].map(content_map_hierarchy)
content_df_meta['content_map_hierarchy_upper1'] = content_df_meta['content1'].map(content_map_hierarchy_upper)


content_df_meta['content2'] = content_df.iloc[:, 2:].apply(lambda row: row.nlargest(2).index[1], axis=1)
content_df_meta['content_map2'] = content_df_meta['content2'].map(content_map_annotate)
content_df_meta['content_map_hierarchy2'] = content_df_meta['content2'].map(content_map_hierarchy)
content_df_meta['content_map_hierarchy_upper2'] = content_df_meta['content2'].map(content_map_hierarchy_upper)

format_map = pd.read_csv(r'D:\Users\Neuro\City College Dropbox\NIKHIL KUPPA\youtube_video_tagging\youtube_project\verify_annotations_tool\notebooks\new_format_labels.tsv', sep='\t')
format_mapping_annotate = format_map.set_index('tag')['mapAnnotation'].to_dict()
format_mapping_hierarchy = format_map.set_index('tag')['mapAnnotationHierarchy'].to_dict()

format_df = pd.read_csv(r"D:\Users\Neuro\City College Dropbox\NIKHIL KUPPA\Nikhil Kuppa\youtube_video_tagging\video_bucket\contruct_classifications_df\df_video_fixed_redo_edu_v4_thought_all.tsv", sep='\t')

format_numeric = format_df.iloc[:, 2:].apply(pd.to_numeric, errors='coerce').fillna(0)
format_df_meta = pd.DataFrame()
format_df_meta['Video'] = format_df['Video'].str.removesuffix('_format_classification')
format_df_meta['format1'] = format_numeric.idxmax(axis=1)
format_df_meta['format_map1'] = format_df_meta['format1'].map(format_mapping_annotate)
format_df_meta['format_map_hierarchy1'] = format_df_meta['format1'].map(format_mapping_hierarchy)
format_df_meta['format2'] = format_numeric.apply(lambda row: row.nlargest(2).index[1] if row.count() >= 2 else None, axis=1)
format_df_meta['format_map2'] = format_df_meta['format2'].map(format_mapping_annotate)
format_df_meta['format_map_hierarchy2'] = format_df_meta['format2'].map(format_mapping_hierarchy)

video_classifications = pd.merge(content_df_meta, format_df_meta, on='Video')
video_classifications = video_classifications[['Video', 'content1', 'content_map1', 'content_map_hierarchy1', 'content_map_hierarchy_upper1', 'format1', 'format_map1', 'format_map_hierarchy1',
                                                'content2', 'content_map2',  'content_map_hierarchy2', 'content_map_hierarchy_upper2', 'format2', 'format_map2', 'format_map_hierarchy2']]

METADATA = pd.read_excel(r"D:\Users\Neuro\City College Dropbox\NIKHIL KUPPA\youtube_video_stats\data\T_text_information.xlsx", engine='openpyxl')
METADATA['Video'] = METADATA['channelName'] + '_' + METADATA['id']
video_classifications_metadata = pd.merge(METADATA, video_classifications, on='Video', how='inner')
video_classifications = video_classifications[~video_classifications['format_map_hierarchy1'].isin(['Others'])]

mapped_cols = [
    'content_map1', 'content_map_hierarchy1', 'content_map_hierarchy_upper1',
    'format_map1', 'format_map_hierarchy1',
    'content_map2', 'content_map_hierarchy2', 'content_map_hierarchy_upper2',
    'format_map2', 'format_map_hierarchy2'
]

video_classifications = video_classifications.dropna(subset=mapped_cols)
video_classifications_metadata = video_classifications_metadata.dropna(subset=mapped_cols)

video_classifications.reset_index(drop=True, inplace=True)
video_classifications_metadata.reset_index(drop=True, inplace=True)

columns_metadata = ['title',	'url',	'duration',	'Video', 'content_map1', 'content_map_hierarchy1', 'content_map_hierarchy_upper1',
    'format_map1', 'format_map_hierarchy1',
    'content_map2', 'content_map_hierarchy2', 'content_map_hierarchy_upper2',
    'format_map2', 'format_map_hierarchy2']

video_classifications_metadata[columns_metadata].to_csv(r'D:\Users\Neuro\City College Dropbox\NIKHIL KUPPA\youtube_video_tagging\youtube_project\verify_annotations_tool\notebooks\video_classifications_metadata.tsv', sep='\t', index=False)