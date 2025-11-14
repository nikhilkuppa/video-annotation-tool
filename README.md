# video-annotation-tool

- To create labels and definitions, copy the content labels to the new_content_labels.tsv and similarly with format.
- copy the definitions to content_definitions.tsv and similarly with format.
- run map_labels_to_metadata.py to create a metadata table with columns title	url	duration	Video	content_map1	content_map_hierarchy1	content_map_hierarchy_upper1	format_map1	format_map_hierarchy1	content_map2	content_map_hierarchy2	content_map_hierarchy_upper2	format_map2	format_map_hierarchy2
- run cells on test.ipynb to make dataset for prolific
- when running app, append URL parameters /?PROLIFIC_PID=TEST_USER_001&STUDY_ID=CONTENT_FORMAT_STUDY_2024&SESSION_ID=SESSION_001