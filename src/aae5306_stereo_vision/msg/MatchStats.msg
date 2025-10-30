# Feature Matching Statistics Message
Header header

# Matcher information
string detector_type           # sift, orb

# Matching results
int32 num_initial_matches      # Initial matches from matcher
int32 num_ratio_filtered       # After ratio test
int32 num_epipolar_filtered    # After epipolar constraint
int32 num_final_matches        # Final valid matches

# Timing
float32 detection_time         # Feature detection time (ms)
float32 matching_time          # Matching time (ms)
float32 filtering_time         # Filtering time (ms)
float32 total_time             # Total processing time (ms)

# Parameters used
float32 ratio_threshold
float32 epipolar_threshold
