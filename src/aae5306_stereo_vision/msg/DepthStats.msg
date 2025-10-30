# Depth Recovery Statistics Message
Header header

# Point cloud information
int32 num_points              # Number of 3D points
int32 num_valid_disparities   # Number of valid disparities

# Depth statistics
float32 mean_depth            # Mean depth in meters
float32 min_depth             # Minimum depth in meters
float32 max_depth             # Maximum depth in meters
float32 std_depth             # Standard deviation of depth

# Disparity statistics
float32 mean_disparity        # Mean disparity in pixels
float32 min_disparity         # Minimum disparity in pixels
float32 max_disparity         # Maximum disparity in pixels

# Camera parameters used
float32 focal_length          # Focal length in pixels
float32 baseline              # Baseline in meters

# Processing time
float32 computation_time      # Depth computation time (ms)
