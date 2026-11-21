# service-local-mapping-rbnx deployment configuration.
#
# rbnx delivers the deployment entry's `config:` mapping through
# Driver(CMD_INIT). The bridge validates it there. This service is a pure
# sensor consumer: the inputs listed under sensor_providers are the only
# sources it may subscribe to.

specVersion: 1
description: >-
  Rolling 2.5D local map built online from lidar, depth and optional RGB
  semantics. It publishes a Nav2-style cost grid in the odometry frame. The
  robot must also have a global map or waypoint source; this service only
  solves local obstacle avoidance and traversability.

properties:
  sensor_providers:
    type: object
    x-group: Inputs
    description: >-
      Sensor role to Atlas provider id (deployment entry name). Only listed
      roles are used. scan_rings is preferred for outdoor operation; if it is
      absent the mapper falls back to scan_low/scan_high, then scan. At least
      one lidar or depth role must be bound. odom is required. Unknown roles
      fail initialization.
    properties:
      scan_rings:
        type: string
        x-provider: robonix/primitive/lidar/lidar_rings
        description: PointCloud2 with per-ring z already expressed in its frame.
      scan_low:
        type: string
        x-provider: robonix/primitive/lidar/lidar_low
        description: Optional low-ring LaserScan, used for short obstacles.
      scan_high:
        type: string
        x-provider: robonix/primitive/lidar/lidar_high
        description: Optional high-ring LaserScan, used for walls and trunks.
      scan:
        type: string
        x-provider: robonix/primitive/lidar/lidar
        description: Legacy single merged LaserScan fallback.
      depth:
        type: string
        x-provider: robonix/primitive/camera/depth
        description: 32FC1 depth image in metres.
      rgb:
        type: string
        x-provider: robonix/primitive/camera/rgb
        description: rgb8 image, used only by the optional semantic model.
      intrinsics:
        type: string
        x-provider: robonix/primitive/camera/intrinsics
        description: CameraInfo matching the depth image.
      extrinsics:
        type: string
        x-provider: robonix/primitive/camera/extrinsics
        description: Static calibration fallback when TF is incomplete.
      odom:
        type: string
        x-provider: robonix/primitive/chassis/odom
        description: Robot odometry; used as proprioception only.

  map_frame:
    type: string
    default: minecraft_odom
    x-group: Frames and time
    description: Frame of the rolling grid and all outputs. Must be the odometry frame.

  base_frame:
    type: string
    default: minecraft_view_base
    x-group: Frames and time
    description: Robot body frame, used for height reference and TF lookups.

  odom_frame:
    type: string
    default: minecraft_odom
    x-group: Frames and time
    description: Frame in which odometry is expressed; normally equal to map_frame.

  base_height_m:
    type: number
    minimum: 0
    maximum: 5
    default: 1.62
    x-group: Frames and time
    description: >-
      Height of the base frame origin above the robot contact plane. The local
      ground reference is base z minus this value.

  use_sim_time:
    type: boolean
    default: true
    x-group: Frames and time
    description: Use /clock for ROS message timestamps and TF.

  allowed_input_frames:
    type: array
    items:
      type: string
    x-group: Frames and time
    description: >-
      Additional TF frames accepted from sensor messages. map_frame, base_frame
      and odom_frame are always accepted. A sensor frame that is absent both
      here and on TF is dropped and counted in health.

  grid_size_m:
    type: number
    minimum: 16
    maximum: 128
    default: 64
    x-group: Grid
    description: Internal square rolling grid edge length in metres.

  publish_size_m:
    type: number
    minimum: 8
    maximum: 64
    default: 32
    x-group: Grid
    description: Published centered window edge length in metres. Must not exceed grid_size_m.

  resolution_m:
    type: number
    minimum: 0.05
    maximum: 0.5
    default: 0.2
    x-group: Grid
    description: Cell edge length in metres.

  update_rate_hz:
    type: number
    minimum: 1
    maximum: 30
    default: 10
    x-group: Grid
    description: Grid integration rate for cached sensor frames.

  publish_rate_hz:
    type: number
    minimum: 1
    maximum: 30
    default: 10
    x-group: Grid
    description: Cost/occupancy/debug publication rate.

  tf_timeout_s:
    type: number
    minimum: 0.05
    maximum: 2
    default: 0.25
    x-group: Frames and time
    description: Maximum TF wait for a sensor frame.

  max_tf_age_s:
    type: number
    minimum: 0.05
    maximum: 2
    default: 0.5
    x-group: Frames and time
    description: Frames whose TF is older than this are dropped and counted.

  depth_stride:
    type: integer
    minimum: 1
    maximum: 16
    default: 4
    x-group: Depth
    description: Pixel step in both directions during depth back-projection.

  depth_min_range_m:
    type: number
    minimum: 0.05
    default: 0.25
    x-group: Depth
    description: Nearer depth readings are ignored.

  depth_max_range_m:
    type: number
    minimum: 1
    maximum: 64
    default: 12
    x-group: Depth
    description: Farther or non-finite depth readings are ignored.

  depth_clearing:
    type: boolean
    default: false
    x-group: Depth
    description: >-
      Ray-clear free space along depth points. Off by default because the
      lidar owns free-space clearing and depth rays are dense.

  lidar_min_range_m:
    type: number
    minimum: 0
    default: 0.3
    x-group: Lidar
    description: Returns closer than this are ignored as self-hits.

  lidar_max_range_m:
    type: number
    minimum: 2
    maximum: 64
    default: 32
    x-group: Lidar
    description: Returns beyond this are ignored.

  ground_percentile:
    type: number
    minimum: 1
    maximum: 50
    default: 10
    x-group: Height model
    description: Per-cell low percentile used as ground height in depth clusters.

  step_height_m:
    type: number
    minimum: 0.05
    maximum: 2
    default: 0.55
    x-group: Height model
    description: Vertical separation between ground and an obstacle.

  slope_max_deg:
    type: number
    minimum: 5
    maximum: 60
    default: 35
    x-group: Height model
    description: Neighbouring-cell height change above this is high cost.

  obstacle_min_height_m:
    type: number
    minimum: 0
    maximum: 1
    default: 0.05
    x-group: Height model
    description: Points above local ground by at least this are obstacles.

  obstacle_max_height_m:
    type: number
    minimum: 0.2
    maximum: 6
    default: 2.0
    x-group: Height model
    description: Obstacles taller than this do not enter the collision layer.

  log_odds_free:
    type: number
    minimum: 0.05
    maximum: 2
    default: 0.4
    x-group: Occupancy
    description: Log-odds magnitude subtracted from cells crossed by a ray.

  log_odds_occupied:
    type: number
    minimum: 0.05
    maximum: 3
    default: 0.85
    x-group: Occupancy
    description: Log-odds magnitude added to a cell hit by a return.

  log_odds_min:
    type: number
    default: -5
    x-group: Occupancy
    description: Lower clamp of accumulated log odds.

  log_odds_max:
    type: number
    default: 5
    x-group: Occupancy
    description: Upper clamp of accumulated log odds.

  decay_seconds:
    type: number
    minimum: 0
    maximum: 120
    default: 8
    x-group: Occupancy
    description: Age after which occupied evidence decays toward free; 0 disables decay.

  observation_radius_m:
    type: number
    minimum: 2
    maximum: 64
    default: 15
    x-group: Occupancy
    description: >-
      Active observation radius around the robot in metres. Previously occupied
      cells outside this radius (or occluded beyond decay_seconds without being
      ray-cleared) are kept as stale obstacles when stale_obstacle_cost > 0.

  stale_obstacle_cost:
    type: integer
    minimum: 0
    maximum: 99
    default: 0
    x-group: Occupancy
    description: >-
      Cost assigned to remembered non-traversable cells that have left the
      active observation radius or decayed without ray-clearing. Re-observing
      them as free clears the stale state. 0 disables stale obstacle memory.

  wall_aversion_radius_m:
    type: number
    minimum: 0
    maximum: 5
    default: 0
    x-group: Occupancy
    description: >-
      Repulsion radius around lethal obstacles in metres to discourage
      wall-hugging paths; 0 disables wall aversion.

  wall_aversion_max_cost:
    type: integer
    minimum: 0
    maximum: 95
    default: 65
    x-group: Occupancy
    description: >-
      Peak soft cost immediately adjacent to lethal obstacles when
      wall_aversion_radius_m > 0. Must stay below lethal (100).

  semantic_model_path:
    type: string
    x-group: Semantics
    description: >-
      Absolute path or path relative to the package root to an ONNX segmentation
      model with 7 output classes. Empty or missing disables semantics and
      leaves the geometry pipeline unchanged.

  semantic_width:
    type: integer
    minimum: 64
    maximum: 1280
    default: 320
    x-group: Semantics
    description: Model input width.

  semantic_height:
    type: integer
    minimum: 64
    maximum: 720
    default: 192
    x-group: Semantics
    description: Model input height.

  semantic_rate_hz:
    type: number
    minimum: 0.5
    maximum: 15
    default: 5
    x-group: Semantics
    description: Maximum semantic inference rate.

  semantic_conf_threshold:
    type: number
    minimum: 0
    maximum: 1
    default: 0.55
    x-group: Semantics
    description: Pixel confidence below this is treated as unknown.

  hazard_conf_threshold:
    type: number
    minimum: 0
    maximum: 1
    default: 0.75
    x-group: Semantics
    description: Confidence required before a hazard class may raise a cell to lethal.

  semantic_ema_frames:
    type: integer
    minimum: 1
    maximum: 20
    default: 3
    x-group: Semantics
    description: Number of consecutive frames blended into the semantic cost EMA.

  rgbd_max_skew_s:
    type: number
    minimum: 0
    maximum: 1
    default: 0.08
    x-group: Semantics
    description: Maximum RGB/depth timestamp difference for semantic projection.

  cost_grid_topic:
    type: string
    default: /robonix/local_mapping/cost_grid
    x-group: Outputs
    description: Absolute topic for the nav_msgs/OccupancyGrid cost output.

  occupancy_topic:
    type: string
    default: /robonix/local_mapping/occupancy
    x-group: Outputs
    description: Absolute topic for the debug occupancy grid.

  obstacles_topic:
    type: string
    default: /robonix/local_mapping/obstacles
    x-group: Outputs
    description: Absolute topic for the debug obstacle PointCloud2.

  health_topic:
    type: string
    default: /robonix/local_mapping/health
    x-group: Outputs
    description: Absolute topic for JSON health strings.

  health_rate_hz:
    type: number
    minimum: 0.1
    maximum: 5
    default: 1
    x-group: Outputs
    description: Health publication rate.

  webui_enabled:
    type: boolean
    default: true
    x-group: Web UI
    description: Serve a read-only local map inspection page over HTTP.

  webui_host:
    type: string
    default: 127.0.0.1
    x-group: Web UI
    description: Bind address for the read-only local map Web UI.

  webui_port:
    type: integer
    minimum: 0
    maximum: 65535
    default: 8092
    x-group: Web UI
    description: Bind port for the read-only local map Web UI; 0 disables it.

  visual_object_url:
    type: string
    default: http://127.0.0.1:8094
    x-group: Web UI
    description: Optional loopback API URL of service-visual-object-rbnx merged into the 8092 Web UI.

required: [sensor_providers]