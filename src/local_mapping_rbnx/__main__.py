# SPDX-License-Identifier: MulanPSL-2.0
"""Entry point for the local mapping service."""

from local_mapping_rbnx.atlas_bridge import local_map

if __name__ == "__main__":
    local_map.run()