"""
SO101 VLA Demo

This package contains a minimal scaffold to:
- connect to a SO101 follower arm via LeRobot
- stream camera + joint observations
- wrap learned policies as "skills" (search_object, grasp_object)
- orchestrate a simple search-then-grasp demo

It is intentionally lightweight and model-agnostic. You will:
- record datasets with `lerobot-record`
- train policies with `lerobot-train` on your AMD VM
- then point the skill loaders at your local checkpoints.
"""

