"""Our own scorer (rule 7: never uses the submission kit's models or outputs).

DINOv2-S/14 and R3D-18 are the same public weights the leaderboard uses; the action term uses
an inverse dynamics model we train on the train split only.
"""
