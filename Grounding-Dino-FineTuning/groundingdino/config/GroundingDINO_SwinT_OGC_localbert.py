# Same as SwinT OGC but uses local BERT (offline training on AutoDL).
from groundingdino.config.GroundingDINO_SwinT_OGC import *  # noqa: F401,F403

text_encoder_type = "/root/autodl-tmp/Grounded-SAM-2/bert-base-uncased"
