# Self-contained evaluation helpers used by the ops-knowledge-optimization-ingest skill.
# These modules run a LOCAL msprof collection on a real Ascend NPU (build the
# operator working copy, wrap its run step with msprof, parse the artifacts) and
# depend only on the Python standard library + the external msprof / npu-smi
# binaries. See the project README ("Local dependency: NPU + CANN + msprof").
