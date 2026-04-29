#!/usr/bin/env bash
# Hosts where neither /usr/share/vulkan/icd.d/nvidia_icd.json nor
# /etc/vulkan/icd.d/nvidia_icd.json exists — SAPIEN cannot render correctly
# without admin intervention. Email action@cs.stanford.edu to install ICD.
# Update after running probe_node.sh on any new host.
# Survey date: 2026-04-28
#
# NOTE: iris9 (L40S, driver 580.82) is EXCLUDED because it passed the fix
# (VK_ICD_FILENAMES=/etc/... → CLEAN), so it is NOT actually bad.
# Only truly unreachable or unfixable nodes go here.
export BAD_VULKAN_NODES="iris-hp-z8,iris-hgx-1"
