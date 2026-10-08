#!/usr/bin/env bash
#SBATCH --job-name=hold-node
#SBATCH --partition=normal
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/work/jvl210002/migration/node/%j.out
#
# Holds one compute node so you can SSH in (VS Code Remote-SSH or a terminal) and run
# the Claude chats there. The node name goes to node/current_node.
# Start:  sbatch /work/jvl210002/migration/node/hold.sbatch
# Stop:   scancel -n hold-node
hostname > /work/jvl210002/migration/node/current_node
# AUTO-RENEW (2026-09-24): queue the successor now so the node, and the feeder it hosts, outlive
# this job's 2-day limit. Only when no hold-node is already pending, so a requeue or a second
# copy never stacks successors. Stop the chain: scancel -n hold-node (cancels both).
squeue -h -u "$USER" -n hold-node -t PENDING | grep -q . || \
  sbatch --dependency=afterany:"$SLURM_JOB_ID" /work/jvl210002/migration/node/hold.sbatch
echo "holding $(hostname) as job $SLURM_JOB_ID until $(date -d '+2 days')"
# Unattended experiment feeder for bidirectional (runs/feeder/feeder.sh). A lock file makes a
# second copy exit, so this is safe on requeue. Remove this line to hold the node without it.
( cd /work/jvl210002/migration/bidirectional && setsid nohup bash runs/feeder/feeder.sh >/dev/null 2>&1 < /dev/null & )
sleep infinity
