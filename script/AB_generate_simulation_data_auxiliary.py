"""
AB_generate_simulation_data_auxiliary.py
=========================================
Abaqus Python auxiliary for CAE modification and FEA job submission.
Runs inside Abaqus CAE; called by AB_generate_simulation_data.py.
"""

from abaqus import *
from abaqusConstants import *
from caeModules import *

import os
import sys
import time

# ========================================
# Parse Command Line Arguments
# ========================================
# Expected arguments (13 total after '--'):
# cae_path, output_dir, h1, h2, h3, draft, mx1, my1, mz1, mx2, my2, mz2, iteration_num

if len(sys.argv) < 14:
    print("Error: Insufficient arguments")
    print("Usage: abaqus cae noGUI=script.py -- <cae_path> <output_dir> <h1> <h2> <h3> <draft> <mx1> <my1> <mz1> <mx2> <my2> <mz2> <iteration_num>")
    sys.exit(1)

cae_path = sys.argv[-13]
output_dir = sys.argv[-12]
h1 = float(sys.argv[-11])
h2 = float(sys.argv[-10])
h3 = float(sys.argv[-9])
draft = float(sys.argv[-8])
mx1 = float(sys.argv[-7])
my1 = float(sys.argv[-6])
mz1 = float(sys.argv[-5])
mx2 = float(sys.argv[-4])
my2 = float(sys.argv[-3])
mz2 = float(sys.argv[-2])
iteration_num = int(sys.argv[-1])

print("=" * 60)
print("AB CAE Modification - Iteration %d" % iteration_num)
print("=" * 60)
print("CAE file: %s" % cae_path)
print("Output dir: %s" % output_dir)
print("Load parameters:")
print("  h1=%.2f, h2=%.2f, h3=%.2f, draft=%.2f" % (h1, h2, h3, draft))
print("  mx1=%.2e, my1=%.2e, mz1=%.2e" % (mx1, my1, mz1))
print("  mx2=%.2e, my2=%.2e, mz2=%.2e" % (mx2, my2, mz2))
print("=" * 60)

# Change to output directory BEFORE starting CAE session
# This ensures .rpy files are generated in the correct location
original_dir = os.getcwd()
os.chdir(output_dir)
print("Changed working directory to: %s" % output_dir)

try:
    t_start = time.time()

    # ========================================
    # 1. Open CAE File
    # ========================================
    print("[Step 1/5] Opening CAE file...")
    t0 = time.time()

    if not os.path.exists(cae_path):
        print("ERROR: CAE file not found: %s" % cae_path)
        os.chdir(original_dir)
        sys.exit(1)

    openMdb(pathName=cae_path)
    model = mdb.models["Model-1"]
    print("  CAE opened successfully (%.2f s)" % (time.time() - t0))

    # ========================================
    # 2. Update Analytical Fields (fields already exist in model_1214.cae)
    # ========================================
    print("[Step 2/5] Updating analytical fields...")
    t0 = time.time()

    # Left cargo pressure field (field name: "left")
    exp_left = "3e-12*9810*((%s-Z)+abs(%s-Z))/2" % (str(h1), str(h1))
    model.analyticalFields["left"].setValues(expression=exp_left)
    print("  Updated 'left' field")

    # Mid cargo pressure field (field name: "mid")
    exp_mid = "3e-12*9810*((%s-Z)+abs(%s-Z))/2" % (str(h2), str(h2))
    model.analyticalFields["mid"].setValues(expression=exp_mid)
    print("  Updated 'mid' field")

    # Right cargo pressure field (field name: "right")
    exp_right = "3e-12*9810*((%s-Z)+abs(%s-Z))/2" % (str(h3), str(h3))
    model.analyticalFields["right"].setValues(expression=exp_right)
    print("  Updated 'right' field")

    # Water pressure field (field name: "water")
    exp_water = "1.025e-12*9810*((%s-Z)+abs(%s-Z))/2" % (str(draft), str(draft))
    model.analyticalFields["water"].setValues(expression=exp_water)
    print("  Updated 'water' field")

    print("  Analytical fields updated (%.2f s)" % (time.time() - t0))

    # ========================================
    # 3. Associate Loads with Updated Fields
    # ========================================
    print("[Step 3/5] Associating loads with updated fields...")
    t0 = time.time()

    # Associate pressure loads with analytical fields
    try:
        model.loads["leftload"].setValues(field="left")
        print("  leftload -> left")
    except KeyError:
        print("  WARNING: Load 'leftload' not found, skipping")

    try:
        model.loads["midload"].setValues(field="mid")
        print("  midload -> mid")
    except KeyError:
        print("  WARNING: Load 'midload' not found, skipping")

    try:
        model.loads["rightload"].setValues(field="right")
        print("  rightload -> right")
    except KeyError:
        print("  WARNING: Load 'rightload' not found, skipping")

    try:
        model.loads["water"].setValues(field="water")
        print("  water -> water")
    except KeyError:
        print("  WARNING: Load 'water' not found, skipping")

    # Modify moment loads
    try:
        model.loads["mom1"].setValues(
            cm1=mx1, cm2=my1, cm3=mz1,
            distributionType=UNIFORM, field=""
        )
        print("  mom1 updated (mx1=%.2e, my1=%.2e, mz1=%.2e)" % (mx1, my1, mz1))
    except KeyError:
        print("  WARNING: Load 'mom1' not found, skipping")

    try:
        model.loads["mom2"].setValues(
            cm1=mx2, cm2=my2, cm3=mz2,
            distributionType=UNIFORM, field=""
        )
        print("  mom2 updated (mx2=%.2e, my2=%.2e, mz2=%.2e)" % (mx2, my2, mz2))
    except KeyError:
        print("  WARNING: Load 'mom2' not found, skipping")

    print("  Loads modified (%.2f s)" % (time.time() - t0))

    # ========================================
    # 4. Clean Stale Lock Files
    # ========================================
    print("[Step 4/5] Cleaning stale lock files...")
    t0 = time.time()

    try:
        lock_file = os.path.join(output_dir, "iteration.lck")
        if os.path.exists(lock_file):
            os.remove(lock_file)
            print("  Removed stale lock file")
    except Exception as e:
        print("  Warning: Could not remove lock file: %s" % str(e))

    print("  Lock cleanup done (%.2f s)" % (time.time() - t0))

    # ========================================
    # 5. Submit Job
    # ========================================
    print("[Step 5/5] Submitting Abaqus job...")
    t0 = time.time()

    # Create job if not exists
    job_name = "iteration"
    if job_name in mdb.jobs:
        del mdb.jobs[job_name]

    job = mdb.Job(
        name=job_name,
        model="Model-1",
        type=ANALYSIS,
        numCpus=1,
        memory=90,
        memoryUnits=PERCENTAGE
    )

    # Save modified CAE file before submitting job
    print("  Saving modified CAE file...")
    mdb.save()
    print("  CAE file saved successfully")

    # Already in output directory, no need to change again
    print("  Job created, submitting...")
    job.submit(consistencyChecking=OFF)

    print("  Waiting for completion...")
    job.waitForCompletion()

    # Check job status
    if job.status == COMPLETED:
        print("  Job completed successfully (%.2f s)" % (time.time() - t0))
    else:
        print("  ERROR: Job failed with status: %s" % str(job.status))
        # Restore directory and exit with error
        os.chdir(original_dir)
        sys.exit(1)

    # ========================================
    # Summary
    # ========================================
    total_time = time.time() - t_start
    print("=" * 60)
    print("Iteration %d completed successfully" % iteration_num)
    print("Total time: %.2f seconds" % total_time)
    print("=" * 60)

    # Restore directory and exit successfully
    os.chdir(original_dir)
    sys.exit(0)

except Exception as e:
    print("=" * 60)
    print("ERROR: Iteration %d failed" % iteration_num)
    print("Exception: %s" % str(e))
    print("=" * 60)

    # Try to save error log
    try:
        error_log = os.path.join(output_dir, "abaqus_error.log")
        with open(error_log, 'w') as f:
            f.write("Iteration %d failed\n" % iteration_num)
            f.write("Error: %s\n" % str(e))
            import traceback
            f.write("Traceback:\n%s\n" % traceback.format_exc())
    except:
        pass

    # Restore directory before exiting
    os.chdir(original_dir)
    sys.exit(1)
