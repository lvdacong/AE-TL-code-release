"""
AA_create_damage_cae_auxiliary.py
==================================
Abaqus Python auxiliary for batch damage model creation.
Runs inside Abaqus CAE; imported by AA_create_damage_cae.py.
"""

import sys
import os
import json
from abaqus import *
from abaqusConstants import *

# ========================================
# Parse Command Line Arguments
# ========================================

# When running with Abaqus: abaqus cae noGUI=script.py -- adjacency template_cae output_base config_file
# sys.argv contains: ['script.py', ..., '--', 'adjacency', 'template_cae', 'output_base', 'config_file']

# Find 'adjacency' keyword in arguments
method = None
user_args = []

for i, arg in enumerate(sys.argv):
    if arg == 'adjacency':
        method = arg
        # User args start from the method name
        user_args = sys.argv[i:]
        break

if method is None:
    print("ERROR: 'adjacency' method keyword not found in arguments")
    print("Usage: abaqus cae noGUI=script.py -- adjacency template_cae output_base config_file")
    print("\nReceived sys.argv: %s" % sys.argv)
    sys.exit(1)

print("=" * 70)
print("Batch Damage Model Creation - Auxiliary Script")
print("=" * 70)
print("Method: adjacency (Direct Element ID List)")
print("=" * 70)

# ========================================
# Extract Arguments
# ========================================

if len(user_args) < 4:
    print("ERROR: Insufficient arguments for adjacency method")
    print("Usage: abaqus cae noGUI=script.py -- adjacency template_cae output_base config_file")
    print("Received: %s" % user_args)
    sys.exit(1)

# Arguments
template_cae = user_args[1]
output_base = user_args[2]
config_file = user_args[3]

print("Template CAE: %s" % template_cae)
print("Output base: %s" % output_base)
print("Config file: %s" % config_file)
print("=" * 70)

# ========================================
# Load Configuration
# ========================================

print("\n[Loading Configuration]")

try:
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
except Exception as e:
    print("ERROR: Failed to load configuration file: %s" % str(e))
    sys.exit(1)

# Extract configuration
predefined_regions = config['regions']
models_config = config['models']
global_thickness = config['global_thickness']
generate_health = config.get('generate_health', True)

print("Configuration loaded successfully:")
print("  Predefined regions: %d" % len(predefined_regions))
for region_name, element_ids in predefined_regions.items():
    print("    - %s: %d elements" % (region_name, len(element_ids)))

print("\n  Models to generate: %d" % len(models_config))
for model_name, model_config in models_config.items():
    print("    - %s: %s" % (model_name, model_config['description']))
    for region in model_config['regions']:
        print("      * %s -> %.1fmm" % (region['name'], region['thickness']))

print("\n  Global (healthy) thickness: %.1fmm" % global_thickness)
print("  Generate health model: %s" % generate_health)
print("=" * 70)

# ========================================
# Main Execution
# ========================================

try:
    print("\n[Step 1/4] Opening template CAE file...")
    if not os.path.exists(template_cae):
        print("ERROR: Template CAE not found: %s" % template_cae)
        sys.exit(1)

    openMdb(pathName=template_cae)
    print("  Template opened")

    # Access model and part
    model = mdb.models["Model-1"]
    part = model.parts["Part-1"]
    print("  Model-1 and Part-1 accessed")
    print("  Total elements: %d" % len(part.elements))

    # Create output directory
    if not os.path.exists(output_base):
        os.makedirs(output_base)

    # ========================================
    # Generate Health Model
    # ========================================

    if generate_health:
        print("\n" + "=" * 70)
        print("[Step 2/N] Generating HEALTH model")
        print("=" * 70)

        # Reload template to ensure consistency with damage model workflow
        print("\nReloading template...")
        openMdb(pathName=template_cae)
        model = mdb.models["Model-1"]
        part = model.parts["Part-1"]
        print("  Template reloaded (Total elements: %d)" % len(part.elements))

        # Create section for health model (global thickness)
        section_name = "Section-health-global-%.0fmm" % global_thickness
        model.HomogeneousShellSection(
            name=section_name,
            material='Material-1',
            thickness=global_thickness,
            numIntPts=5
        )
        print("\n  Created section: %s (thickness=%.1fmm)" % (section_name, global_thickness))

        # Create set for all elements (complementary set = all elements for health model)
        all_labels = tuple([e.label for e in part.elements])
        all_elements_seq = part.elements.sequenceFromLabels(labels=all_labels)
        set_name = "Set-health-All"
        part.Set(name=set_name, elements=all_elements_seq)

        # Assign section to all elements
        part.SectionAssignment(
            region=part.sets[set_name],
            sectionName=section_name
        )
        print("  Section assigned to all %d elements via set: %s" % (len(all_labels), set_name))

        # Regenerate assembly to reflect part modifications
        print("\n  Regenerating assembly...")
        try:
            model.rootAssembly.regenerate()
            print("  Assembly regenerated successfully")
        except Exception as e:
            print("  WARNING: Assembly regeneration failed: %s" % str(e))

        print("\n  Summary for health model:")
        print("    No specific damage regions")
        print("    Global thickness assigned to all %d elements" % len(all_labels))

        # Save health model
        health_cae = os.path.join(output_base, "health.cae")
        print("\nSaving health model...")
        mdb.saveAs(pathName=health_cae)
        print("  Saved: %s" % health_cae)

        # Generate INP for health model
        print("\nGenerating INP file for health model...")
        try:
            job_name = "Job-health"
            mdb.Job(
                name=job_name,
                model='Model-1',
                type=ANALYSIS,
                explicitPrecision=SINGLE,
                nodalOutputPrecision=SINGLE,
                description='Job for health',
                userSubroutine='',
                numCpus=1,
                memory=90,
                memoryUnits=PERCENTAGE
            )
            mdb.jobs[job_name].writeInput(consistencyChecking=OFF)

            default_inp = os.path.join(output_base, "%s.inp" % job_name)
            health_inp = os.path.join(output_base, "health.inp")
            if os.path.exists(default_inp):
                if os.path.exists(health_inp):
                    os.remove(health_inp)
                os.rename(default_inp, health_inp)
                inp_size_kb = os.path.getsize(health_inp) / 1024.0
                print("  Saved: %s (%.1f KB)" % (health_inp, inp_size_kb))
        except Exception as e:
            print("  WARNING: INP generation failed: %s" % str(e))

    # ========================================
    # Process Damage Models
    # ========================================

    print("\n[Step 3/N] Processing damage models...")

    model_idx = 0
    for model_name, model_config in models_config.items():
        model_idx += 1
        print("\n" + "=" * 70)
        print("Model %d/%d: %s" % (model_idx, len(models_config), model_name))
        print("Description: %s" % model_config['description'])
        print("=" * 70)

        # Reload template
        print("\nReloading template...")
        openMdb(pathName=template_cae)
        model = mdb.models["Model-1"]
        part = model.parts["Part-1"]
        print("  Template reloaded (Total elements: %d)" % len(part.elements))

        # Track all assigned elements
        all_assigned_labels = set()

        # Process each region in this model
        print("\nProcessing %d region(s) for this model..." % len(model_config['regions']))

        for region_idx, region_spec in enumerate(model_config['regions'], 1):
            region_name = region_spec['name']
            region_thickness = region_spec['thickness']

            # Get element IDs from predefined regions
            if region_name not in predefined_regions:
                print("  ERROR: Region '%s' not found in predefined regions!" % region_name)
                sys.exit(1)

            region_element_ids = predefined_regions[region_name]

            print("\n  Region %d/%d: %s" % (region_idx, len(model_config['regions']), region_name))
            print("    Elements: %d" % len(region_element_ids))
            print("    Thickness: %.1fmm" % region_thickness)

            # Create section for this region
            section_name = "Section-%s-%s-%.0fmm" % (model_name, region_name, region_thickness)
            model.HomogeneousShellSection(
                name=section_name,
                material='Material-1',
                thickness=region_thickness,
                numIntPts=5
            )
            print("    Created section: %s" % section_name)

            # Create element set and assign section
            region_labels_tuple = tuple(region_element_ids)
            region_seq = part.elements.sequenceFromLabels(labels=region_labels_tuple)
            set_name = "Set-%s-%s" % (model_name, region_name)
            part.Set(name=set_name, elements=region_seq)
            part.SectionAssignment(
                region=part.sets[set_name],
                sectionName=section_name
            )
            print("    Section assigned to set: %s" % set_name)

            # Track assigned elements
            all_assigned_labels.update(region_element_ids)

        # Calculate complementary region (remaining elements)
        all_labels_set = set([e.label for e in part.elements])
        remaining_labels = all_labels_set - all_assigned_labels

        # Find or create section for complementary region with global thickness
        print("\n  Processing complementary region (remaining elements):")
        print("    Remaining elements: %d" % len(remaining_labels))

        if len(remaining_labels) > 0:
            # Try to find original section with global thickness from template
            print("    Finding original section with thickness=%.1fmm from template..." % global_thickness)
            original_section_name = None
            for section_name in model.sections.keys():
                section = model.sections[section_name]
                if hasattr(section, 'thickness') and abs(section.thickness - global_thickness) < 0.01:
                    original_section_name = section_name
                    print("      Found original section: %s (thickness=%.1fmm)" % (section_name, section.thickness))
                    break

            # If not found, create new section for global thickness
            if original_section_name is None:
                print("      Original section not found, creating new section for global thickness...")
                original_section_name = "Section-%s-Global-%.0fmm" % (model_name, global_thickness)
                model.HomogeneousShellSection(
                    name=original_section_name,
                    material='Material-1',
                    thickness=global_thickness,
                    numIntPts=5
                )
                print("      Created section: %s (thickness=%.1fmm)" % (original_section_name, global_thickness))

            # Create set for complementary region and assign section
            remaining_labels_tuple = tuple(sorted(remaining_labels))
            remaining_seq = part.elements.sequenceFromLabels(labels=remaining_labels_tuple)
            remaining_set_name = "Set-%s-Complementary" % model_name
            part.Set(name=remaining_set_name, elements=remaining_seq)

            # Assign section to complementary region
            part.SectionAssignment(
                region=part.sets[remaining_set_name],
                sectionName=original_section_name
            )
            print("    Complementary region assigned to section %s via set: %s" % (original_section_name, remaining_set_name))
        else:
            print("    All elements assigned to specific regions (no complementary region needed)")

        print("\n  Summary for %s:" % model_name)
        print("    Specific regions: %d (%d elements total)" % (
            len(model_config['regions']),
            len(all_assigned_labels)
        ))
        print("    Global thickness: %d elements" % len(remaining_labels))

        # Regenerate assembly to reflect part modifications
        print("\n  Regenerating assembly...")
        try:
            model.rootAssembly.regenerate()
            print("  Assembly regenerated successfully")
        except Exception as e:
            print("  WARNING: Assembly regeneration failed: %s" % str(e))

        # Save model
        output_cae = os.path.join(output_base, "%s.cae" % model_name)
        print("\nSaving model...")
        mdb.saveAs(pathName=output_cae)
        print("  Saved: %s" % output_cae)

        # Generate INP file
        print("\nGenerating INP file...")
        try:
            job_name = "Job-%s" % model_name
            mdb.Job(
                name=job_name,
                model='Model-1',
                type=ANALYSIS,
                explicitPrecision=SINGLE,
                nodalOutputPrecision=SINGLE,
                description='Job for %s' % model_name,
                userSubroutine='',
                numCpus=1,
                memory=90,
                memoryUnits=PERCENTAGE
            )
            mdb.jobs[job_name].writeInput(consistencyChecking=OFF)

            default_inp = os.path.join(output_base, "%s.inp" % job_name)
            output_inp = os.path.join(output_base, "%s.inp" % model_name)
            if os.path.exists(default_inp):
                if os.path.exists(output_inp):
                    os.remove(output_inp)
                os.rename(default_inp, output_inp)
                inp_size_kb = os.path.getsize(output_inp) / 1024.0
                print("  Saved: %s (%.1f KB)" % (output_inp, inp_size_kb))
        except Exception as e:
            print("  WARNING: INP generation failed: %s" % str(e))

    # ========================================
    # Final Summary
    # ========================================

    print("\n" + "=" * 70)
    print("Batch Generation Complete!")
    print("=" * 70)

    total_models = len(models_config) + (1 if generate_health else 0)
    print("Models generated: %d" % total_models)
    if generate_health:
        print("  - 1 health model")
    print("  - %d damage/repaired models" % len(models_config))

    print("\nOutput directory: %s" % output_base)
    print("\nGenerated files:")

    all_model_names = []
    if generate_health:
        all_model_names.append("health")
    all_model_names.extend(list(models_config.keys()))

    for name in all_model_names:
        cae_file = os.path.join(output_base, "%s.cae" % name)
        inp_file = os.path.join(output_base, "%s.inp" % name)

        cae_status = "OK" if os.path.exists(cae_file) else "MISSING"
        inp_status = "OK" if os.path.exists(inp_file) else "MISSING"

        print("  - %s:" % name)
        print("      CAE: %s" % cae_status)
        print("      INP: %s" % inp_status)
    print("=" * 70)

    sys.exit(0)

except Exception as e:
    print("\n" + "=" * 70)
    print("ERROR: Batch generation failed")
    print("=" * 70)
    print("Exception: %s" % str(e))

    import traceback
    print("\nTraceback:")
    print(traceback.format_exc())
    print("=" * 70)

    sys.exit(1)
