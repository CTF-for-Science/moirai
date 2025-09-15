from pathlib import Path

top_dir = Path(__file__).parent.parent

bash_template_0 = \
"""\
repo="/home/alexey/Git/CTF-for-Science/models/moirai"

# Create logs directory and set up logging
mkdir -p $repo/logs
exec > >(tee -a $repo/logs/{log_filename}) 2>&1

echo "Running Python"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /home/alexey/.virtualenvs/moirai/bin/activate

"""

bash_template_1 = \
"""\
python -u $repo/ctf/forecast_ctf.py --dataset "{dataset}" --pair_id "{pair_id}" --recon_ctx "{recon_ctx}" --validation "{validation}" --identifier "{identifier}" --device "{device}"

"""

bash_template_2 = \
"""\
echo "Finished running Python"

"""

# Parameters
n_parallel = 2
datasets = ["seismo"]
pair_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
validation = 0
recon_ctx = 50

# Create and clean up bash repo
bash_dir = top_dir / 'bash'
bash_dir.mkdir(exist_ok=True)
for file in bash_dir.glob('*.sh'):
    file.unlink()

device_counter = 0
devices = ["cuda:0", "cuda:1"]
total_scripts = len(devices) * n_parallel

# Initialize bash scripts for each device and parallel index
bash_scripts = {}
for device in devices:
    for parallel_idx in range(n_parallel):
        script_key = f"{device}_{parallel_idx}"
        device_num = device.split(':')[1]
        log_filename = f"run_cuda_{device_num}_{parallel_idx}.log"
        bash_scripts[script_key] = bash_template_0.format(log_filename=log_filename)

for dataset in datasets:
    for pair_id in pair_ids:
        identifier = f"{dataset}_{pair_id}"

        # Determine which device and parallel script to use based on counter
        device_idx = device_counter % len(devices)
        parallel_idx = (device_counter // len(devices)) % n_parallel
        current_device = devices[device_idx]
        script_key = f"{current_device}_{parallel_idx}"
        
        cmd = bash_template_1.format(
            dataset=dataset,
            pair_id=pair_id,
            recon_ctx=recon_ctx,
            validation=validation,
            identifier=identifier,
            device=current_device,
        )

        # Add the command to the appropriate bash script
        bash_scripts[script_key] += cmd

        device_counter += 1

# Add the closing template to each script and write to files
for script_key, script_content in bash_scripts.items():
    script_content += bash_template_2
    
    # Parse device and parallel index from script_key
    device, parallel_idx = script_key.rsplit('_', 1)
    device_num = device.split(':')[1]  # Extract number from "cuda:X"
    filename = f"run_cuda_{device_num}_{parallel_idx}.sh"
    filepath = bash_dir / filename
    
    with open(filepath, 'w') as f:
        f.write(script_content)
    
    # Make the script executable
    filepath.chmod(0o755)
    
    print(f"Generated bash script: {filepath}")

print(f"Total jobs: {len(datasets) * len(pair_ids)}")
print(f"Total scripts generated: {total_scripts}")
print(f"Jobs per script: ~{len(datasets) * len(pair_ids) // total_scripts} (with remainder distributed)")
