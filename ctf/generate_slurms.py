from pathlib import Path

top_dir = Path(__file__).parent.parent

cmd_template = \
"""\
#!/bin/bash

#SBATCH --account=amath
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=0
#SBATCH --mem={memory}G
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --nice=0

#SBATCH --job-name="{identifier}"
#SBATCH --output=/mmfs1/home/alexeyy/storage/CTF-for-Science/models/moirai/logs/"{identifier}".out

#SBATCH --mail-type=NONE
#SBATCH --mail-user=alexeyy@uw.edu

identifier={identifier}

repo="/mmfs1/home/alexeyy/storage/CTF-for-Science/models/moirai"
datasets="/mmfs1/home/alexeyy/storage/data"

recon_ctx={recon_ctx}
dataset={dataset}
pair_id={pair_id}
validation={validation}

echo "Running Apptainer"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

apptainer run --nv --cwd "/app/code" --bind "$repo":"/app/code" gpu.sif python -u /app/code/ctf/forecast_ctf.py --dataset $dataset --pair_id $pair_id --validation $validation --identifier $identifier

echo "Finished running Apptainer"

#echo "Running Python"

#source "$repo"/.venv/bin/activate

#python -u "$repo"/ctf/forecast_ctf.py --dataset "$dataset" --pair_id "$pair_id" --recon_ctx "$recon_ctx" --validation "$validation" --identifier "$identifier"

#echo "Finished running Python"
"""

# Clean up slurm repo
slurm_dir = top_dir / 'slurms'
for file in slurm_dir.glob('*.slurm'):
    file.unlink()

datasets = ["ODE_Lorenz", "PDE_KS", "KS_Official", "Lorenz_Official"]
pair_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
validations = [True, False]
recon_ctxs = [20]
partition = "cpu-g2"
memory = 64

skip_count = 0
write_count = 0
total_count = 0

for dataset in datasets:
    for pair_id in pair_ids:
        for validation in validations:
            for recon_ctx in recon_ctxs:
                identifier = f"{dataset}_p{pair_id}_v{validation:d}_r{recon_ctx}"

                cmd = cmd_template.format(
                    dataset=dataset,
                    pair_id=pair_id,
                    validation=validation,
                    recon_ctx=recon_ctx,
                    identifier=identifier,
                    partition=partition,
                    memory=memory,
                )

                total_count += 1

                # Skip creating slurms that are completed
                pickle_file = top_dir / 'pickles' / f'{identifier}.pkl'

                if pickle_file.exists():
                    #print(f'Skipping {identifier}')
                    skip_count += 1
                    continue

                with open(top_dir / 'slurms' / f'{identifier}.slurm', "w") as f:
                    f.write(cmd)
                    write_count += 1

print(f"Skipped {skip_count} jobs")
print(f"Created {write_count} jobs")
print(f"Total jobs: {total_count}")
