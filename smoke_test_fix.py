import subprocess
import sys
import time

algorithms = [
    'FedFair',
    'FedRenyi', 
    'FedProto',
    'FedMix',
    'NaiveMix',
    'FL_FairBatch',
    'SeparateTraining',
    'OSFL',
    'DOSFL',
    'ProxProbability',
    'CoBoosting'
]

tasks = [
    ('main_Tabular_CLF.py', 'COMPAS', 'ANN'),
]

results = {}

for algo in algorithms:
    results[algo] = {'success': [], 'failed': []}
    for main_file, dataset, model in tasks:
        cmd = [
            sys.executable, main_file,
            '-dataset', dataset,
            '-algorithm', algo,
            '-model_type', model,
            '-system_data_count', '50',
            '-num_clients_K', '2',
            '-communication_round_I', '1',
            '-algorithm_epoch_T', '1',
            '-split_strategy', 'Dirichlet1'
        ]
        
        print(f"\n{'='*60}")
        print(f"Testing {algo} on {dataset} ({model})")
        print(f"Command: {' '.join(cmd)}")
        print('='*60)
        
        start_time = time.time()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd='.')
            elapsed = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✓ SUCCESS in {elapsed:.1f}s")
                results[algo]['success'].append(f"{dataset} ({model})")
            else:
                print(f"✗ FAILED in {elapsed:.1f}s")
                print(f"STDERR:\n{result.stderr[-1500:]}" if len(result.stderr) > 1500 else f"STDERR:\n{result.stderr}")
                print(f"STDOUT:\n{result.stdout[-500:]}" if len(result.stdout) > 500 else f"STDOUT:\n{result.stdout}")
                results[algo]['failed'].append(f"{dataset} ({model})")
                
        except subprocess.TimeoutExpired:
            print(f"✗ TIMEOUT after {elapsed:.1f}s")
            results[algo]['failed'].append(f"{dataset} ({model})")
        except Exception as e:
            print(f"✗ ERROR: {e}")
            results[algo]['failed'].append(f"{dataset} ({model})")

print("\n\n" + "="*80)
print("SMOKE TEST SUMMARY")
print("="*80)

total_success = 0
total_failed = 0

for algo, result in results.items():
    succ = len(result['success'])
    fail = len(result['failed'])
    total_success += succ
    total_failed += fail
    
    status = "✓ ALL PASS" if fail == 0 else f"✗ {fail} FAILED"
    print(f"\n{algo}: {status}")
    if result['success']:
        print(f"  Success: {', '.join(result['success'])}")
    if result['failed']:
        print(f"  Failed: {', '.join(result['failed'])}")

print(f"\nTotal: {total_success} passed, {total_failed} failed out of {total_success + total_failed} tests")
