import os
import sys
import subprocess
import time
import json
from datetime import datetime

TEST_CONFIG = {
    'system_data_count': 200,
    'epoch_T': 1,
    'communication_round_I': 2,
    'num_clients_K': 3,
    'exp_repeat_times': 1,
    'parallel_repeats': 1,
    'checkpoint_save_freq': 0,
    'split_strategy': 'Uniform',
}

ALGORITHMS = [
    'FedAvg',
    'FedProx',
    'Scaffold',
    'FedNova',
    'FedRep',
    'FedProto',
    'FedFB',
    'FedRenyi',
    'OSFL',
    'FairFed',
    'FedFair',
    'FL_FairBatch',
    'FedMix',
    'NaiveMix',
    'mFairFL',
    'PDFFed',
    'PraFFL',
    'FedFACT',
    'LoGoFair',
    'DOSFL',
    'Co_Boosting',
]

TASKS = {
    'SENT_CLF': {
        'script': 'main_SENT_CLF.py',
        'datasets': ['moji'],
        'algorithms': ALGORITHMS,
    },
    'IMG_CLF': {
        'script': 'main_IMG_CLF.py',
        'datasets': ['CelebA'],
        'algorithms': ALGORITHMS,
    },
    'Tabular_CLF': {
        'script': 'main_Tabular_CLF.py',
        'datasets': ['ADULT'],
        'algorithms': ALGORITHMS,
        'model_types': ['ANN'],
    },
}

EXCLUSION_RULES = [
    ('IMG_CLF', 'FedRep'),
    ('IMG_CLF', 'FedProto'),
    ('Tabular_CLF', 'FedRep'),
    ('Tabular_CLF', 'FedProto'),
]


def should_skip(task, algorithm):
    return (task, algorithm) in EXCLUSION_RULES


def run_test(task, dataset, algorithm, model_type=None):
    script = TASKS[task]['script']
    cmd = [
        sys.executable, script,
        '-algorithm', algorithm,
        '-dataset', dataset,
        '-task', task,
        '-system_data_count', str(TEST_CONFIG['system_data_count']),
        '-split_strategy', TEST_CONFIG['split_strategy'],
        '-communication_round_I', str(TEST_CONFIG['communication_round_I']),
        '-num_clients_K', str(TEST_CONFIG['num_clients_K']),
        '-exp_repeat_times', str(TEST_CONFIG['exp_repeat_times']),
        '-parallel_repeats', str(TEST_CONFIG['parallel_repeats']),
        '-checkpoint_save_freq', str(TEST_CONFIG['checkpoint_save_freq']),
    ]

    if task == 'Tabular_CLF' and model_type:
        cmd.extend(['-model_type', model_type])

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ''

    print(f"\n{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*80}")

    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        duration = time.time() - start_time
        success = result.returncode == 0

        if success:
            print(f"[PASS] {task} | {dataset} | {algorithm} | Duration: {duration:.2f}s")
        else:
            print(f"[FAIL] {task} | {dataset} | {algorithm} | Exit code: {result.returncode} | Duration: {duration:.2f}s")
            print("STDOUT:")
            print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
            print("\nSTDERR:")
            print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)

        return {
            'task': task,
            'dataset': dataset,
            'algorithm': algorithm,
            'model_type': model_type,
            'success': success,
            'returncode': result.returncode,
            'duration': duration,
            'stdout': result.stdout,
            'stderr': result.stderr,
        }
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print(f"[TIMEOUT] {task} | {dataset} | {algorithm} | Duration: {duration:.2f}s")
        return {
            'task': task,
            'dataset': dataset,
            'algorithm': algorithm,
            'model_type': model_type,
            'success': False,
            'returncode': -1,
            'duration': duration,
            'stdout': '',
            'stderr': 'Timeout after 300 seconds',
        }
    except Exception as e:
        duration = time.time() - start_time
        print(f"[ERROR] {task} | {dataset} | {algorithm} | {e} | Duration: {duration:.2f}s")
        return {
            'task': task,
            'dataset': dataset,
            'algorithm': algorithm,
            'model_type': model_type,
            'success': False,
            'returncode': -2,
            'duration': duration,
            'stdout': '',
            'stderr': str(e),
        }


def main():
    results = []
    total_tests = 0
    passed_tests = 0
    failed_tests = 0

    for task, config in TASKS.items():
        for dataset in config['datasets']:
            for algorithm in config['algorithms']:
                if should_skip(task, algorithm):
                    print(f"\n[SKIP] {task} | {dataset} | {algorithm} (not supported)")
                    continue

                if task == 'Tabular_CLF':
                    for model_type in config['model_types']:
                        result = run_test(task, dataset, algorithm, model_type)
                        results.append(result)
                        total_tests += 1
                        if result['success']:
                            passed_tests += 1
                        else:
                            failed_tests += 1
                else:
                    result = run_test(task, dataset, algorithm)
                    results.append(result)
                    total_tests += 1
                    if result['success']:
                        passed_tests += 1
                    else:
                        failed_tests += 1

    print(f"\n{'='*80}")
    print("SMOKE TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Pass rate: {passed_tests / total_tests * 100:.2f}%" if total_tests > 0 else "N/A")

    report = {
        'timestamp': datetime.now().isoformat(),
        'test_config': TEST_CONFIG,
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'failed_tests': failed_tests,
        'pass_rate': passed_tests / total_tests * 100 if total_tests > 0 else 0,
        'results': results,
    }

    report_path = f"smoke_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to: {report_path}")

    if failed_tests > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
