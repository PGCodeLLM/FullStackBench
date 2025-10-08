import argparse
import asyncio
import logging
from pathlib import Path

from sandbox_fusion import (
    SubmitRequest,
    TestConfig,
    set_endpoint,
    submit_async,
)
from tqdm.asyncio import tqdm_asyncio

from utils import read_jsonl, write_jsonl

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def evaluate_sample(sample: dict) -> list[dict]:
    completions: str | list[str] = sample.get('completion', [])
    if isinstance(completions, str):
        completions = [completions]
    n = len(completions)
    results = []
    pass_at_k = False
    all_k_correct = True
    for completion in completions:
        result = {}
        eval_result = await submit_async(
            SubmitRequest(
                dataset='FullStackBench',
                id=sample['id'],
                completion=completion,
                config=TestConfig(
                    compile_timeout=50,
                    run_timeout=50,
                    dataset_type='AutoEvalDataset',
                    provided_data=sample
                )
            )
        )
        passed = eval_result.accepted
        result['accepted'] = passed
        result['pass_at_1'] = 1 if passed else 0
        result['task_id']= sample['id']
        result['n' ] = n
        if passed:
            pass_at_k = True
        else:
            all_k_correct = False
        if 'labels' in sample and 'programming_language' in sample['labels']:
            result['programming_language'] = sample['labels']['programming_language']
            result['category'] = sample['labels'].get('category', '')
            result['difficulty'] = sample['labels'].get('difficulty', '')
        results.append(result)
    for result in results:
        result['pass_at_k'] = 1 if pass_at_k else 0
        result['all_k_correct'] = 1 if all_k_correct else 0
    return results


async def main():
    parser = argparse.ArgumentParser(
        description='Evaluate FullStackBench predictions using SandboxFusion'
    )
    parser.add_argument(
        '--inference-file',
        type=str,
        required=True,
        help='Path to inference output file (JSONL format)'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        required=True,
        help='Path to save evaluation results (JSONL format)'
    )
    parser.add_argument(
        '--sandbox-endpoint',
        type=str,
        default='http://localhost:8080',
        help='SandboxFusion server endpoint'
    )

    args = parser.parse_args()

    inference_file = Path(args.inference_file)
    if not inference_file.exists():
        logger.error(f"Inference file not found: {args.inference_file}")
        return 1

    set_endpoint(args.sandbox_endpoint)
    logger.info(f"Using SandboxFusion endpoint: {args.sandbox_endpoint}")

    logger.info(f"Reading inference results from: {args.inference_file}")
    samples = read_jsonl(args.inference_file)
    logger.info(f"Found {len(samples)} samples to evaluate")

    logger.info("Starting evaluation...")
    tasks = [evaluate_sample(sample) for sample in samples]
    results = []

    for result in await tqdm_asyncio.gather(*tasks, desc="Evaluating"):
        for r in result:
            results.append(r)

    pass_count = sum([r['pass_at_1'] for r in results])
    pass_rate = pass_count / len(results) if results else 0.0

    logger.info("Evaluation complete!")
    logger.info(f"Pass rate: {pass_rate:.4%} ({pass_count}/{len(results)})")

    logger.info(f"Writing results to: {args.output_file}")
    write_jsonl(args.output_file, [r for r in results])

    return 0


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    exit(exit_code)
