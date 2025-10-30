import docker
import json
import logging
import traceback
from argparse import ArgumentParser
from pathlib import Path, PurePosixPath

# Import necessary components from the swebench library
from swebench.harness.constants import (
    DOCKER_PATCH, DOCKER_USER, DOCKER_WORKDIR, KEY_INSTANCE_ID,
    KEY_PREDICTION, LOG_REPORT, LOG_INSTANCE, LOG_TEST_OUTPUT,
    RUN_EVALUATION_LOG_DIR, UTF8, APPLY_PATCH_FAIL, APPLY_PATCH_PASS
)
from swebench.harness.docker_utils import (
    cleanup_container, copy_to_container, exec_run_with_timeout
)
from swebench.harness.docker_build import setup_logger, close_logger
from swebench.harness.grading import get_eval_report
from swebench.harness.test_spec.test_spec import make_test_spec
from swebench.harness.utils import (
    EvaluationError, load_swebench_dataset, get_predictions_from_file
)

# [cite_start]A list of commands to attempt applying the patch, adapted from the official script [cite: 917]
GIT_APPLY_CMDS = [
    "git apply --verbose",
    "git apply --verbose --reject",
    "patch --batch --fuzz=5 -p1 -i",
]


def run_evaluation_on_image(
    image_name: str,
    instance: dict,
    prediction: dict,
    client: docker.DockerClient,
    run_id: str,
    timeout: int,
):
    """
    Runs the SWE-bench evaluation for a single instance on a pre-existing Docker image.

    This function adapts the core logic from `swebench.harness.run_evaluation.run_instance`.
    Instead of building a new Docker image, it directly creates a container from the
    provided `image_name` and then proceeds with the standard evaluation steps.

    Args:
        image_name (str): The name of the Docker image to use.
        instance (dict): The SWE-bench task instance dictionary.
        prediction (dict): The prediction dictionary containing the model's patch.
        client (docker.DockerClient): The Docker client object.
        run_id (str): A unique identifier for this evaluation run.
        timeout (int): The timeout in seconds for running the tests.
    """
    # Create a TestSpec object to get necessary scripts and platform info from the instance
    test_spec = make_test_spec(instance)
    instance_id = test_spec.instance_id

    # Set up a directory for logs and reports for this specific run
    log_dir = RUN_EVALUATION_LOG_DIR / run_id / "custom_image_run" / instance_id
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / LOG_INSTANCE
    logger = setup_logger(instance_id, log_file)

    container = None
    try:
        # 1. Create and start a container from the specified existing image
        logger.info(f"Creating container for {instance_id} from image {image_name}...")
        container_name = test_spec.get_instance_container_name(run_id)
        container = client.containers.create(
            image=image_name,
            name=container_name,
            user=DOCKER_USER,
            detach=True,
            command="tail -f /dev/null",  # Keep the container running
            platform=test_spec.platform,
        )
        container.start()
        logger.info(f"Container for {instance_id} started: {container.id}")

        # 2. Copy the model-generated patch into the container
        patch_path = log_dir / "patch.diff"
        patch_path.write_text(prediction.get(KEY_PREDICTION) or "")
        copy_to_container(container, patch_path, PurePosixPath(DOCKER_PATCH))
        logger.info(f"Patch for {instance_id} copied to container.")

        # 3. Attempt to apply the patch within the container
        applied_patch = False
        apply_output = ""
        for cmd in GIT_APPLY_CMDS:
            apply_result = container.exec_run(
                f"{cmd} {DOCKER_PATCH}",
                workdir=DOCKER_WORKDIR,
                user=DOCKER_USER,
            )
            apply_output = apply_result.output.decode(UTF8)
            if apply_result.exit_code == 0:
                logger.info(f"{APPLY_PATCH_PASS}:\n{apply_output}")
                applied_patch = True
                break
            else:
                logger.warning(f"Patch apply command '{cmd}' failed. Trying next...")

        if not applied_patch:
            raise EvaluationError(instance_id, f"{APPLY_PATCH_FAIL}:\n{apply_output}", logger)

        # 4. Copy the auto-generated evaluation script into the container
        eval_script_path = log_dir / "eval.sh"
        eval_script_path.write_text(test_spec.eval_script)
        copy_to_container(container, eval_script_path, PurePosixPath("/eval.sh"))
        logger.info(f"Evaluation script for {instance_id} copied to container.")

        # 5. Execute the evaluation script with a timeout
        logger.info(f"Running evaluation script with a timeout of {timeout} seconds...")
        test_output, timed_out, _ = exec_run_with_timeout(container, "/bin/bash /eval.sh", timeout)

        test_output_path = log_dir / LOG_TEST_OUTPUT
        test_output_path.write_text(test_output)
        logger.info(f"Test output saved to {test_output_path}")

        if timed_out:
            (log_dir / LOG_TEST_OUTPUT).write_text(test_output + f"\n\nTimeout Error: Task exceeded {timeout} seconds.")
            raise EvaluationError(instance_id, f"Test timed out after {timeout} seconds.", logger)

        # 6. Grade the results from the logs and generate a report
        logger.info(f"Grading results for {instance_id}...")
        report = get_eval_report(
            test_spec=test_spec,
            prediction=prediction,
            test_log_path=test_output_path,
            include_tests_status=True,
        )
        report_path = log_dir / LOG_REPORT
        report_path.write_text(json.dumps(report, indent=4))
        logger.info(f"Report for {instance_id} saved to {report_path}")
        print(f"✅ Evaluation successful for {instance_id}. Report at: {report_path}")

    except Exception as e:
        error_msg = f"Error during evaluation for {instance_id}: {e}\n{traceback.format_exc()}"
        logger.error(error_msg)
        print(f"❌ Evaluation failed for {instance_id}. Check logs for details: {log_file}")
    finally:
        # 7. Ensure the container is stopped and removed
        if container:
            cleanup_container(client, container, logger)
            logger.info(f"Container {container.name} cleaned up.")
        close_logger(logger)


def main():
    parser = ArgumentParser(description="Run a SWE-bench evaluation on a pre-built Docker image.")
    parser.add_argument("--image_name", type=str, required=True, help="The name of the existing Docker image to use for evaluation (e.g., 'ghcr.io/epoch-research/swe-bench.eval.x86_64.astropy__astropy-13236').")
    parser.add_argument("--instance_id", type=str, required=True, help="The SWE-bench instance ID to evaluate (e.g., 'astropy__astropy-13236').")
    parser.add_argument("--predictions_path", type=str, required=True, help="Path to the JSON/JSONL file containing the model's patch prediction.")
    parser.add_argument("--dataset_name", type=str, default="princeton-nlp/SWE-bench", help="The SWE-bench dataset on Hugging Face to load instance data from.")
    parser.add_argument("--split", type=str, default="test", help="The dataset split to use (e.g., 'test' or 'dev').")
    parser.add_argument("--run_id", type=str, default="custom_run", help="A unique identifier for this evaluation run, used for logging.")
    parser.add_argument("--timeout", type=int, default=1800, help="Timeout in seconds for running the evaluation script inside the container.")
    args = parser.parse_args()

    client = docker.from_env()

    # 1. Verify that the specified Docker image exists locally
    try:
        client.images.get(args.image_name)
        print(f"Found Docker image: {args.image_name}")
    except docker.errors.ImageNotFound:
        print(f"Error: Docker image '{args.image_name}' not found.")
        print("Please ensure the image is available on your local machine before running.")
        return
    except Exception as e:
        print(f"An error occurred while checking for the Docker image: {e}")
        return

    # 2. Load the specified task instance from the dataset
    try:
        instance = load_swebench_dataset(args.dataset_name, args.split, [args.instance_id])[0]
        print(f"Loaded instance '{args.instance_id}' from {args.dataset_name}/{args.split}")
    except Exception as e:
        print(f"Error loading instance '{args.instance_id}': {e}")
        return

    # 3. Load the corresponding prediction (patch) for the instance
    try:
        predictions = get_predictions_from_file(args.predictions_path, args.dataset_name, args.split)
        prediction = next((p for p in predictions if p[KEY_INSTANCE_ID] == args.instance_id), None)
        if prediction is None:
            print(f"Error: No prediction found for instance_id '{args.instance_id}' in '{args.predictions_path}'.")
            return
        print(f"Loaded prediction for '{args.instance_id}'.")
    except Exception as e:
        print(f"Error loading predictions from '{args.predictions_path}': {e}")
        return

    # 4. Run the evaluation using the loaded data and existing image
    run_evaluation_on_image(
        image_name=args.image_name,
        instance=instance,
        prediction=prediction,
        client=client,
        run_id=args.run_id,
        timeout=args.timeout,
    )

if __name__ == "__main__":
    main()
