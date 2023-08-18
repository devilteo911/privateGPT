import os
from pathlib import Path
import argparse
import boto3
from dotenv import load_dotenv
from loguru import logger
from mlflow.tracking import MlflowClient


def main(args):
    """
    Main function that performs the desired operations.

    Args:
        args: The command-line arguments passed to the script.
    """
    load_dotenv()

    mlflow_cli = MlflowClient(tracking_uri=os.environ["MLFLOW_ENDPOINT"])

    required_path = args.path
    experiment_id = args.experiment_id

    run = mlflow_cli.create_run(
        experiment_id=experiment_id, run_name=required_path.split("/")[-1]
    )

    logger.info(
        f'u: {os.environ["AWS_ACCESS_KEY_ID"]}, pwd:{os.environ["AWS_SECRET_ACCESS_KEY"]}, tok: {os.environ["AWS_SESSION_TOKEN"]}, ep: {os.environ["MINIO_COLLAUDO"]}'
    )

    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ["MINIO_COLLAUDO"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        aws_session_token=os.environ["AWS_SESSION_TOKEN"],
        use_ssl=False,
        verify=False,
    )

    dir = Path(required_path)
    Path(required_path + "/MLmodel").touch()
    files = dir.rglob("*")
    for file in files:
        if not file.is_dir():
            if file.name == "MLmodel":
                s3_dest = experiment_id + f"/{run.info.run_id}/artifacts/model/MLmodel"
                s3.upload_file(str(file), "mlflow-master", s3_dest)
            else:
                s3_dest = (
                    experiment_id
                    + f"/{run.info.run_id}/artifacts/models--{required_path.split('/')[-1]}"
                    + str(file).split(required_path)[-1]
                )
                try:
                    s3.upload_file(str(file), "mlflow-master", s3_dest)
                except RecursionError:
                    logger.warning(
                        "RecursionError on file: {} . This is probably due to a boto3 version that is not supported by mlflow".format(
                            str(file)
                        )
                    )

    mlflow_cli.set_terminated(run.info.run_id)


if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="MLFlow Uploader")

    # Add the path argument
    parser.add_argument("--path", "-p", type=str, help="The required path")

    # Add the experiment id argument
    parser.add_argument(
        "--experiment_id", "-e", type=str, help="The ID of the experiment on mlflow"
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Execute the main function with the parsed arguments
    main(args)
