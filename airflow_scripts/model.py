import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from ml.train import Trainer
from ml.models import LinearModel
from ml.data import load_mnist_data
from ml.utils import set_device
from urllib.parse import urlparse

mlflow.set_tracking_uri("sqlite:///db/backend.db")
mlflowclient = MlflowClient(mlflow.get_tracking_uri(), mlflow.get_registry_uri())


def train_model(model_name: str, hyperparams: dict, epochs: int):
    # Setup env
    device = set_device()
    # Set MLflow tracking
    mlflow.set_experiment("MNIST")
    with mlflow.start_run() as run:
        # Log hyperparameters
        mlflow.log_params(hyperparams)

        # Prepare for training
        print("Loading data...")
        train_dataloader, test_dataloader = load_mnist_data()

        # Train
        print("Training model")
        model = LinearModel(hyperparams).to(device)
        trainer = Trainer(model, device=device)  # Default configs
        history = trainer.train(epochs, train_dataloader, test_dataloader)

        print("Logging results")
        # Log in mlflow
        for metric_name, metric_values in history.items():
            for metric_value in metric_values:
                mlflow.log_metric(metric_name, metric_value)

        # Register model
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        print(f"{tracking_url_type_store=}")

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            mlflow.pytorch.log_model(model, "LinearModel", registered_model_name=model_name, conda_env=mlflow.pytorch.get_default_conda_env())
        else:
            mlflow.pytorch.log_model(model, "LinearModel-MNIST", registered_model_name=model_name)
        # Transition to production. We search for the last model with the name and we stage it to production
        mv = mlflowclient.search_model_versions(f"name='{model_name}'")[-1]  # Take last model version
        mlflowclient.transition_model_version_stage(name=mv.name, version=mv.version, stage="production")
