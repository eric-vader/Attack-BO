# Generate the illustrations used
mlflow run . -P param_file=sample/illustrations/1d.yml
mlflow run . -P param_file=sample/illustrations/forrester.yml
mlflow run . -P param_file=sample/illustrations/levy.yml
mlflow run . -P param_file=sample/illustrations/levy_hard.yml
mlflow run . -P param_file=sample/illustrations/bohachevsky.yml
mlflow run . -P param_file=sample/illustrations/bohachevsky_hard.yml
mlflow run . -P param_file=sample/illustrations/branin.yml
mlflow run . -P param_file=sample/illustrations/camelback.yml