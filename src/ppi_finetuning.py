import os
from argparse import ArgumentParser
from typing import Dict

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor)
from test_tube.argparse_hopt import TTNamespace

from src import npe_ppi_logger, settings
from src.utils.ProtBertPPIArgParser import ProtBertPPIArgParser
from src.modeling.ProtBertPPIModel import ProtBertPPIModel

# %% Set environment variables
# works on mac only with this:.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# %%
def generate_parser():
    parser = ArgumentParser()

    parser = Trainer.add_argparse_args(parser)
    parser._action_groups[1].title = 'PyTorch Lightning Trainer options'

    parser = ProtBertPPIModel.add_model_specific_args(parser)
    parser._action_groups[1].title = 'Model options'

    # general options
    parser2 = parser.add_argument_group(title= "General options")
    parser2.add_argument("--seed", type=int, default=3, help="Training seed.")
    parser2.add_argument('--test_tube_log_path', default=settings.TEST_TUBE_LOG_PATH)

    # checkpointing options
    parser2 = parser.add_argument_group(title= "Checkpointing options")
    parser2.add_argument(
        "--save_top_k",
        default=1,
        type=int,
        help="The best k models according to the quantity monitored will be saved.",
    )
    parser2.add_argument(
        "--save_own_checkpoint",
        default=False,
        type=bool,
        help="Turn on saving own checkpoints.",
    )

    # Early Stopping
    parser2 = parser.add_argument_group(title= "Early Stopping options")
    parser2.add_argument(
        "--deactivate_earlystopping", default=False, type=bool, help="Deactivate early stopping."
    )
    parser2.add_argument(
        "--monitor", default="val_AUROC", type=str, help="Quantity to monitor."
    )
    parser2.add_argument(
        "--metric_mode",
        default="max",
        type=str,
        help="If we want to min/max the monitored quantity.",
        choices=["auto", "min", "max"],
    )
    parser2.add_argument(
        "--patience",
        default=10,
        type=int,
        help=(
            "Number of epochs with no improvement "
            "after which training will be stopped."
        ),
    )

    # Options for testing with specific checkpoint 
    parser2 = parser.add_argument_group(title= "Options for testing with specific checkpoint.")
    parser2.add_argument(
        "--perform_testing_with_checkpoint", default=False, type=bool, help="Perform testing with a specific checkpoint. Will deactivate Training procedure completely."
    )
    parser2.add_argument(
        "--testing_checkpoint", default=None, type=str, help="File path of checkpoint to be used for testing."
    )
    parser2.add_argument(
        "--testing_mlflow_study_name", default="ppi-prediction-testing", type=str, help="Name of the mlflow study to be used for testing."
    )

    # Options for testing with best checkpoint 
    parser2 = parser.add_argument_group(title= "Options for testing with best checkpoint (identified by pytorch lightning)")
    parser2.add_argument(
        "--perform_testing_with_best_checkpoint", default=False, type=bool, help="Perform testing with best checkpoint. This will train first and then do the testing."
    )

    # Options for prediction with a certain checkpoint 
    parser2 = parser.add_argument_group(title= "Prediction options.")
    parser2.add_argument(
        "--perform_prediction", default=False, type=bool, help="Perform prediction with a specific checkpoint. Will deactivate training procedure completely."
    )
    parser2.add_argument(
        "--prediction_checkpoint", default=None, type=str, help="File path of checkpoint to be used for prediction."
    )
    parser2.add_argument(
        "--prediction_output_file", default=settings.BASE_DATA_DIR + "/generated/ml/predict.txt", type=str, help="File path of output to be used to save prediction results."
    )

    # Hack: set default options from outside
    # TODO: probably better to inherit the trainer class and set the defaults there, if that works
    idx = [a.dest for a in parser._actions].index('gpus')
    parser._actions[idx].default = None
    idx = [a.dest for a in parser._actions].index('checkpoint_callback')
    parser._actions[idx].default = False
    idx = [a.dest for a in parser._actions].index('accelerator')
    parser._actions[idx].default = "ddp"

    idx = [a.dest for a in parser._actions].index('limit_train_batches')
    parser._actions[idx].default = 1.0
    idx = [a.dest for a in parser._actions].index('limit_val_batches')
    parser._actions[idx].default = 1.0
    idx = [a.dest for a in parser._actions].index('limit_test_batches')
    parser._actions[idx].default = 1.0

    idx = [a.dest for a in parser._actions].index('accumulate_grad_batches')
    parser._actions[idx].default = 64
    idx = [a.dest for a in parser._actions].index('max_epochs')
    parser._actions[idx].default = 20
    idx = [a.dest for a in parser._actions].index('min_epochs')
    parser._actions[idx].default = 1

    idx = [a.dest for a in parser._actions].index('val_check_interval')
    parser._actions[idx].default = 0.5

    idx = [a.dest for a in parser._actions].index('deterministic')
    parser._actions[idx].default = True

    idx = [a.dest for a in parser._actions].index('weights_summary')
    parser._actions[idx].default = None

    idx = [a.dest for a in parser._actions].index('progress_bar_refresh_rate')
    parser._actions[idx].default = 10

    return parser

def prepare_params():
    
    logger = npe_ppi_logger.get_custom_logger(name=__name__)

    logger.info("Starting parsing arguments...")
    parser: ProtBertPPIArgParser = generate_parser()
    params: TTNamespace = parser.parse_args()
    logger.info("Finishing parsing arguments.")

    setattr(params, 'local_logger', logger)
    return params

def main(params: TTNamespace):
    """
    Main program that prepares the module, the trainer, the callbacks, the logger

    Args:
        params (TTNamespace): All kind of parameters to setup trainer and the module

    """

    if not params:
        params = prepare_params()

    # setup logger
    logger = params.local_logger
    if params.logger == True:
        params.logger = npe_ppi_logger.get_mlflow_logger_for_PL("ppi-prediction-hp-opt")

    # set seeds
    seed_everything(params.seed)

    # INIT TRAINER -------
    # Init early callback
    early_stop_callback = EarlyStopping(
        monitor=params.monitor,
        min_delta=0.0,
        patience=params.patience,
        verbose=True,
        mode=params.metric_mode,
    )

    # Save checkpoints
    ckpt_path = os.path.join(
        settings.MLFLOW_ARTIFACTS_DIR,
        #params.logger.name,
        #f"version_{params.logger.version}",
        "checkpoints",
    )

    params.callbacks = []    
    params.callbacks.append(LearningRateMonitor(logging_interval='step'))
    if params.deactivate_earlystopping == False:
        params.callbacks.append(early_stop_callback)

    # Add trainer params from outside: https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-in-python-scripts
    trainer = Trainer.from_argparse_args(params)

    # INIT LIGHTNING MODEL
    model = ProtBertPPIModel(params)
    global_rank = model.global_rank
    
    # EXECUTE TRAINING
    if params.perform_testing_with_checkpoint == False and params.perform_prediction == False:
        if global_rank == 0:
            logger.info("Starting training.")
    
        trainer.fit(model)
        trainer.save_checkpoint(settings.BASE_MODELS_DIR + "/vp1_ppi_model.ckpt")
    
        if global_rank == 0:
            logger.info("Finishing training.")

    # TESTING with specific path ###########
    if hasattr(params, "perform_testing_with_checkpoint") and params.perform_testing_with_checkpoint == True:
        checkpoint_path = params.testing_checkpoint
        if global_rank == 0:
            logger.info("Starting testing with the specific path checkpoint.")
            logger.info("Loading model with checkpoint: %s", checkpoint_path)

        model = model.load_from_checkpoint(checkpoint_path)
        model.eval()

        model.hparams.test_csv = params.test_csv
        model.hparams.gpus = params.gpus
        result = trainer.test(model)

        if global_rank == 0:
            logger.info("Results: %s", result)
            logger.info("Finishing testing with the best model.")
    
    # PREDICTION with specific path ###########
    if hasattr(params, "perform_prediction") and params.perform_prediction == True:
        checkpoint_path = params.prediction_checkpoint
        if global_rank == 0:
            logger.info("Starting predicting with the current model.")
            logger.info("Loading model with checkpoint: %s", checkpoint_path)

        model = model.load_from_checkpoint(checkpoint_path)
        model.eval()

        model.hparams.predict_csv = params.predict_csv
        model.hparams.per_device_predict_batch_size = params.per_device_predict_batch_size
        predictions = trainer.predict(model)

        import pandas as pd
        results = pd.DataFrame()
        for prediction in predictions:
            results = results.append(pd.DataFrame.from_dict(prediction), ignore_index=True)

        logger.info("Writing predictions in file: %s", params.prediction_output_file)
        results = results.rename(columns={'seqA': "receptor_protein_sequence", "seqB": "capsid_protein_sequence"})
        results.to_csv(params.prediction_output_file, sep="\t", index=False)

        logger.info("Model predictions: %s", results)

# %%
if __name__ == '__main__':
    params = prepare_params()
    logger = params.local_logger
    main(params)
