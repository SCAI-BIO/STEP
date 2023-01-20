# %% Imports
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from torch.utils.data.dataloader import DataLoader
from modeling.ProtBertPPIModel import ProtBertPPIModel
from typing import List
import pandas as pd
from pandas.core.frame import DataFrame
from transformers import BertTokenizer
from pytorch_lightning import Trainer, seed_everything


import settings
import npe_ppi_logger
from data.VirHostNetDataset import VirHostNetData

logger = npe_ppi_logger.get_custom_logger(name=__name__)

def generate_parser():
    parser = ArgumentParser()

    parser.add_argument(
        "--train_csv",
        default=settings.BASE_DATA_DIR + "/generated/sarscov2/ml/all_data.txt",
        type=str,
        help="Path to the file containing the train data.",
    )
    parser.add_argument(
        "--predict_csv",
        default=settings.BASE_DATA_DIR + "/generated/sarscov2/ml/predict_omicron_spike_interactions_template.txt",
        type=str,
        help="Path to the file containing the inferencing data.",
    )
    parser.add_argument(
        "--perform_training", default=True, type=bool, help="Perform training."
    )
    parser.add_argument(
        "--prediction_checkpoint", default=None, type=str, help="File path of checkpoint to be used for prediction."
    )

    return parser

def prepare_params():
    
    logger = npe_ppi_logger.get_custom_logger(name=__name__)

    logger.info("Starting parsing arguments...")
    parser = generate_parser()
    params = parser.parse_args()
    logger.info("Finishing parsing arguments.")

    return params

# %% Predict
def main(params):

    if params.perform_training == True:
        logger.info("Starting training.")
    
        model_name = "Rostlab/prot_bert_bfd"
        target_col = 'class'
        seq_col_a = 'sequenceA'
        seq_col_b = 'sequenceB'
        max_len = 1536
        batch_size = 8
        seed = 42
        seed_everything(seed)
        
        model_params = {}
        # TODO: update params from paper
        model_params["encoder_learning_rate"] = 5e-06
        model_params["warmup_steps"] = 200
        model_params["max_epochs"] = 20
        model_params["min_epochs"] = 5
        model = ProtBertPPIModel(model_params)
        
        trainer = Trainer(
            gpus=[0],
            max_epochs = model_params["max_epochs"], 
            # callbacks=callbacks, 
            # checkpoint_callback=checkpoint_callback,
            progress_bar_refresh_rate=5,
            num_sanity_val_steps=0,
            #logger = npe_ppi_logger.get_mlflow_logger_for_PL(trial.study.study_name)
        )

        tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False, local_files_only=True)
        # %% Read dataset
        data: DataFrame = pd.read_csv(params.train_csv, sep = "\t", header=0)
        dataset = VirHostNetData(data, 
            tokenizer = tokenizer, 
            max_len = max_len, 
            seq_col_a = seq_col_a, 
            seq_col_b = seq_col_b, 
            target_col = target_col
        )
        train_loader = DataLoader(dataset, batch_size= batch_size, num_workers = 8) # type:ignore
        trainer.fit(model, train_loader)
        trainer.save_checkpoint(settings.BASE_MODELS_DIR + "/sarscov2_ppi_model.ckpt")
        
        logger.info("Finishing training.")

    else:

        df_to_output = pd.read_csv(params.predict_csv, sep="\t", header=0)

        # Load model
        logger.info("Loading model.")

        model: ProtBertPPIModel = ProtBertPPIModel.load_from_checkpoint(
            params.prediction_checkpoint, 
        )

        # Predict
        logger.info("Loading dataset.")
        dataset = VirHostNetData(
            df=df_to_output, 
            tokenizer=model.tokenizer, 
            max_len=1536, 
            seq_col_a="receptor_protein_sequence", 
            seq_col_b="spike_protein_sequence", 
            target_col=False
        )
            
        logger.info("Predicting.")
        trainer = Trainer(gpus=[0], deterministic=True)
        predict_dataloader = DataLoader(dataset, num_workers=8) #type: ignore
        predictions = trainer.predict(model=model, dataloaders=predict_dataloader, return_predictions=True)
        for ix in range(len(predictions)): # type:ignore
            score = predictions[ix]['probability'][0] # type:ignore
            logger.info("For entry %s, we found a score of %s", ix, score)
            df_to_output.at[ix,"score"] = score
            
        # Save results
        results_file = settings.BASE_DATA_DIR + "/generated/sarscov2/ml/predict_omicron_spike_interactions.txt"
        df_to_output.to_csv(results_file, sep="\t", index=False)

if __name__ == '__main__':
    params = prepare_params()
    main(params)
