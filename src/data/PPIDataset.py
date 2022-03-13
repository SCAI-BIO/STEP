from src import settings
from pandas.core.frame import DataFrame
import re
from torchnlp.datasets.dataset import Dataset
import pandas as pd

class PPIDataset():
    """
    Loads the Dataset from the csv files passed to the parser.
    """
    def  __init__(self) -> None:
        return

    def collate_lists(self, seqA: list, seqB: list, labels: list, protA_ids: list, protB_ids: list, ncbi_geneA_id: list, ncbi_geneB_id: list) -> list:
        """ Converts each line into a dictionary. """
        collated_dataset = []
        for i in range(len(seqA)):
            collated_dataset.append({
                "seqA": str(seqA[i]), 
                "seqB": str(seqB[i]), 
                "label": str(labels[i]),
                "protA_ids": str(protA_ids[i]).split(","), 
                "protB_ids": str(protB_ids[i]).split(","), 
                "ncbi_geneA_id": str(ncbi_geneA_id[i]), 
                "ncbi_geneB_id": str(ncbi_geneB_id[i]), 
            })
        return collated_dataset

    def _retrieve_dataframe(self, path) -> DataFrame:
        column_names = [
            "class", 
            'sequenceA', 'sequenceB', 
            "protein_A_ids", "protein_B_ids", 
            "ncbi_gene_A_id", "ncbi_gene_B_id"
        ] 
        df: DataFrame = pd.read_csv(path, sep = "\t", names=column_names, header=0) #type:ignore
        return df

    def calculate_stat(self,path):
        df = self._retrieve_dataframe(path)
        self.nSamples_dic = df['class'].value_counts()

    def load_predict_dataset(self, path):
        column_names = [
            "interaction", "probability",
            'receptor_protein_id',
            "receptor_protein_label",
            'receptor_protein_name',
            'receptor_protein_sequence',
            'capsid_protein_sequence',
        ] 
        df: DataFrame = pd.read_csv(path, sep = "\t", names=column_names, header=0) #type:ignore

        interactions = list(df['interaction'])
        probabilities = list(df['probability'])
        receptor_protein_ids = list(df['receptor_protein_id'])
        receptor_protein_labels = list(df['receptor_protein_label'])
        receptor_protein_names = list(df['receptor_protein_name'])
        receptor_protein_seqs = list(df['receptor_protein_sequence'])
        capsid_protein_sequences = list(df['capsid_protein_sequence']) 

        # Make sure there is a space between every token, and map rarely amino acids
        receptor_protein_seqs = [" ".join("".join(sample.split())) for sample in receptor_protein_seqs]
        receptor_protein_seqs = [re.sub(r"[UZOB]", "X", sample) for sample in receptor_protein_seqs]
        
        capsid_protein_sequences = [" ".join("".join(sample.split())) for sample in capsid_protein_sequences]
        capsid_protein_sequences = [re.sub(r"[UZOB]", "X", sample) for sample in capsid_protein_sequences]

        assert len(receptor_protein_seqs) == len(interactions)
        assert len(capsid_protein_sequences) == len(interactions)
        assert len(receptor_protein_ids) == len(interactions)
        assert len(receptor_protein_names) == len(interactions)
        assert len(receptor_protein_labels) == len(interactions)
        assert len(probabilities) == len(interactions)

        collated_dataset = []
        for i in range(len(interactions)):
            collated_dataset.append({
                "label": str(interactions[i]),
                "probability": str(probabilities[i]), 
                "receptor_protein_id": str(receptor_protein_ids[i]), 
                "receptor_protein_label": str(receptor_protein_labels[i]), 
                "receptor_protein_name": str(receptor_protein_names[i]), 
                "seqA": str(receptor_protein_seqs[i]), 
                "seqB": str(capsid_protein_sequences[i]), 
            })

        return Dataset(collated_dataset)

    def load_dataset(self, path):
        df = self._retrieve_dataframe(path)

        labels = list(df['class'])
        seqA = list(df['sequenceA'])
        seqB = list(df['sequenceB'])
        protA_ids = list(df['protein_A_ids']) # TODO: Rename to protein_A_ids
        protB_ids = list(df['protein_B_ids']) # TODO: Rename to protein_A_ids
        ncbi_geneA_id = list(df['ncbi_gene_A_id'])
        ncbi_geneB_id = list(df['ncbi_gene_B_id'])

        # Make sure there is a space between every token, and map rarely amino acids
        seqA = [" ".join("".join(sample.split())) for sample in seqA]
        seqA = [re.sub(r"[UZOB]", "X", sample) for sample in seqA]
        
        seqB = [" ".join("".join(sample.split())) for sample in seqB]
        seqB = [re.sub(r"[UZOB]", "X", sample) for sample in seqB]

        assert len(seqA) == len(labels)
        assert len(seqB) == len(labels)
        assert len(protA_ids) == len(labels)
        assert len(protB_ids) == len(labels)
        
        return Dataset(self.collate_lists(seqA, seqB, labels, protA_ids, protB_ids, ncbi_geneA_id, ncbi_geneB_id))


