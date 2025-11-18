from model_training.rnn_trainer import BrainToTextDecoder_Trainer
from omegaconf import OmegaConf


def train_morphemes_model():
    args = OmegaConf.load("brain_to_text/model_training/training_args.yaml")
    trainer = BrainToTextDecoder_Trainer(args)
    metrics = trainer.train()


if __name__ == "__main__":
    train_morphemes_model()
