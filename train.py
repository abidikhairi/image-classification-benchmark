import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from project.lookup import models, datasets


def main(args):
    os.environ['DATA_DIR'] = './data'

    data, num_classes = datasets[args.dataset]
    model = models[args.model](num_classes=num_classes)
    experiment_name = f'{args.dataset}_{args.model}'

    logger = WandbLogger(name=experiment_name, project='image-classification-benchmarks')
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=10, logger=logger)

    trainer.fit(model, datamodule=data)
    trainer.test(model, datamodule=data)

    trainer.save_checkpoint('checkpoints/{}'.format(experiment_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, required=True, choices=models.keys(), help='model to use')
    parser.add_argument('--dataset', type=str, required=True, choices=datasets.keys(), help='dataset to use')

    args = parser.parse_args()
    main(args)

