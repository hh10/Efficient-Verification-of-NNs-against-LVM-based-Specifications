import unittest
from ddt import ddt, data
from glob import glob

from utils import load_params, load_model
from train import setup_and_train
from verify import setup_and_verify
from model import VaeClassifier


@ddt
class Tests(unittest.TestCase):

    @data(*glob("sample_configs/*.txt"))
    def test_load_params(self, config_file):
        params = load_params(config_file)
        self.assertTrue(all(key in params for key in ['desc', 'notes']))
        # check for dparams load (stakeholder: dataloader.py/CustomDataloader class)
        self.assertTrue(all(key in params['dataset'] for key in ['dataset', 'input_shape', 'batch_size', 'data_balance_method', 'classes', 'conditional']))
        # check for mparams load (stakeholder: model.py)
        self.assertTrue(all(key in params['model'] for key in ['source', 'fdn_args', 'enc_args', 'dataset', 'model_path', 'classifier_path', 'latent_dim']))
        self.assertTrue(all(key in params['training'] for key in ['num_epochs', 'lr', 'train_cla', 'GAN_start_training_epochs',
                                                                  'only_vae_training_epochs', 'only_cla_training_epochs',
                                                                  'loss_conditional_weight', 'loss_recons_weight', 'loss_kl_weight']))

    def test_end2end_train_verify(self, config_file):
        print(f"\n\n\n#### TESTING {config_file} ####")
        # TRAINING RUN
        _, trained_model_path, params = setup_and_train(config_file, dry_run=True)
        dataset_name = params['dataset']['dataset']
        args = {"model_path": trained_model_path, "save_images": True, "num_test_images": 5, "test_attribute": '', "target_attributes": [], "flip_head": False}
        if dataset_name == "CelebA":
            args["test_attribute"] = "Eyeglasses"
        elif dataset_name in ['Objects10_3Dpose']:
            args["test_attribute"] = "dist_8__elev_3__azim_0"
            args["target_attributes"] = ["dist_14__elev_3__azim_0", "dist_8__elev_6__azim_0"]
        # VERIFICATION RUN
        # setup_and_verify(args)

    def test_pretrained_cla_load(self, config_file):
        params, _ = load_params(config_file)
        self.assertFalse(params['training']['train_vae'])
        self.assertEqual(params['model']['enc_args']['linear_layers'], 1)

        _, trained_cla_path, params = setup_and_train(params, dry_run=True)
        self.assertIsNone(params['model']['model_path'])
        self.assertEqual(params['model']['classifier_path'], trained_cla_path)

        params['training']['train_vae'] = True
        params['model']['train_vae'] = True
        params['model']['enc_args']['linear_layers'] = 3
        vae_cla = VaeClassifier(params['model'], device="cpu", attributes=[])
        load_model(vae_cla, params['model'], device="cpu")


if __name__ == '__main__':
    # unittest.main()
    test_runner = Tests()
    for config_file in glob("sample_configs/*.txt"):
        test_runner.test_end2end_train_verify(config_file)
    test_runner.test_pretrained_cla_load("sample_configs/CelebA_mobilenet_v2.txt")
