import copy
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models as torch_models

from octopod.vision.helpers import _dense_block, _Identity


MODEL_CLASSES = {
    'resnet18':  torch_models.resnet18,
    'resnet34':  torch_models.resnet34,
    'resnet50': torch_models.resnet50,
    'resnet101': torch_models.resnet101,
    'resnet152': torch_models.resnet152,
    'wide_resnet50_2': torch_models.wide_resnet50_2,
    'wide_resnet101_2': torch_models.wide_resnet101_2
}


class CNNForMultiTaskClassification(nn.Module):
    """
    PyTorch image attribute model. This model allows you to load
    in some pretrained tasks in addition to creating new ones.

    Examples
    --------
    To instantiate a completely new instance of CNNForMultiTaskClassification
    and load the weights into this architecture you can set `pretrained_core` to True::

        model = CNNForMultiTaskClassification(
            new_task_dict=new_task_dict,
            load_pretrained_cnn = True
        )

        # DO SOME TRAINING

        model.save(SOME_FOLDER, SOME_MODEL_ID)

    To instantiate an instance of CNNForMultiTaskClassification that has layers for
    pretrained tasks and new tasks, you would do the following::

        model = CNNForMultiTaskClassification(
            pretrained_task_dict=pretrained_task_dict,
            new_task_dict=new_task_dict
        )

        model.load(SOME_FOLDER, SOME_MODEL_DICT)

        # DO SOME TRAINING

    Parameters
    ----------
    pretrained_task_dict: dict
        dictionary mapping each pretrained task to the number of labels it has
    new_task_dict: dict
        dictionary mapping each new task to the number of labels it has
    use_full_image: bool
        boolean indicating whether to include the full image as input to the core model.
    use_cropped_image: bool
        boolean indicating whether to include a cropped version of the image as input to the
        core model. If both this and use_full_image are set to true, the full and croppped
        images are sent independently through the core model, their respective outputs
        concatenated together before being processed by the dense layers.
    pretrained_core: boolean
        flag for whether or not to load in pretrained imagenet weights.
        useful for the first round of training before there are fine tuned weights
    """
    def __init__(
        self,
        model_arch='resnet50',
        pretrained_task_dict=None,
        new_task_dict=None,
        use_full_image=True,
        use_cropped_image=True,
        pretrained_core=False):

        super(CNNForMultiTaskClassification, self).__init__()

        self.core_model = MODEL_CLASSES[model_arch](pretrained=pretrained_core)
        self.core_model.fc = _Identity()

        self.use_full_image = use_full_image
        self.use_cropped_image = use_cropped_image
        assert self.use_full_image or self.use_cropped_image, \
        'Either use_full_image or use_cropped_image must be True'

        output_size_multiplier = 2 if self.use_cropped_image and self.use_full_image else 1

        self.dense_layers = nn.Sequential(
            _dense_block(2048*output_size_multiplier, 1024, 2e-3),
            _dense_block(1024, 512, 2e-3),
            _dense_block(512, 256, 2e-3),
        )

        if pretrained_task_dict is not None:
            pretrained_layers = {}
            for key, task_size in pretrained_task_dict.items():
                pretrained_layers[key] = nn.Linear(256, task_size)
            self.pretrained_classifiers = nn.ModuleDict(pretrained_layers)
        if new_task_dict is not None:
            new_layers = {}
            for key, task_size in new_task_dict.items():
                new_layers[key] = nn.Linear(256, task_size)
            self.new_classifiers = nn.ModuleDict(new_layers)

    def forward(self, x):
        """
        Defines forward pass for image model

        Parameters
        ----------
        x: dict of image tensors containing tensors for
        full and cropped images. the full image tensor
        has the key 'full_img' and the cropped tensor has
        the key 'crop_img'

        Returns
        ----------
        A dictionary mapping each task to its logits
        """
        if self.use_full_image and self.use_cropped_image:
            full_img = self.core_model(x['full_img']).squeeze()
            crop_img = self.core_model(x['crop_img']).squeeze()

            if x[next(iter(x))].shape[0] == 1:
                # if batch size is 1, or a single image, during inference
                dense_layer_input = torch.cat((full_img, crop_img), 0).unsqueeze(0)
            else:
                dense_layer_input = torch.cat((full_img, crop_img), 1)

        elif self.use_full_image:
            dense_layer_input = self.core_model(x['full_img']).squeeze()
        elif self.use_cropped_image:
            dense_layer_input = self.core_model(x['crop_img']).squeeze()

        dense_layer_output = self.dense_layers(dense_layer_input)

        logit_dict = {}
        if hasattr(self, 'pretrained_classifiers'):
            for key, classifier in self.pretrained_classifiers.items():
                logit_dict[key] = classifier(dense_layer_output)
        if hasattr(self, 'new_classifiers'):
            for key, classifier in self.new_classifiers.items():
                logit_dict[key] = classifier(dense_layer_output)

        return logit_dict

    def freeze_core(self):
        """Freeze all core model layers"""
        for param in self.core_model.parameters():
            param.requires_grad = False

    def freeze_dense(self):
        """Freeze all core model layers"""
        for param in self.dense_layers.parameters():
            param.requires_grad = False

    def freeze_all_pretrained(self):
        """Freeze pretrained classifier layers and core model layers"""
        self.freeze_core()
        self.freeze_dense()
        if hasattr(self, 'pretrained_classifiers'):
            for param in self.pretrained_classifiers.parameters():
                param.requires_grad = False
        else:
            print('There are no pretrained_classifier layers to be frozen.')

    def unfreeze_pretrained_classifiers(self):
        """Unfreeze pretrained classifier layers"""
        if hasattr(self, 'pretrained_classifiers'):
            for param in self.pretrained_classifiers.parameters():
                param.requires_grad = True
        else:
            print('There are no pretrained_classifier layers to be unfrozen.')

    def unfreeze_pretrained_classifiers_and_core(self):
        """Unfreeze pretrained classifiers and core model layers"""
        for param in self.core_model.parameters():
            param.requires_grad = True
        for param in self.dense_layers.parameters():
            param.requires_grad = True
        self.unfreeze_pretrained_classifiers()

    def save(self, folder, model_id):
        """
        Saves the model state dicts to a specific folder.
        Each part of the model is saved separately to allow for
        new classifiers to be added later.

        Note: if the model has `pretrained_classifiers` and `new_classifers`,
        they will be combined into the `pretrained_classifiers_dict`.

        Parameters
        ----------
        folder: str or Path
            place to store state dictionaries
        model_id: int
            unique id for this model

        Side Effects
        ------------
        saves three files:
            - folder / f'rcore_model_dict_{model_id}.pth'
            - folder / f'dense_layers_dict_{model_id}.pth'
            - folder / f'pretrained_classifiers_dict_{model_id}.pth'
        """
        if hasattr(self, 'pretrained_classifiers'):
            # PyTorch's update method isn't working because it doesn't think ModuleDict is a Mapping
            classifiers_to_save = copy.deepcopy(self.pretrained_classifiers)
            if hasattr(self, 'new_classifiers'):
                for key, module in self.new_classifiers.items():
                    classifiers_to_save[key] = module
        else:
            classifiers_to_save = copy.deepcopy(self.new_classifiers)

        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)

        torch.save(
            self.core_model.state_dict(),
            folder / f'core_model_dict_{model_id}.pth'
        )
        torch.save(
            self.dense_layers.state_dict(),
            folder / f'dense_layers_dict_{model_id}.pth'
        )

        torch.save(
            classifiers_to_save.state_dict(),
            folder / f'pretrained_classifiers_dict_{model_id}.pth'
        )

    def load(self, folder, model_id):
        """
        Loads the model state dicts from a specific folder.

        Parameters
        ----------
        folder: str or Path
            place where state dictionaries are stored
        model_id: int
            unique id for this model

        Side Effects
        ------------
        loads from three files:
            - folder / f'resnet_dict_{model_id}.pth'
            - folder / f'dense_layers_dict_{model_id}.pth'
            - folder / f'pretrained_classifiers_dict_{model_id}.pth'
        """
        folder = Path(folder)

        if torch.cuda.is_available():
            self.core_model.load_state_dict(torch.load(folder / f'core_model_dict_{model_id}.pth'))
            self.dense_layers.load_state_dict(
                torch.load(folder / f'dense_layers_dict_{model_id}.pth'))
            self.pretrained_classifiers.load_state_dict(
                torch.load(folder / f'pretrained_classifiers_dict_{model_id}.pth')
            )
        else:
            self.core_model.load_state_dict(
                torch.load(
                    folder / f'core_model_dict_{model_id}.pth',
                    map_location=lambda storage,
                    loc: storage
                )
            )
            self.dense_layers.load_state_dict(
                torch.load(
                    folder / f'dense_layers_dict_{model_id}.pth',
                    map_location=lambda storage,
                    loc: storage
                )
            )
            self.pretrained_classifiers.load_state_dict(
                torch.load(
                    folder / f'pretrained_classifiers_dict_{model_id}.pth',
                    map_location=lambda storage,
                    loc: storage
                )
            )

    def export(self, folder, model_id, model_name=None):
        """
        Exports the entire model state dict to a specific folder.
        Note: if the model has `pretrained_classifiers` and `new_classifiers`,
        they will be combined into the `pretrained_classifiers` attribute before being saved.

        Parameters
        ----------
        folder: str or Path
            place to store state dictionaries
        model_id: int
            unique id for this model
        model_name: str (defaults to None)
            Name to store model under, if None, will default to `multi_task_bert_{model_id}.pth`

        Side Effects
        ------------
        saves one file:
            - folder / model_name
        """
        if hasattr(self, 'new_classifiers'):
            hold_new_classifiers = copy.deepcopy(self.new_classifiers)
        else:
            hold_new_classifiers = None

        hold_pretrained_classifiers = None
        if not hasattr(self, 'pretrained_classifiers'):
            self.pretrained_classifiers = copy.deepcopy(self.new_classifiers)
        else:
            hold_pretrained_classifiers = copy.deepcopy(self.pretrained_classifiers)
            # PyTorch's update method isn't working because it doesn't think ModuleDict is a Mapping
            if hasattr(self, 'new_classifiers'):
                for key, module in self.new_classifiers.items():
                    self.pretrained_classifiers[key] = module
        if hasattr(self, 'new_classifiers'):
            del self.new_classifiers

        if model_name is None:
            model_name = f'multi_task_resnet_{model_id}.pth'

        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)

        torch.save(
            self.state_dict(),
            folder / model_name
        )
        if hold_pretrained_classifiers is not None:
            self.pretrained_classifiers = hold_pretrained_classifiers
        else:
            del self.pretrained_classifiers

        if hold_new_classifiers is not None:
            self.new_classifiers = hold_new_classifiers
