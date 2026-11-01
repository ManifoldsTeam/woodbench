"""Head/backbone parameter groups used for differential learning rates."""

import pytest
import timm

from src.train.optimizer import create_optimizer, split_head_parameters

# Share of parameters the historical substring rule puts in the head group
# (17 classes, timm defaults), against the true classifier size.
CASES = {
    "convnext_tiny": 13073,
    "swin_tiny_patch4_window7_224": 13073,
    "mobilenetv3_large_100": 21777,
    "tf_efficientnet_b3": 26129,
    "resnet50": 34833,
}


@pytest.mark.parametrize("name,classifier", CASES.items())
def test_classifier_selector_uses_only_final_classifier(name, classifier):
    model = timm.create_model(name, pretrained=False, num_classes=17)
    head, backbone = split_head_parameters(model, "classifier")
    assert sum(p.numel() for p in head) == classifier
    assert sum(p.numel() for p in head) + sum(p.numel() for p in backbone) == sum(
        p.numel() for p in model.parameters()
    )
    optimizer = create_optimizer(
        model, lr=1e-4, backbone_lr=1e-5, head_selector="classifier"
    )
    groups = {group["name"]: group for group in optimizer.param_groups}
    assert groups["head"]["lr"] == 1e-4 and groups["backbone"]["lr"] == 1e-5
    assert sum(p.numel() for p in groups["head"]["params"]) == classifier


def test_substring_default_is_unchanged_for_earlier_runs():
    model = timm.create_model("convnext_tiny", pretrained=False, num_classes=17)
    head, _ = split_head_parameters(model)
    # mlp.fc1/fc2 of every block match "fc": most of ConvNeXt is "head".
    assert sum(p.numel() for p in head) > 0.9 * sum(p.numel() for p in model.parameters())
    default = create_optimizer(model, lr=1e-4, backbone_lr=1e-5)
    assert sum(p.numel() for p in default.param_groups[0]["params"]) == sum(
        p.numel() for p in head
    )


def test_equal_learning_rates_use_one_group():
    model = timm.create_model("resnet50", pretrained=False, num_classes=17)
    optimizer = create_optimizer(model, lr=1e-4, backbone_lr=1e-4, head_selector="classifier")
    assert len(optimizer.param_groups) == 1


def test_unknown_selector_is_rejected():
    model = timm.create_model("resnet50", pretrained=False, num_classes=17)
    with pytest.raises(ValueError, match="head_selector"):
        split_head_parameters(model, "names")
