import torch
import logging

from functional_attacks.attacks import AffineTransforms, ColorTransforms, Delta, \
    SpatialFlowFields, ThinPlateSplines, ConvolutionalKernel, AdjustBrightnessContrast, \
    AdjustGamma, AdjustHueSaturation, AdjustSharpness, GaussianBlur
from functional_attacks.loss import CWLossF6, l2_grid_smoothness

logger = logging.getLogger(__name__)


@torch.no_grad()
def validation(classifier, x, x_adv, y):
    """
    A method to run during each iteration of the pgd_attack_with_perturbation method
    :param classifier: the model that is being attacked
    :param x: original batch of images
    :param x_adv: perturbed batch of images
    :param y: ground truth values for the original images
    :return: success-rate of the attack
    """
    y_pred = classifier(x_adv).topk(1)[1].cpu().numpy()
    y = y.cpu().numpy()
    attack_successful = [y[i] != y_pred[i] for i in range(x.shape[0])]
    linf_metric = torch.norm(x.view(x.shape[0], -1) - x_adv.view(x.shape[0], -1), p=float('inf'), dim=1).cpu().numpy()
    logger.debug(f"linf: {linf_metric}")
    logger.debug(f"pgd_attack_with_perturbation state: (idx, success, gt, linf) : "
                 f"{list(zip(list(range(y.shape[0])), attack_successful, y, linf_metric))}")
    return (100 * sum(attack_successful) / x.shape[0]).item()


def pgd_attack_linf(perturbation, classifier, examples, labels, num_iterations=20, keep_best=True,
                    validator=validation, l2_smoothing_loss=False, l2_smooth_loss_weight=1e-4, early_stopping=True):
    """
    A method to combine global and local adversarial attacks
    :param perturbation: nn.sequential model of all the transforms to be applied
    :param classifier: Classifier network to attack
    :param examples: N x C x H X W batch
    :param labels: ground truth of the batch N x 1
    :param num_iterations: max num of iterations to run the PGD based iterative attack
    :param keep_best: save and return the perturbed images with the least loss
    :param validator: A method that takes arguments (classifier, x, x_adv, y)
                      and return a value from [0.0, 100.0] indicating the success rate of the attack
    :param l2_smoothing_loss: If true applies the lsmooth loss from the paper https://arxiv.org/pdf/1801.02612.pdf
                              to both ReColor and SpatialFlowField attacks
    :param l2_smooth_loss_weight: weight for the l2_smooth_loss
    :param early_stopping: if the loss doesn't improve stop the iterations
    :return:
    """
    num_examples = examples.shape[0]
    best_perturbed_examples = torch.empty(*examples.shape, dtype=examples.dtype,
                                          device=examples.device, requires_grad=False).copy_(examples)
    best_loss_per_example = [float('inf') for _ in range(num_examples)]
    loss_fn = CWLossF6(classifier)

    if validator is not None:
        validation = validator(classifier, examples, best_perturbed_examples, labels)
        logger.info("iter: %d attack-success-rate: %.3f", 0, validation)

    perturbed_examples = None
    previous_loss = torch.tensor(0.0)
    for iter_no in range(num_iterations):
        # unravel the sequential model and separate out linf from deformation methods
        # apply linf based perturbations and clip the combined perturbations
        perturbed_examples = perturbation(examples)

        # total loss
        total_loss_per_example = torch.zeros(size=(examples.shape[0],), requires_grad=True, device=examples.device)

        # apply CW Linf loss without constraint to minimize the linf budget
        cw_f6_loss_per_example = loss_fn(perturbed_examples, labels)
        total_loss_per_example = total_loss_per_example + cw_f6_loss_per_example
        logger.info("cw_f6 iter:%d loss: %.3f", iter_no, cw_f6_loss_per_example.sum().item())

        # smoothness loss for params of ReColor and SpatialFlowField
        if l2_smoothing_loss:
            for module in perturbation.modules():
                if type(module) in (ColorTransforms, SpatialFlowFields):
                    smooth_loss_per_example = l2_grid_smoothness(
                        module.xform_params - module.identity_params) * l2_smooth_loss_weight
                    total_loss_per_example = total_loss_per_example + smooth_loss_per_example
                    logger.info("[%s] smooth loss iter:%d loss: %.3f", type(module).__name__, iter_no,
                                 smooth_loss_per_example.sum().item())

        # clear the gradients of the perturbation model using the optimizer
        perturbation.zero_grad()
        # this is not exactly needed but a good practice
        classifier.zero_grad()

        # backpropagate the loss
        total_loss_per_example.sum().backward()
        logger.info("total loss iter: %d, loss: %.3f", iter_no, total_loss_per_example.sum().item())

        with torch.no_grad():
            # update the weights of the perturbation network
            for module in perturbation.modules():
                if type(module) in (AdjustBrightnessContrast, AdjustGamma, AdjustHueSaturation, AdjustSharpness,
                                      GaussianBlur, AffineTransforms, Delta, ColorTransforms, ConvolutionalKernel,
                                      SpatialFlowFields, ThinPlateSplines):
                    module.update_and_project_params()
                else:
                    logger.debug(f"not updating {type(module).__name__}")
                    continue

        # bookkeeping to track the best perturbed examples
        if keep_best:
            for i, el in enumerate(total_loss_per_example):
                cur_best_loss = best_loss_per_example[i]
                if cur_best_loss > float(el):
                    best_loss_per_example[i] = float(el)
                    best_perturbed_examples[i].copy_(perturbed_examples[i].data)

        # If validation is provided do early stopping
        if validator is not None:
            validation = validator(classifier, examples, best_perturbed_examples, labels)
            logger.info("iter: %d attack-success-rate: %.3f", iter_no + 1, validation)

        # early stopping
        if early_stopping and torch.allclose(total_loss_per_example.sum(), previous_loss):
            logger.warning("the loss didn't improve from previous iteration, stopping PGD optimization for the batch")
            break

        previous_loss = total_loss_per_example.sum()

    # clean up the accumulated gradients
    classifier.zero_grad()
    perturbation.zero_grad()
    if keep_best:
        return best_perturbed_examples
    return perturbed_examples.detach()
