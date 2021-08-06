import torch
import logging

from functional_attacks.attacks import AffineTransforms, RotationTranslationTransforms, IndependentChannelColorTransforms, SWIRColorTransforms, ColorTransforms, Delta, \
    SpatialFlowFields, ThinPlateSplines, ConvolutionalKernel, AdjustBrightnessContrast, \
    AdjustGamma, AdjustHueSaturation, AdjustSharpness, GaussianBlur
from functional_attacks.loss import CWLossF6, l2_grid_smoothness, CrossEntropyLoss

logger = logging.getLogger(__name__)


@torch.no_grad()
def validation(classifier, x, x_adv, y, norm=False):
    """
    A method to run during each iteration of the pgd_attack_with_perturbation method
    :param classifier: the model that is being attacked
    :param x: original batch of images
    :param x_adv: perturbed batch of images
    :param y: ground truth values for the original images
    :return: success-rate of the attack
    """
    attack_success_rate = (100 * (classifier(x_adv).topk(1)[1] != y.view(-1, 1)).sum() / x_adv.shape[0]).item()
    if norm:
        linf_metric = torch.norm(x.reshape(x.shape[0], -1) - x_adv.reshape(x_adv.shape[0], -1), p=float('inf'), dim=1).item()
        logger.debug(f"linf: {linf_metric}")
        logger.debug(f"pgd_attack_with_perturbation state: (idx, success, gt, linf) : "
                     f"{list(zip(list(range(y.shape[0])), attack_successful, y, linf_metric))}")
        return attack_success_rate, linf_metric
    return attack_success_rate


def pgd_attack_linf(perturbation, classifier, examples, labels, num_iterations=20, keep_best=True,
                    validator=validation, l2_smoothing_loss=False, l2_smooth_loss_weight=1e-4,
                    loss_fn_type='cw_f6'):
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
    :param loss_fn_type: the loss to use for adversarial optimization
    :return:
    """
    if torch.lt(examples, 0).any() or torch.gt(examples, 1).any():
        logger.error("out of range values found :: range of values in batch (%f, %f)", examples.min(), examples.max())
        raise RuntimeError('negative values found in pgd_linf function')
    
    num_examples = examples.shape[0]
    if loss_fn_type == 'cw_f6':
        loss_fn = CWLossF6(classifier)
    elif loss_fn_type == "xent":
        loss_fn = CrossEntropyLoss(classifier)
    else:
        raise RuntimeError(f"{loss_fn_type} is not defined for pgd_attack_linf method")

    with torch.no_grad():
        initial_predictions = torch.argmax(classifier(examples), dim=1)
        validation = validator(classifier, examples, examples, labels)
        logger.info("initial attack-success-rate: %.3f", validation)
        # initialize the best loss based on un-attacked model
        loss_per_example = loss_fn(examples, labels)
        best_loss_per_example = [float(loss) for loss in loss_per_example]
        best_perturbed_examples = torch.empty(*examples.shape, dtype=examples.dtype,
                                          device=examples.device, requires_grad=False).copy_(examples)
        
    for iter_no in range(num_iterations):
        perturbed_examples = perturbation(examples)
        total_loss_per_example = loss_fn(perturbed_examples, labels)

        # note: the grid loss needs to be made for each example (not the combined norm for adding to total loss)
        # smoothness loss for params of ReColor and SpatialFlowField
        # if l2_smoothing_loss:
        #     for module in perturbation.modules():
        #         if type(module) in (ColorTransforms, SpatialFlowFields):
        #             smooth_loss_per_example = l2_grid_smoothness(
        #                 module.xform_params - module.identity_params) * l2_smooth_loss_weight
        #             total_loss_per_example = total_loss_per_example + smooth_loss_per_example
        #             logger.info("[%s] smooth loss iter:%d loss: %.3f", type(module).__name__, iter_no,
        #                          smooth_loss_per_example.sum().item())

        # clear the gradients of the perturbation model using the optimizer
        perturbation.zero_grad()
        # this is not exactly needed but a good practice
        classifier.zero_grad()
        # backpropagate the loss
        total_loss_per_example.sum().backward()

        log_msg = f"iter: {iter_no}, cur_loss: {total_loss_per_example.sum().item():.4f} "

        with torch.no_grad():
            # update the weights of the perturbation network
            for module in perturbation.modules():
                if type(module) in (AdjustBrightnessContrast, AdjustGamma, AdjustHueSaturation, AdjustSharpness,
                                    GaussianBlur, AffineTransforms, RotationTranslationTransforms,
                                    Delta, IndependentChannelColorTransforms, SWIRColorTransforms,
                                    ColorTransforms, ConvolutionalKernel,
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
                        logger.debug(f"best loss for idx:{i} found in iter:{iter_no} = {float(el)}")
                        best_loss_per_example[i] = float(el)
                        best_perturbed_examples[i].copy_(perturbed_examples[i].data)

                attack_success_rate = (100 * (classifier(best_perturbed_examples).topk(1)[1] != labels.view(-1, 1)).sum() / num_examples).item()
                log_msg += f"best_loss: {loss_fn(best_perturbed_examples, labels).sum().item():.4f} "\
                           f"attack_success_rate:{attack_success_rate:.2f}"
        
        logger.info(log_msg)

    # clean up the accumulated gradients
    classifier.zero_grad()
    perturbation.zero_grad()
    if keep_best:
        return best_perturbed_examples
    return perturbed_examples.detach()
