import torch


class CWLossF6(torch.nn.Module):
    def __init__(self, classifier, kappa=0.01):
        """
        F6 loss from https://arxiv.org/pdf/1608.04644.pdf
        :param classifier:
        :param kappa: confidence
        """
        super().__init__()
        self.classifier = classifier
        self.kappa = kappa

    def forward(self, images, labels, **kwargs):
        classifier_out = self.classifier.forward(images)
        # get target logits
        target_logits = torch.gather(classifier_out, 1, labels.view(-1, 1))
        # get largest non-target logits
        max_2_logits, argmax_2_logits = torch.topk(classifier_out, 2, dim=1)
        top_max_logits, second_max_logits = max_2_logits.chunk(2, dim=1)
        top_argmax, _ = argmax_2_logits.chunk(2, dim=1)
        targets_eq_max = top_argmax.squeeze().eq(labels).float().view(-1, 1)
        targets_ne_max = top_argmax.squeeze().ne(labels).float().view(-1, 1)
        max_other_logits = targets_eq_max * second_max_logits + targets_ne_max * top_max_logits
        f6 = torch.clamp(target_logits - max_other_logits, min=-1.0 * self.kappa)
        return f6.squeeze()