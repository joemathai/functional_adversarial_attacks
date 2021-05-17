import torch


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self, classifier):
        """
        :param classifier:
        """
        super().__init__()
        self.classifier = classifier
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, images, labels, **kwargs):
        classifier_out = self.classifier.forward(images)
        # pose this us a minimization objective for the perturbation network
        return -1.0 * self.loss_fn(classifier_out, labels)

