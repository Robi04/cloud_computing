import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def get_model_instance_segmentation(num_classes):
    # Charger un modèle pré-entraîné pour la segmentation d'instance
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # Obtenir le nombre de canaux d'entrée pour le classificateur
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Remplacer la tête de classificateur pré-entraîné avec une nouvelle
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Obtenir le nombre de canaux d'entrée pour le masque classificateur
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    
    # Et remplacer la tête de prédiction de masque avec une nouvelle
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # Réinitialiser les gradients
        optimizer.zero_grad()
        # Calculer les gradients
        losses.backward()
        # Mettre à jour les paramètres
        optimizer.step()

# Remplacer ceci par votre propre code de test
def evaluate(model, data_loader, device):
    model.eval()
    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        # ici, vous pourriez implémenter une évaluation de votre modèle

if __name__ == "__main__":
    # Définir le device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Notre dataset a deux classes uniquement - fond et votre classe d'objet
    num_classes = 2

    # Utiliser notre fonction helper pour obtenir le modèle
    model = get_model_instance_segmentation(num_classes).to(device)

    # Définir l'optimiseur
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Charger vos données
    data_loader = torch.utils.data.DataLoader('./train_data')
    data_loader_test = torch.utils.data.DataLoader('./train_data')

    # Et finalement, entraîner et évaluer le modèle
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch)
        evaluate(model, data_loader_test, device)
