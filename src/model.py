from transformers import SegformerForSemanticSegmentation

def get_model(num_classes=19, id2label=None, label2id=None):
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-cityscapes-768-768",
        num_labels=num_classes,
        id2label=id2label, label2id=label2id,
        ignore_mismatched_sizes=True   # head yeniden yazılacak
    )
    return model
