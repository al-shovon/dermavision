"""The seven HAM10000 lesion classes, with display names and risk groups."""

CLASS_INFO = {
    "akiec": {"name": "Actinic keratoses / intraepithelial carcinoma", "group": "pre-malignant"},
    "bcc": {"name": "Basal cell carcinoma", "group": "malignant"},
    "bkl": {"name": "Benign keratosis-like lesions", "group": "benign"},
    "df": {"name": "Dermatofibroma", "group": "benign"},
    "mel": {"name": "Melanoma", "group": "malignant"},
    "nv": {"name": "Melanocytic nevi", "group": "benign"},
    "vasc": {"name": "Vascular lesions", "group": "benign"},
}

DISCLAIMER = (
    "DermaVision AI is a learning and screening-support tool, not a medical device, "
    "and it cannot diagnose skin cancer. Please see a doctor about any lesion that "
    "changes, bleeds or worries you."
)
