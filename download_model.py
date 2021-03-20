from google_drive_download import download_file_from_google_drive


def download_models():
    paths = ["models/breast-idc-model-1"]
    # breast cancer IDC
    download_file_from_google_drive("1D51jhAm4Rub6h3oLrBvpoOWV7V84PE-a", paths[0] + ".pth")

    return paths
