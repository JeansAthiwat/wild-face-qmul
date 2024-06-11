import os


class Config:
    def __init__(self):
        self.BASE_DIR = "/home/jeans/internship/wild-face-qmul"
        self.QMUL_DIR = os.path.join(self.BASE_DIR, "data/raw/QMUL-SurvFace")
        self.DATA_DIR = os.path.join(self.QMUL_DIR, "Face_Identification_Test_Set")
        self.GALLERY_DIR = os.path.join(self.DATA_DIR, "gallery")
        self.QUERY_DIR = os.path.join(self.DATA_DIR, "mated_probe")
        self.OUTPUT_DIR = os.path.join(self.BASE_DIR, "data/processed")
