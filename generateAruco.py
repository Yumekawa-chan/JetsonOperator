import cv2
from cv2 import aruco
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

aruco_dict_type = aruco.DICT_6X6_250
marker_size = 1600  # 1つのマーカーのサイズ (ピクセル)
num_markers = 48  # 生成するマーカーの数

A0_width, A0_height = 9933, 14043  # A0サイズ: 841 x 1189 mm (約300 DPI)

aruco_dict = aruco.getPredefinedDictionary(aruco_dict_type)

combined_image = np.ones((A0_height, A0_width, 3), dtype=np.uint8) * 255


spacing = 50

for i in range(num_markers):
    marker_image = aruco.generateImageMarker(aruco_dict, i, marker_size)
    
    x = (i % 6) * (marker_size + spacing) + spacing
    y = (i // 6) * (marker_size + spacing) + spacing + 50*3

    combined_image[y:y + marker_size, x:x + marker_size] = cv2.cvtColor(marker_image, cv2.COLOR_GRAY2BGR)



output_filename = 'aruco_markers_A0.pdf'
with PdfPages(output_filename) as pdf:
    plt.figure(figsize=(A0_width / 100, A0_height / 100), dpi=100)
    plt.imshow(combined_image)
    plt.axis('off')
    pdf.savefig(bbox_inches='tight', pad_inches=0)
    plt.close()