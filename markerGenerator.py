import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

aruco_dict_type = cv2.aruco.DICT_6X6_250
marker_size_mm = 350
paper_width_mm = 594
paper_height_mm = 420

dpi = 300
margin_x_px = int(((paper_width_mm - marker_size_mm) / 2) / 25.4 * dpi)
margin_y_px = int(((paper_height_mm - marker_size_mm) / 2) / 25.4 * dpi)

aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)

def generateImageMarker(marker_id):
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, int(marker_size_mm / 25.4 * dpi))

    marker_img_3ch = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)

    paper_size_px = (int(paper_width_mm / 25.4 * dpi), int(paper_height_mm / 25.4 * dpi))
    full_img = 255 * np.ones((paper_size_px[1], paper_size_px[0], 3), dtype=np.uint8)

    start_x = margin_x_px
    start_y = margin_y_px
    end_x = start_x + marker_img_3ch.shape[1]
    end_y = start_y + marker_img_3ch.shape[0]
    full_img[start_y:end_y, start_x:end_x] = marker_img_3ch

    fig, ax = plt.subplots(figsize=(paper_width_mm/25.4, paper_height_mm/25.4), dpi=dpi)
    ax.imshow(full_img, aspect='equal')
    ax.axis('off')

    pdf_filename = f'marker/aruco_marker_{marker_id}.pdf'
    with PdfPages(pdf_filename) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)  

def main():
    for i in range(6):
        generateImageMarker(i)
    
if __name__ == '__main__':
    main()